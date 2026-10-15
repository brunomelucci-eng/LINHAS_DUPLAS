"""System resource guardrails and default capacity limits (e.g., 70% cap)."""

from __future__ import annotations

import contextlib
import logging
import math
import os
from dataclasses import dataclass

import psutil
import torch

logger = logging.getLogger(__name__)

DEFAULT_RESOURCE_FRACTION = 0.70


@dataclass(frozen=True, slots=True)
class SystemResourceStatus:
    """Snapshot of host system resources and active limits."""

    cpu_cores: int
    cpu_threads_allocated: int
    total_ram_gb: float
    available_ram_gb: float
    total_swap_gb: float
    free_swap_gb: float
    cuda_available: bool
    cuda_device_name: str | None
    cuda_total_vram_gb: float | None
    cuda_vram_limit_gb: float | None
    resource_fraction: float


def get_system_resource_status(
    fraction: float = DEFAULT_RESOURCE_FRACTION,
    device: str | torch.device | None = None,
) -> SystemResourceStatus:
    """Inspect current hardware capacity and projected 70% allocations."""

    cores = os.cpu_count() or 1
    threads = max(1, math.floor(cores * fraction))
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()

    cuda_avail = torch.cuda.is_available()
    dev_name = None
    vram_gb = None
    vram_limit = None

    if cuda_avail and (device is None or str(device) not in {"cpu"}):
        with contextlib.suppress(Exception):
            dev_idx = (
                torch.cuda.current_device()
                if device is None or str(device) == "auto"
                else (
                    device.index
                    if isinstance(device, torch.device) and device.index is not None
                    else 0
                )
            )
            dev_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            vram_gb = round(props.total_memory / (1024**3), 2)
            vram_limit = round(vram_gb * fraction, 2)

    return SystemResourceStatus(
        cpu_cores=cores,
        cpu_threads_allocated=threads,
        total_ram_gb=round(vm.total / (1024**3), 2),
        available_ram_gb=round(vm.available / (1024**3), 2),
        total_swap_gb=round(sm.total / (1024**3), 2),
        free_swap_gb=round(sm.free / (1024**3), 2),
        cuda_available=cuda_avail,
        cuda_device_name=dev_name,
        cuda_total_vram_gb=vram_gb,
        cuda_vram_limit_gb=vram_limit,
        resource_fraction=fraction,
    )


def apply_resource_caps(
    fraction: float = DEFAULT_RESOURCE_FRACTION,
    device: str | torch.device | None = None,
) -> SystemResourceStatus:
    """Enforce a maximum fraction (default 70%) of CPU cores and GPU VRAM.

    This prevents PyTorch or background tasks from pegging 100% of host resources,
    preventing Windows kernel paging timeouts, GPU driver crashes, and unexpected reboots.
    """
    if not 0.1 <= fraction <= 1.0:
        raise ValueError(f"resource fraction must lie between 0.1 and 1.0; got {fraction}")

    cores = os.cpu_count() or 1
    allocated_threads = max(1, math.floor(cores * fraction))
    torch.set_num_threads(allocated_threads)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(allocated_threads)

    if torch.cuda.is_available() and (device is None or str(device) not in {"cpu"}):
        try:
            dev_idx = (
                torch.cuda.current_device()
                if device is None or str(device) == "auto"
                else (
                    device.index
                    if isinstance(device, torch.device) and device.index is not None
                    else 0
                )
            )
            # Instruct PyTorch caching allocator to not exceed 70% of total VRAM
            torch.cuda.set_per_process_memory_fraction(float(fraction), dev_idx)
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("Could not set CUDA memory fraction: %s", exc)

    return get_system_resource_status(fraction=fraction, device=device)


def check_virtual_memory_safety() -> list[str]:
    """Return warning messages if Windows virtual memory / swap configuration is risky."""

    warnings: list[str] = []
    sm = psutil.swap_memory()
    total_swap_gb = sm.total / (1024**3)
    if total_swap_gb < 8.0:
        warnings.append(
            f"Arquivo de paginação do Windows está com apenas {total_swap_gb:.1f} GB. "
            "Recomenda-se configurar a Memória Virtual do Windows para pelo menos 32 GB "
            "para evitar desligamento ou BSOD por esgotamento de commit."
        )
    return warnings
