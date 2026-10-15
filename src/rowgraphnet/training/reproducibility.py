"""Explicit reproducibility controls for training and data-loader workers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class SeedState:
    """Summary of the global state configured by :func:`seed_everything`."""

    seed: int
    deterministic_algorithms: bool
    warn_only: bool
    cuda_seeded: bool


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in the uint32 range [0, 2**32)")
    return seed


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
) -> SeedState:
    """Seed Python, NumPy, CPU/CUDA Torch and deterministic kernels.

    This function does not depend on Lightning and is safe to call in a plain
    PyTorch program. ``CUBLAS_WORKSPACE_CONFIG`` is set before CUDA work begins
    so deterministic matrix multiplication can be honored when available.
    """

    seed = _validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_seeded = bool(torch.cuda.is_available())
    if cuda_seeded:
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    return SeedState(
        seed=seed,
        deterministic_algorithms=deterministic,
        warn_only=warn_only,
        cuda_seeded=cuda_seeded,
    )


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from Torch's per-worker initial seed."""

    if isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0:
        raise ValueError("worker_id must be a non-negative integer")
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seeded_generator(seed: int) -> torch.Generator:
    """Create the generator passed to a deterministic DataLoader."""

    generator = torch.Generator()
    generator.manual_seed(_validate_seed(seed))
    return generator


__all__ = ["SeedState", "seed_dataloader_worker", "seed_everything", "seeded_generator"]
