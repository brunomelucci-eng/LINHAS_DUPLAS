from __future__ import annotations

import math
import os

import pytest
import torch

from rowgraphnet.inference.engine import InferenceConfig
from rowgraphnet.resources import (
    DEFAULT_RESOURCE_FRACTION,
    apply_resource_caps,
    check_virtual_memory_safety,
    get_system_resource_status,
)


def test_default_resource_fraction_is_70_percent() -> None:
    assert DEFAULT_RESOURCE_FRACTION == 0.70


def test_system_resource_status_fields() -> None:
    status = get_system_resource_status(0.70)
    assert status.cpu_cores >= 1
    assert status.cpu_threads_allocated == max(1, math.floor(status.cpu_cores * 0.70))
    assert status.total_ram_gb > 0
    assert status.available_ram_gb > 0
    assert status.resource_fraction == 0.70


def test_apply_resource_caps_enforces_threads() -> None:
    status = apply_resource_caps(fraction=0.70)
    expected_threads = max(1, math.floor((os.cpu_count() or 1) * 0.70))
    assert status.cpu_threads_allocated == expected_threads
    assert torch.get_num_threads() == expected_threads


def test_apply_resource_caps_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="resource fraction"):
        apply_resource_caps(fraction=0.05)
    with pytest.raises(ValueError, match="resource fraction"):
        apply_resource_caps(fraction=1.5)


def test_inference_config_defaults_to_70_percent() -> None:
    config = InferenceConfig()
    assert config.resource_fraction == 0.70
    assert config.memory_budget_fraction == 0.70


def test_check_virtual_memory_safety_returns_list() -> None:
    warnings = check_virtual_memory_safety()
    assert isinstance(warnings, list)
