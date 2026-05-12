"""T009 – Unit tests for device.py.

Validates explicit cpu/xpu contract, rejection of invalid strings, and
that no automatic fallback to cpu occurs when xpu is unavailable.

Tests must FAIL before src/device.py is implemented.
"""

import pytest
import torch
from unittest.mock import patch


def test_cpu_returns_cpu_device():
    from src.device import get_device

    d = get_device("cpu")
    assert d == torch.device("cpu")


def test_invalid_device_raises_value_error():
    from src.device import get_device

    with pytest.raises(ValueError, match="Unsupported device"):
        get_device("gpu")


def test_invalid_device_cuda_raises_value_error():
    from src.device import get_device

    with pytest.raises(ValueError, match="Unsupported device"):
        get_device("cuda")


def test_xpu_unavailable_raises_runtime_error_no_fallback():
    """When XPU is explicitly requested but unavailable, RuntimeError must be raised.
    No fallback to cpu should occur.
    """
    from src.device import get_device

    with patch("src.device._check_xpu_available", side_effect=RuntimeError("XPU unavailable")):
        with pytest.raises(RuntimeError, match="XPU"):
            get_device("xpu")


def test_xpu_available_returns_xpu_device():
    from src.device import get_device

    with patch("src.device._check_xpu_available"):  # no-op → available
        d = get_device("xpu")
    assert d == torch.device("xpu")
