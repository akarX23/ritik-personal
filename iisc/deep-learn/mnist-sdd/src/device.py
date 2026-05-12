"""Device selection and validation for the MNIST training pipeline.

Only 'cpu' and 'xpu' are valid device identifiers. No automatic fallback
is performed; an unavailable device raises RuntimeError immediately.
"""

from __future__ import annotations

import torch


SUPPORTED_DEVICES: tuple[str, ...] = ("cpu", "xpu")


def get_device(device_str: str) -> torch.device:
    """Return a :class:`torch.device` for *device_str*.

    Parameters
    ----------
    device_str:
        Must be ``"cpu"`` or ``"xpu"``.

    Raises
    ------
    ValueError
        If *device_str* is not in the supported set.
    RuntimeError
        If ``"xpu"`` is requested but the runtime reports it unavailable.
    """
    if device_str not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device '{device_str}'. "
            f"Choose one of: {', '.join(SUPPORTED_DEVICES)}"
        )

    if device_str == "xpu":
        _check_xpu_available()

    return torch.device(device_str)


def _check_xpu_available() -> None:
    """Raise :class:`RuntimeError` if XPU is not available at runtime.

    Uses the PyTorch XPU availability API; never falls back to CPU.
    """
    available = getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
    if not available:
        raise RuntimeError(
            "XPU device was explicitly requested but is not available on this "
            "system. Ensure the Intel XPU driver and PyTorch nightly XPU build "
            "are correctly installed, or re-run with '-d cpu' to use the CPU."
        )
