"""Shared PyTorch device selection for training and evaluation."""

from __future__ import annotations

import torch


DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def mps_is_usable() -> bool:
    """Probe MPS with a real allocation instead of trusting is_available()."""
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_built():
        return False
    try:
        torch.empty(1, device="mps")
        return True
    except (RuntimeError, OSError):
        return False


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit or automatic PyTorch device with useful errors."""
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {requested!r}; choose from {DEVICE_CHOICES}")

    cuda_available = torch.cuda.is_available()
    mps_available = mps_is_usable()

    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but PyTorch cannot see an NVIDIA CUDA device.")
    if requested == "mps" and not mps_available:
        raise RuntimeError(
            "MPS was requested, but a test tensor could not be allocated on the Apple GPU. "
            "Use an arm64 Python/PyTorch build on an Apple Silicon Mac, or pass --device cpu."
        )

    if requested != "auto":
        return torch.device(requested)
    if cuda_available:
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def device_status(device: torch.device) -> str:
    return (
        f"device={device} cuda_available={torch.cuda.is_available()} "
        f"mps_available={mps_is_usable()}"
    )
