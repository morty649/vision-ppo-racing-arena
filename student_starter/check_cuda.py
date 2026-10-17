"""Check whether PyTorch can use a local CUDA or Apple MPS GPU."""

from __future__ import annotations

import sys
import platform

import torch

from device import mps_is_usable


CUDA_TORCH_INSTALL = (
    r".\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall "
    r"torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
)


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"torch:  {torch.__version__}")
    print(f"cuda:   {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"mps built:      {torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False}")
    print(f"mps usable:     {mps_is_usable()}")
    if torch.cuda.is_available():
        print(f"gpu:    {torch.cuda.get_device_name(0)}")
        return 0

    if mps_is_usable():
        print("gpu:    Apple Metal (MPS)")
        return 0

    if platform.system() == "Darwin":
        print("\nPyTorch is installed, but a test tensor could not use Apple MPS.")
        print("Confirm this is an Apple Silicon Mac running an arm64 Python build.")
        return 1

    print("\nPyTorch is installed, but this environment cannot see CUDA.")
    print("On an RTX 4070 Windows machine, reinstall the CUDA PyTorch wheel:")
    print(f"\n{CUDA_TORCH_INSTALL}\n")
    print("Then run this check again from the student_starter folder:")
    print(r".\.venv\Scripts\python.exe check_cuda.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
