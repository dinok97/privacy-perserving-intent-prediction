#!/usr/bin/env python3
"""
check_env.py — quick environment check for the data_collection_v2 pipeline.

Run this first. It installs nothing; it just reports whether the packages the
live collection script needs can be imported, so you catch problems before you
start recording.

    python check_env.py
"""

import importlib
import sys

# (import name, pip name, what it is used for)
PACKAGES = [
    ("numpy",       "numpy",         "core arrays"),
    ("cv2",         "opencv-python", "camera capture and video writing"),
    ("torch",       "torch",         "runs the YOLO pose model (CPU or CUDA)"),
    ("ultralytics", "ultralytics",   "YOLO pose model + BoT-SORT tracking"),
]


def main():
    print("Python:", sys.version.split()[0])
    print()

    width = max(len(name) for name, _, _ in PACKAGES)
    missing = []

    for import_name, pip_name, use in PACKAGES:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "?")
            print(f"  ok    {import_name:<{width}}  {version:<10}  {use}")
        except Exception:
            print(f"  MISS  {import_name:<{width}}  {'(missing)':<10}  {use}")
            missing.append(pip_name)

    # CUDA is optional; the local video-file collector can run on CPU.
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  CUDA: available — {torch.cuda.get_device_name(0)}")
        else:
            print("\n  CUDA: not available (CPU only — fine for testing; "
                  "use the Jetson GPU for real-time collection)")
    except Exception:
        pass

    print()
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install dependencies with:  pip install -r requirements.txt")
        return 1
    print("Environment looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
