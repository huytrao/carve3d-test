#!/usr/bin/env python3
"""Saved Kaggle command for the project's four real camera views.

Run this file from the checked-out repository:

    python run_import_four_views.py

It is intentionally not a prompt launcher.  Change only INPUT_DIR if Kaggle
mounts the dataset at a different location.
"""

from __future__ import annotations

import os
from pathlib import Path

from kaggle_import_four_views import main


INPUT_DIR = os.environ.get("CARVE3D_INPUT_DIR")
OUTPUT_DIR = os.environ.get("CARVE3D_OUTPUT_DIR", "/kaggle/working/carve3d-four-view-output")
LGM_ROOT = os.environ.get("LGM_ROOT", "/kaggle/working/LGM")


if __name__ == "__main__":
    arguments = [
        "--output-dir",
        str(Path(OUTPUT_DIR)),
        "--lgm-root",
        str(Path(LGM_ROOT)),
    ]
    # Omitting --input-dir lets the Kaggle launcher try both the historical
    # owner-qualified path and Kaggle's standard /kaggle/input/<slug> mount.
    if INPUT_DIR is not None:
        arguments.extend(("--input-dir", str(Path(INPUT_DIR))))
    main(arguments)
