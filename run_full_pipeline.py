#!/usr/bin/env python3
"""One-command Kaggle T4x2 prompt pipeline runner.

Run from the cloned repository root:

    python run_full_pipeline.py

Optional Kaggle cell overrides:

    %env CARVE3D_PROMPT=a ceramic teapot, studio product photograph, centered object, white background
    %env CARVE3D_OUTPUT_DIR=/kaggle/working/teapot-output
    !python run_full_pipeline.py

The first run downloads LGM automatically. Internet and a Kaggle T4 x2
accelerator must be enabled.
"""

from __future__ import annotations

import os
import shlex
import sys


DEFAULT_PROMPT = "a wooden chair, studio product photograph, centered object, white background"
DEFAULT_OUTPUT_DIR = "/kaggle/working/carve3d-prompt-output"
DEFAULT_LGM_ROOT = "/kaggle/working/LGM"


def build_pipeline_arguments() -> list[str]:
    """Build the reproducible prompt command, allowing Kaggle env overrides."""

    prompt = os.environ.get("CARVE3D_PROMPT", DEFAULT_PROMPT)
    output_dir = os.environ.get("CARVE3D_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    lgm_root = os.environ.get("CARVE3D_LGM_ROOT", DEFAULT_LGM_ROOT)
    seed = os.environ.get("CARVE3D_SEED", "42")
    extra_arguments = shlex.split(os.environ.get("CARVE3D_EXTRA_ARGS", ""))
    return [
        "--lgm-root",
        lgm_root,
        "--prompt",
        prompt,
        "--seed",
        seed,
        "--output-dir",
        output_dir,
        *extra_arguments,
    ]


def main() -> None:
    from kaggle_full_pipeline_prompt import main as kaggle_prompt_main

    arguments = build_pipeline_arguments()
    print("Running prompt:", arguments[arguments.index("--prompt") + 1])
    print("Writing results to:", arguments[arguments.index("--output-dir") + 1])
    sys.argv = ["kaggle_full_pipeline_prompt.py", *arguments]
    kaggle_prompt_main()


if __name__ == "__main__":
    main()
