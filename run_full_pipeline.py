#!/usr/bin/env python3
"""One-command Kaggle T4x2 prompt pipeline runner.

Run from the cloned repository root, or upload/paste this file by itself into
Kaggle. In the latter case it clones the required pipeline code automatically:

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
import subprocess
import sys
from pathlib import Path


DEFAULT_PROMPT = "a wooden chair, studio product photograph, centered object, white background"
DEFAULT_OUTPUT_DIR = "/kaggle/working/carve3d-prompt-output"
DEFAULT_LGM_ROOT = "/kaggle/working/LGM"
DEFAULT_REPOSITORY_DIR = Path("/kaggle/working/carve3d-test")
REPOSITORY_URL = "https://github.com/huytrao/carve3d-test.git"
REPOSITORY_BRANCH = "implement_full_pipeline"


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


def _has_launcher(directory: Path) -> bool:
    return (directory / "kaggle_full_pipeline_prompt.py").is_file()


def resolve_pipeline_repository() -> Path:
    """Return a repo containing the launcher, cloning it when run standalone."""

    # `__file__` exists when the file is executed with Python, but not when a
    # user pastes its contents into a Kaggle/Jupyter code cell. In that case
    # fall back to the notebook working directory and bootstrap from there.
    script_file = globals().get("__file__")
    script_directory = Path(script_file).resolve().parent if script_file else Path.cwd()
    if _has_launcher(script_directory):
        return script_directory

    repository = Path(os.environ.get("CARVE3D_REPO_DIR", str(DEFAULT_REPOSITORY_DIR)))
    if _has_launcher(repository):
        return repository
    if repository.exists() and any(repository.iterdir()):
        raise RuntimeError(
            f"{repository} exists but does not contain kaggle_full_pipeline_prompt.py. "
            "Set CARVE3D_REPO_DIR to an empty path or a clone of the pipeline branch."
        )
    repository.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning pipeline code from {REPOSITORY_URL} ({REPOSITORY_BRANCH})...")
    subprocess.run(
        ["git", "clone", "--branch", REPOSITORY_BRANCH, REPOSITORY_URL, str(repository)],
        check=True,
    )
    if not _has_launcher(repository):  # Defensive check if the remote changes.
        raise RuntimeError("Downloaded repository does not contain kaggle_full_pipeline_prompt.py.")
    return repository


def main() -> None:
    repository = resolve_pipeline_repository()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from kaggle_full_pipeline_prompt import main as kaggle_prompt_main

    arguments = build_pipeline_arguments()
    print("Running prompt:", arguments[arguments.index("--prompt") + 1])
    print("Writing results to:", arguments[arguments.index("--output-dir") + 1])
    sys.argv = ["kaggle_full_pipeline_prompt.py", *arguments]
    kaggle_prompt_main()


if __name__ == "__main__":
    main()
