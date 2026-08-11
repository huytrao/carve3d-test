"""Standalone Kaggle T4x2 prompt pipeline script.

Upload this file to Kaggle and run it, or paste its complete contents into one
Code cell. It clones the pipeline branch, downloads public checkpoints on the
first run, then generates four views, reconstructs 3D, renders, and computes
MRC. Enable Internet and choose the T4 x2 accelerator first.
"""

from pathlib import Path
import os
import subprocess
import sys


# ===== Change only these values when needed =====
PROMPT = "a wooden chair, studio product photograph, centered object, white background"
OUTPUT_DIR = "/kaggle/working/chair-output"
SEED = "42"

# ===== Pipeline bootstrap =====
REPO_URL = "https://github.com/huytrao/carve3d-test.git"
BRANCH = "implement_full_pipeline"
REPO_DIR = Path("/kaggle/working/carve3d-test")
LGM_ROOT = "/kaggle/working/LGM"


def main() -> None:
    if not (REPO_DIR / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)],
            check=True,
        )
    else:
        subprocess.run(
            ["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", BRANCH],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(REPO_DIR), "checkout", "-B", BRANCH, f"origin/{BRANCH}"],
            check=True,
        )

    # GPU 0: MVDream. GPU 1: LGM reconstruction, rendering, and MRC.
    sys.path.insert(0, str(REPO_DIR))
    os.environ["CARVE3D_PROMPT"] = PROMPT
    os.environ["CARVE3D_OUTPUT_DIR"] = OUTPUT_DIR
    os.environ["CARVE3D_SEED"] = SEED
    os.environ["CARVE3D_LGM_ROOT"] = LGM_ROOT

    from kaggle_full_pipeline_prompt import main as pipeline_main

    pipeline_main([
        "--lgm-root",
        LGM_ROOT,
        "--prompt",
        PROMPT,
        "--seed",
        SEED,
        "--output-dir",
        OUTPUT_DIR,
    ])


if __name__ == "__main__":
    main()
