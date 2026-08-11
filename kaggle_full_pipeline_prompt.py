#!/usr/bin/env python3
"""Kaggle T4x2 launcher for prompt -> 4 views -> LGM -> render -> LPIPS MRC.

GPU 0 runs MVDream's diffusion sampling. GPU 1 holds the LGM reconstructor,
Gaussian renderer, and MRC model. Splitting the models avoids relying on one
T4 to host both large checkpoints at once.

Example (from the repository root in a Kaggle notebook):

    python kaggle_full_pipeline_prompt.py \\
      --prompt "a wooden chair" \\
      --output-dir /kaggle/working/chair

The first run downloads and compiles LGM automatically. Enable Kaggle Internet
and choose the **T4 x2** accelerator before running it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


DEFAULT_LGM_ROOT = Path("/kaggle/working/LGM")
LGM_CHECKPOINT_URL = "https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors"
MVDREAM_MODEL_URL = "https://huggingface.co/ashawkey/mvdream-sd2.1-diffusers"
SETUP_VERSION = "3"


def _has_option(arguments: list[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in arguments)


def _option_value(arguments: list[str], option: str, default: str | None = None) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 == len(arguments):
                raise ValueError(f"{option} needs a value")
            return arguments[index + 1]
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
    return default


def _require_prompt(arguments: list[str]) -> None:
    if not _has_option(arguments, "--prompt"):
        raise SystemExit("This prompt-only launcher requires `--prompt \"...\"`.")
    if _has_option(arguments, "--input-dir") or _has_option(arguments, "--views"):
        raise SystemExit("Use full_pipeline/run.py for real input images; this launcher accepts only --prompt.")


def _verify_dual_t4() -> None:
    probe = """
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit('Kaggle T4 x2 is required: enable the dual-GPU accelerator.')
for index in range(2):
    name = torch.cuda.get_device_name(index)
    if 'T4' not in name:
        raise SystemExit(f'Expected T4 x2 but GPU {index} is {name!r}.')
print('Using GPU 0:', torch.cuda.get_device_name(0))
print('Using GPU 1:', torch.cuda.get_device_name(1))
"""
    subprocess.run([sys.executable, "-c", probe], check=True)


def _bootstrap(repo_root: Path, lgm_root: Path, force: bool) -> None:
    checkpoint = lgm_root / "pretrained" / "model_fp16_fixrot.safetensors"
    setup_stamp = lgm_root / ".carve3d_setup_version"
    setup_is_current = setup_stamp.is_file() and setup_stamp.read_text().strip() == SETUP_VERSION
    if force or not checkpoint.is_file() or not setup_is_current:
        print(f"Preparing LGM dependencies; checkpoint source: {LGM_CHECKPOINT_URL}")
        subprocess.run(["bash", str(repo_root / "scripts" / "setup_lgm_kaggle.sh"), str(lgm_root)], check=True)
    print(f"MVDream will be fetched automatically from: {MVDREAM_MODEL_URL}")


def main(arguments: list[str] | None = None) -> None:
    """Run the prompt pipeline from explicit args or normal CLI arguments."""

    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if "--help" in arguments or "-h" in arguments:
        print(__doc__)
        print("All remaining flags are documented by full_pipeline/run.py --help.")
        return
    _require_prompt(arguments)

    force_bootstrap = "--bootstrap" in arguments
    skip_bootstrap = "--no-bootstrap" in arguments
    if force_bootstrap and skip_bootstrap:
        raise SystemExit("Choose at most one of --bootstrap and --no-bootstrap.")
    arguments = [arg for arg in arguments if arg not in {"--bootstrap", "--no-bootstrap"}]
    repo_root = Path(__file__).resolve().parent
    lgm_root = Path(_option_value(arguments, "--lgm-root", str(DEFAULT_LGM_ROOT)))

    _verify_dual_t4()
    if not skip_bootstrap:
        _bootstrap(repo_root, lgm_root, force_bootstrap)

    if not _has_option(arguments, "--lgm-root"):
        arguments.extend(("--lgm-root", str(lgm_root)))
    if not _has_option(arguments, "--lgm-device"):
        arguments.extend(("--lgm-device", "1"))
    if not _has_option(arguments, "--prompt-device"):
        arguments.extend(("--prompt-device", "0"))

    # Delegate reconstruction, rendering, artifact saving, and MRC evaluation
    # to the shared runner while preserving usual command-line semantics.
    from full_pipeline.run import main as pipeline_main

    sys.argv = ["full_pipeline/run.py", *arguments]
    pipeline_main()


if __name__ == "__main__":
    main()
