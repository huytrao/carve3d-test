#!/usr/bin/env python3
"""Kaggle launcher: four captured views -> LGM -> renders -> Carve3D MRC.

This launcher deliberately has no prompt or MVDream stage.  It reads the four
real photographs already mounted in Kaggle, in this fixed camera order:

    view_000.png, view_090.png, view_180.png, view_270.png

The default input location is the dataset location supplied for this project:
``/kaggle/input/datasets/traoanhuy/carve3d/views``.  Kaggle normally removes
the owner component from mounted datasets, so ``/kaggle/input/carve3d/views``
is tried as a fallback too.

Example:

    python kaggle_import_four_views.py \\
      --input-dir /kaggle/input/datasets/traoanhuy/carve3d/views \\
      --output-dir /kaggle/working/carve3d-four-view-output

Enable Kaggle Internet and a GPU accelerator.  On T4 x2 the launcher uses GPU
1 for LGM, leaving GPU 0 free; LGM's single reconstruction cannot be split
across the two cards, so this is intentional rather than a misleading
"dual-GPU" claim.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


DEFAULT_LGM_ROOT = Path("/kaggle/working/LGM")
DEFAULT_INPUT_DIRS = (
    Path("/kaggle/input/datasets/traoanhuy/carve3d/views"),
    Path("/kaggle/input/carve3d/views"),
)
SETUP_VERSION = "3"
LGM_CHECKPOINT_URL = "https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors"
ANGLES = (0, 90, 180, 270)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


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


def _without_option(arguments: list[str], option: str) -> list[str]:
    """Remove one ``--option value`` or ``--option=value`` pair safely."""

    result: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == option:
            index += 2
        elif argument.startswith(f"{option}="):
            index += 1
        else:
            result.append(argument)
            index += 1
    return result


def resolve_input_dir(value: str | None) -> Path:
    """Resolve a supplied folder, or the two known Kaggle mount variants."""

    candidates = (Path(value),) if value is not None else DEFAULT_INPUT_DIRS
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find the four-view input folder. Checked:\n"
        f"  - {checked}\n"
        "Set --input-dir to the folder that contains view_000.png, "
        "view_090.png, view_180.png, and view_270.png."
    )


def validate_four_views(input_dir: Path) -> list[Path]:
    """Validate canonical view names before downloading or compiling anything."""

    paths: list[Path] = []
    missing: list[str] = []
    for angle in ANGLES:
        # Cameras are commonly named either view_090.png or view_90.png.
        # Accept both, while preserving the required [0, 90, 180, 270] order.
        aliases = {
            f"view_{angle:03d}",
            f"view{angle:03d}",
            f"view_{angle}",
            f"view{angle}",
            str(angle),
        }
        matches = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.stem.lower() in aliases
        )
        if len(matches) != 1:
            missing.append(f"view_{angle:03d}.png (found {len(matches)})")
        else:
            paths.append(matches[0])
    if missing:
        raise FileNotFoundError(
            "Need exactly four camera views in this order: 000, 090, 180, 270.\n"
            f"Folder: {input_dir}\n"
            f"Missing/ambiguous: {', '.join(missing)}\n"
            "Your supplied 180 and 270 files are valid, but view_000 and view_090 "
            "must also be present in the same folder."
        )
    return paths


def select_lgm_device(requested: str | None) -> int:
    """Pick GPU 1 on T4 x2; otherwise use the only available CUDA device."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. In Kaggle, enable a GPU accelerator first.")
    count = torch.cuda.device_count()
    if requested is None:
        device_index = 1 if count >= 2 else 0
    else:
        try:
            device_index = int(requested)
        except ValueError as exc:
            raise ValueError("--lgm-device must be a CUDA index such as 0 or 1") from exc
    if not 0 <= device_index < count:
        raise RuntimeError(f"--lgm-device {device_index} is unavailable; detected {count} CUDA device(s).")
    for index in range(count):
        print(f"Detected GPU {index}: {torch.cuda.get_device_name(index)}")
    if count >= 2 and device_index == 1:
        print("Using GPU 1 for LGM. GPU 0 is intentionally free; LGM cannot split one reconstruction across T4 x2.")
    else:
        print(f"Using GPU {device_index} for LGM.")
    return device_index


def bootstrap(repo_root: Path, lgm_root: Path, force: bool) -> None:
    """Install LGM once and download its public checkpoint when needed."""

    checkpoint = lgm_root / "pretrained" / "model_fp16_fixrot.safetensors"
    stamp = lgm_root / ".carve3d_setup_version"
    current = stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == SETUP_VERSION
    if force or not checkpoint.is_file() or not current:
        print(f"Preparing LGM dependencies; public checkpoint: {LGM_CHECKPOINT_URL}")
        subprocess.run(["bash", str(repo_root / "scripts" / "setup_lgm_kaggle.sh"), str(lgm_root)], check=True)
    else:
        print(f"Reusing LGM and checkpoint: {checkpoint}")


def _reject_prompt_flags(arguments: list[str]) -> None:
    forbidden = ("--prompt", "--prompt-device", "--views")
    present = [option for option in forbidden if _has_option(arguments, option)]
    if present:
        raise SystemExit(
            "This is the direct-four-image launcher; it does not accept "
            f"{', '.join(present)}. Put all four real images in --input-dir."
        )


def main(arguments: list[str] | None = None) -> None:
    """Validate local Kaggle images, then delegate reconstruction to the common runner."""

    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if "--help" in arguments or "-h" in arguments:
        print(__doc__)
        print("Extra LGM options accepted: --render-size, --mrc-metric, --no-orbit, --no-remove-background.")
        return
    _reject_prompt_flags(arguments)
    force_bootstrap = "--bootstrap" in arguments
    skip_bootstrap = "--no-bootstrap" in arguments
    if force_bootstrap and skip_bootstrap:
        raise SystemExit("Choose at most one of --bootstrap and --no-bootstrap.")
    arguments = [argument for argument in arguments if argument not in {"--bootstrap", "--no-bootstrap"}]

    input_dir = resolve_input_dir(_option_value(arguments, "--input-dir"))
    view_paths = validate_four_views(input_dir)
    print("Validated source views:")
    for angle, path in zip(ANGLES, view_paths):
        print(f"  {angle:03d}° -> {path}")

    lgm_device = select_lgm_device(_option_value(arguments, "--lgm-device"))
    repo_root = Path(__file__).resolve().parent
    lgm_root = Path(_option_value(arguments, "--lgm-root", str(DEFAULT_LGM_ROOT)))
    if not skip_bootstrap:
        bootstrap(repo_root, lgm_root, force_bootstrap)

    # Pass the validated paths directly. This lets this launcher accept common
    # non-padded camera names such as view_90.png even though the shared runner
    # itself intentionally defaults to the strict canonical file layout.
    arguments = _without_option(arguments, "--input-dir")
    arguments.extend(("--views", *(str(path) for path in view_paths)))
    if not _has_option(arguments, "--lgm-root"):
        arguments.extend(("--lgm-root", str(lgm_root)))
    if not _has_option(arguments, "--lgm-device"):
        arguments.extend(("--lgm-device", str(lgm_device)))

    from full_pipeline.run import main as pipeline_main

    sys.argv = ["full_pipeline/run.py", *arguments]
    pipeline_main()


if __name__ == "__main__":
    main()
