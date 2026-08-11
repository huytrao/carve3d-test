#!/usr/bin/env python3
"""Smoke-test the public MVDream prompt-to-four-view stage on one GPU.

This deliberately stops after prompt -> four generated views. It is the fast
first check before spending time on LGM reconstruction, rendering, and MRC.
The output ordering is view_000, view_090, view_180, view_270, ready for the
four-real-view mode of full_pipeline/run.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


MVDREAM_CHECKPOINT = "ashawkey/mvdream-sd2.1-diffusers"
ANGLES = (0, 90, 180, 270)


def silence_huggingface_progress() -> None:
    """Hide model-loading/generation tqdm output while retaining real errors."""

    # Set before importing MVDream, which imports Diffusers/Transformers.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except (ImportError, AttributeError):
        pass
    try:
        from diffusers.utils import logging as diffusers_logging

        diffusers_logging.set_verbosity_error()
        diffusers_logging.disable_progress_bar()
    except (ImportError, AttributeError):
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and save four MVDream views from a text prompt.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/prompt-smoke-test"))
    parser.add_argument("--lgm-root", type=Path, default=Path("/kaggle/working/LGM"), help="LGM source contains the official MVDream diffusers pipeline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=30, help="Use 20 for a faster smoke test; 30 matches the full prompt runner.")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def save_image(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    if not args.lgm_root.is_dir():
        raise FileNotFoundError(f"LGM source missing at {args.lgm_root}; run scripts/setup_lgm_kaggle.sh first.")
    sys.path.insert(0, str(args.lgm_root.resolve()))

    silence_huggingface_progress()
    import torch
    from mvdream.pipeline_mvdream import MVDreamPipeline

    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {args.device} is unavailable.")
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = MVDreamPipeline.from_pretrained(
        MVDREAM_CHECKPOINT, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    images = pipe(
        args.prompt,
        negative_prompt="",
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        elevation=0,
        device=device,
    )

    # LGM's official MVDream convention maps these source indices to [0, 90,
    # 180, 270] camera rays respectively.
    ordered_views = [np.asarray(images[index], dtype=np.float32) for index in (1, 2, 3, 0)]
    for angle, image in zip(ANGLES, ordered_views):
        save_image(image, args.output_dir / f"view_{angle:03d}.png")
    grid = np.concatenate(
        (np.concatenate((ordered_views[0], ordered_views[1]), axis=1), np.concatenate((ordered_views[2], ordered_views[3]), axis=1)),
        axis=0,
    )
    save_image(grid, args.output_dir / "prompt_grid.png")

    del pipe
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    print(f"Prompt views saved to: {args.output_dir}")
    print(f"Grid: {args.output_dir / 'prompt_grid.png'}")


if __name__ == "__main__":
    main()
