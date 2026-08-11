#!/usr/bin/env python3
"""Run the missing, executable half of the public Carve3D pipeline.

Carve3D's released training code intentionally leaves the multiview diffusion
checkpoint and sparse-view reconstruction model open.  This runner supplies a
reproducible research replacement:

* `--input-dir`: four real views -> LGM Gaussian reconstruction -> render/MRC
* `--prompt`: MVDream checkpoint -> four generated views -> LGM -> render/MRC

It does not claim to recreate the unpublished Carve3D/Instant3D checkpoint.
The generated prompt route is an open-checkpoint baseline; the four-image route
is the direct MRC/reconstruction experiment requested by the paper's metric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

ANGLES = (0, 90, 180, 270)
LGM_CHECKPOINT_URL = "https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors"
MVDREAM_CHECKPOINT = "ashawkey/mvdream-sd2.1-diffusers"


def _silence_huggingface_progress() -> None:
    """Hide Diffusers/Transformers tqdm noise without suppressing exceptions."""

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
    parser = argparse.ArgumentParser(
        description="Four-view LGM reconstruction and Carve3D-style MRC evaluation."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path, help="Folder containing view_000, view_090, view_180, view_270 images.")
    source.add_argument("--views", nargs=4, type=Path, metavar=("VIEW_000", "VIEW_090", "VIEW_180", "VIEW_270"), help="Four images in [0, 90, 180, 270] degree order.")
    source.add_argument("--prompt", help="Generate four views with the open MVDream checkpoint before reconstructing.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full_pipeline"))
    parser.add_argument("--lgm-root", type=Path, default=Path(os.environ.get("LGM_ROOT", "LGM")), help="Clone of 3DTopia/LGM installed by scripts/setup_lgm_kaggle.sh.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="LGM model_fp16_fixrot.safetensors. Defaults to <lgm-root>/pretrained/.")
    parser.add_argument("--preset", choices=("big", "small"), default="big", help="LGM architecture. The published fp16_fixrot checkpoint requires 'big'.")
    parser.add_argument("--elevation", type=float, default=0.0, help="Common elevation in degrees for all four input cameras.")
    parser.add_argument(
        "--elevation-candidates",
        nargs="+",
        type=float,
        default=None,
        help=(
            "For four real photographs, reconstruct at every listed common elevation and keep the lowest "
            "same-pose MRC. Example: --elevation-candidates -10 -5 0 5 10."
        ),
    )
    parser.add_argument("--render-size", type=int, default=512, help="Output render edge length; 256 lowers GPU use.")
    parser.add_argument("--mrc-metric", choices=("lpips", "l1"), default="lpips", help="LPIPS is Carve3D MRC; L1 is only a dependency-light smoke-test metric.")
    parser.add_argument(
        "--refine-steps",
        type=int,
        default=0,
        help=(
            "Direct-four-view only: optimize the LGM-initialized Gaussians against the supplied photos. "
            "0 keeps feed-forward-only inference."
        ),
    )
    parser.add_argument(
        "--refine-learning-rate",
        type=float,
        default=0.02,
        help="Learning rate for direct Gaussian refinement (used only when --refine-steps is positive).",
    )
    parser.add_argument(
        "--refine-eval-every",
        type=int,
        default=10,
        help="Report/save the best direct-refinement MRC every N optimization steps.",
    )
    parser.add_argument("--lgm-device", type=int, default=0, help="CUDA device used for LGM reconstruction, rendering and MRC.")
    parser.add_argument("--prompt-device", type=int, default=0, help="CUDA device used for MVDream in --prompt mode.")
    parser.add_argument("--no-remove-background", dest="remove_background", action="store_false", help="Keep existing image backgrounds. Normally real images should be mattes on white.")
    parser.add_argument("--no-recenter", dest="recenter", action="store_false", help="Do not center/crop each foreground matte.")
    parser.add_argument("--no-orbit", dest="orbit", action="store_false", help="Skip 360-degree preview video.")
    parser.add_argument("--orbit-frames", type=int, default=90, help="Frames in the preview orbit video.")
    parser.add_argument("--seed", type=int, default=42, help="MVDream seed for --prompt mode.")
    parser.set_defaults(remove_background=True, recenter=True, orbit=True)
    return parser.parse_args()


def find_view_paths(input_dir: Path) -> list[Path]:
    """Find four canonical view images without relying on filesystem ordering."""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    extensions = (".png", ".jpg", ".jpeg", ".webp")
    paths = []
    for angle in ANGLES:
        matches = [
            candidate
            for candidate in input_dir.iterdir()
            if candidate.is_file()
            and candidate.suffix.lower() in extensions
            and candidate.stem.lower() in {f"view_{angle:03d}", f"view{angle:03d}", str(angle)}
        ]
        if len(matches) != 1:
            expected = input_dir / f"view_{angle:03d}.png"
            raise FileNotFoundError(
                f"expected exactly one image for {angle} degrees, e.g. {expected}; found {len(matches)}"
            )
        paths.append(matches[0])
    return paths


def _load_lgm(lgm_root: Path, checkpoint: Path, preset: str, render_size: int, device_index: int):
    """Load LGM through its official modules, keeping this repo dependency-light."""

    if not lgm_root.is_dir():
        raise FileNotFoundError(
            f"LGM source not found at {lgm_root}. Run scripts/setup_lgm_kaggle.sh first or pass --lgm-root."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"LGM checkpoint not found at {checkpoint}. The setup script downloads it from {LGM_CHECKPOINT_URL}."
        )
    if str(lgm_root.resolve()) not in sys.path:
        sys.path.insert(0, str(lgm_root.resolve()))

    import torch
    from safetensors.torch import load_file
    from core.models import LGM
    from core.options import config_defaults

    if not torch.cuda.is_available():
        raise RuntimeError("LGM Gaussian rasterization requires a CUDA GPU. Enable a Kaggle GPU accelerator.")
    if not 0 <= device_index < torch.cuda.device_count():
        raise RuntimeError(
            f"--lgm-device {device_index} is unavailable; detected {torch.cuda.device_count()} CUDA device(s)."
        )
    opt = config_defaults[preset]
    opt.output_size = render_size
    opt.lambda_lpips = 0.0  # MRC uses its own AlexNet LPIPS metric below.
    model = LGM(opt)
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=False)
    device = torch.device(f"cuda:{device_index}")
    model = model.half().to(device).eval()
    # LGM initializes this field on GPU 0. Move it explicitly so the original
    # Gaussian renderer works on GPU 1 in a dual-T4 Kaggle session too.
    model.gs.bg_color = model.gs.bg_color.to(device)
    return model, opt, device


def _prepare_rgb(array: np.ndarray, remove_background: bool, recenter_foreground: bool) -> np.ndarray:
    """Normalize an RGB array as a square, white-background LGM input."""

    if remove_background:
        try:
            import rembg
            from kiui.op import recenter
        except ImportError as exc:  # pragma: no cover - runtime dependency error
            raise RuntimeError("Background removal needs rembg and kiui; run the Kaggle setup script.") from exc
        rgba = rembg.remove(array)
        if recenter_foreground:
            rgba = recenter(rgba, rgba[..., 3] > 0, border_ratio=0.2)
        rgb = rgba[..., :3].astype(np.float32) / 255.0
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        composite = rgb * alpha + (1.0 - alpha)
        # `recenter` preserves the original aspect resolution.  Resize after
        # compositing so a four-camera capture with mixed source resolutions
        # is still a valid [4, H, W, 3] LGM tensor.
        square = Image.fromarray(np.clip(composite * 255, 0, 255).astype(np.uint8)).resize(
            (256, 256), Image.Resampling.LANCZOS
        )
        return np.asarray(square, dtype=np.float32) / 255.0
    square = Image.fromarray(array).resize((256, 256), Image.Resampling.LANCZOS)
    return np.asarray(square, dtype=np.float32) / 255.0


def _prepare_image(path: Path, remove_background: bool, recenter_foreground: bool) -> np.ndarray:
    """Load a real view as a square, white-background float RGB image."""

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return _prepare_rgb(np.asarray(image, dtype=np.uint8), remove_background, recenter_foreground)


def _save_image(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8)).save(destination)


def _make_grid(images: Sequence[np.ndarray]) -> np.ndarray:
    """Make the conventional 2x2 [0, 90; 180, 270] diagnostic grid."""

    top = np.concatenate((images[0], images[1]), axis=1)
    bottom = np.concatenate((images[2], images[3]), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _source_views_from_prompt(prompt: str, seed: int, device, elevation: float) -> list[np.ndarray]:
    _silence_huggingface_progress()
    import torch
    import kiui
    from mvdream.pipeline_mvdream import MVDreamPipeline

    kiui.seed_everything(seed)
    pipe = MVDreamPipeline.from_pretrained(
        MVDREAM_CHECKPOINT, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    generated = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=7.5,
        elevation=elevation,
        device=device,
    )
    # The reconstructor is already resident on the GPU.  Release MVDream before
    # LGM's forward pass so a 16 GB Kaggle GPU has the largest practical margin.
    del pipe
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    # This reorder is the official LGM/MVDream convention: these source images
    # align with LGM rays [0, 90, 180, 270].
    return [
        _prepare_rgb(
            np.clip(np.asarray(generated[index]) * 255, 0, 255).astype(np.uint8),
            remove_background=True,
            recenter_foreground=True,
        )
        for index in (1, 2, 3, 0)
    ]


def _render_views(model, opt, device, gaussians, angles: Sequence[int], elevation: float):
    import torch
    from kiui.cam import orbit_camera

    poses = torch.from_numpy(
        np.stack([orbit_camera(elevation, angle, radius=opt.cam_radius, opengl=True) for angle in angles])
    ).to(device)
    poses[:, :3, 1:3] *= -1  # LGM's official OpenGL -> rasterizer conversion.
    cam_view = torch.inverse(poses).transpose(1, 2).unsqueeze(0)
    projection = model.gs.proj_matrix.to(device)
    cam_view_proj = cam_view @ projection
    cam_pos = (-poses[:, :3, 3]).unsqueeze(0)
    return model.gs.render(gaussians, cam_view, cam_view_proj, cam_pos)["image"][0]


def _save_orbit(model, opt, device, gaussians, output_path: Path, elevation: float, frames: int) -> None:
    import imageio.v2 as imageio

    angles = np.linspace(0, 360, num=frames, endpoint=False, dtype=int)
    movie = []
    for angle in angles:
        image = _render_views(model, opt, device, gaussians, (int(angle),), elevation)[0]
        movie.append((image.permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    imageio.mimwrite(output_path, movie, fps=30, quality=8)


def _refine_direct_gaussians(
    model,
    opt,
    device,
    initial_gaussians,
    target_images,
    elevation: float,
    initial_mrc: dict,
    mrc_metric: str,
    lpips_model,
    steps: int,
    learning_rate: float,
    evaluate_every: int,
):
    """Photometrically refine LGM's feed-forward Gaussian initialization.

    This is deliberately a *direct four-photo* fitting step.  It optimizes the
    same Gaussian scene rendered at the four supplied camera poses, so unlike
    prompt RL it can improve the reconstruction the user is actually viewing.
    Parameter deltas are bounded around LGM's prediction to retain its 3D
    prior instead of overfitting each photograph independently.
    """

    import torch
    import torch.nn.functional as functional

    if steps < 1:
        raise ValueError("refinement needs at least one step")
    if learning_rate <= 0:
        raise ValueError("--refine-learning-rate must be positive")
    if evaluate_every < 1:
        raise ValueError("--refine-eval-every must be at least 1")

    from full_pipeline.mrc import compute_mrc

    target = functional.interpolate(
        target_images, size=(opt.output_size, opt.output_size), mode="bilinear", align_corners=False
    ).float()
    # White is the preprocessing convention.  Downweight its pixels so the
    # refinement concentrates on the object rather than winning by changing
    # an already-white background.
    foreground = (target < 0.97).any(dim=1, keepdim=True).float()
    weights = foreground * 0.95 + 0.05
    base = initial_gaussians.detach().float()
    base_positions = base[..., 0:3]
    base_opacity = base[..., 3:4].clamp(1e-4, 1 - 1e-4)
    base_scales = base[..., 4:7].clamp_min(1e-5)
    base_rotations = base[..., 7:11]
    base_colors = base[..., 11:].clamp(1e-4, 1 - 1e-4)

    # Keep geometry changes local to the LGM prediction. Colour and opacity
    # can move more freely; that is important for real photographs whose
    # lighting does not exactly match LGM's synthetic training distribution.
    position_delta = torch.nn.Parameter(torch.zeros_like(base_positions))
    opacity_logits = torch.nn.Parameter(torch.logit(base_opacity))
    log_scales = torch.nn.Parameter(torch.log(base_scales))
    color_logits = torch.nn.Parameter(torch.logit(base_colors))
    optimizer = torch.optim.Adam(
        [
            {"params": [position_delta], "lr": learning_rate * 0.10},
            {"params": [opacity_logits], "lr": learning_rate},
            {"params": [log_scales], "lr": learning_rate * 0.25},
            {"params": [color_logits], "lr": learning_rate},
        ]
    )

    def current_gaussians():
        positions = base_positions + torch.tanh(position_delta) * 0.08
        opacity = torch.sigmoid(opacity_logits)
        scales = torch.exp(log_scales).clamp(1e-5, 0.25)
        colors = torch.sigmoid(color_logits)
        return torch.cat((positions, opacity, scales, base_rotations, colors), dim=-1)

    best_gaussians = base.detach().clone()
    best_rendered = _render_views(model, opt, device, best_gaussians, ANGLES, elevation).detach()
    best_mrc = initial_mrc
    history = [{"step": 0, "mrc": initial_mrc["mrc"], "accepted": True}]
    print(f"[Refine 0/{steps}] baseline MRC={initial_mrc['mrc']:.6f}", flush=True)

    for step in range(1, steps + 1):
        gaussians = current_gaussians()
        rendered = _render_views(model, opt, device, gaussians, ANGLES, elevation)
        photometric = ((rendered.float() - target).abs() * weights).sum() / (weights.sum() * 3)
        position_prior = position_delta.square().mean()
        scale_prior = (torch.exp(log_scales) - base_scales).square().mean()
        loss = photometric + 0.002 * position_prior + 0.002 * scale_prior
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"] + optimizer.param_groups[1]["params"] + optimizer.param_groups[2]["params"] + optimizer.param_groups[3]["params"], 1.0)
        optimizer.step()

        if step % evaluate_every != 0 and step != steps:
            continue
        with torch.no_grad():
            candidate_gaussians = current_gaussians().detach()
            candidate_rendered = _render_views(model, opt, device, candidate_gaussians, ANGLES, elevation).detach()
            candidate_mrc, _ = compute_mrc(
                target, candidate_rendered, ANGLES, metric=mrc_metric, lpips_model=lpips_model
            )
        improved = candidate_mrc["mrc"] < best_mrc["mrc"]
        if improved:
            best_gaussians = candidate_gaussians.clone()
            best_rendered = candidate_rendered.clone()
            best_mrc = candidate_mrc
        history.append(
            {
                "step": step,
                "photometric_l1": float(photometric.detach().cpu()),
                "mrc": candidate_mrc["mrc"],
                "best_mrc": best_mrc["mrc"],
                "accepted": improved,
            }
        )
        marker = "new best" if improved else "kept best"
        print(
            f"[Refine {step}/{steps}] photo_L1={photometric.detach().item():.6f} "
            f"MRC={candidate_mrc['mrc']:.6f}; best={best_mrc['mrc']:.6f} ({marker})",
            flush=True,
        )

    return best_gaussians, best_rendered, best_mrc, {
        "enabled": True,
        "steps": steps,
        "learning_rate": learning_rate,
        "evaluate_every": evaluate_every,
        "objective": "foreground-weighted photometric L1 with bounded LGM Gaussian deltas",
        "history": history,
    }


def main() -> None:
    args = parse_args()
    import torch

    if args.refine_steps < 0:
        raise ValueError("--refine-steps cannot be negative")
    if args.refine_steps and args.prompt:
        raise ValueError("--refine-steps is for four real input views, not --prompt mode.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or args.lgm_root / "pretrained" / "model_fp16_fixrot.safetensors"
    model, opt, device = _load_lgm(
        args.lgm_root, checkpoint, args.preset, args.render_size, args.lgm_device
    )

    if args.prompt:
        if args.elevation_candidates and len(args.elevation_candidates) > 1:
            raise ValueError("--elevation-candidates is for real four-view input, not --prompt mode.")
        if not 0 <= args.prompt_device < torch.cuda.device_count():
            raise RuntimeError(
                f"--prompt-device {args.prompt_device} is unavailable; detected {torch.cuda.device_count()} CUDA device(s)."
            )
        prompt_device = torch.device(f"cuda:{args.prompt_device}")
        source_views = _source_views_from_prompt(args.prompt, args.seed, prompt_device, args.elevation)
        source_description = {
            "mode": "open_mvdream_prompt",
            "prompt": args.prompt,
            "seed": args.seed,
            "mvdream_device": args.prompt_device,
        }
    else:
        paths = list(args.views) if args.views else find_view_paths(args.input_dir)
        source_views = [_prepare_image(path, args.remove_background, args.recenter) for path in paths]
        source_description = {"mode": "four_real_views", "paths": [str(path) for path in paths]}

    import torch.nn.functional as F

    if len(source_views) != 4:
        raise RuntimeError(f"expected exactly four source views, got {len(source_views)}")
    prepared_dir = args.output_dir / "prepared_views"
    for angle, source in zip(ANGLES, source_views):
        _save_image(source, prepared_dir / f"view_{angle:03d}.png")
    _save_image(_make_grid(source_views), args.output_dir / "input_grid.png")

    input_tensor = torch.from_numpy(np.stack(source_views)).permute(0, 3, 1, 2).float().to(device)
    input_tensor = F.interpolate(input_tensor, size=(opt.input_size, opt.input_size), mode="bilinear", align_corners=False)
    normalized = input_tensor.clone()
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    normalized.sub_(mean).div_(std)
    # A turntable/capture rig often sits a few degrees above or below the
    # horizontal LGM convention.  The former one-shot path hard-coded 0° and
    # therefore made a correct object look inconsistent simply because its
    # cameras were mis-specified.  Search only for real supplied views; prompt
    # images already use the exact elevation passed to MVDream.
    elevations = args.elevation_candidates or [args.elevation]
    if args.prompt:
        elevations = [args.elevation]

    # Import here so --help and input-layout validation remain lightweight.
    from full_pipeline.mrc import _lpips_model, compute_mrc

    lpips_model = _lpips_model(device) if args.mrc_metric == "lpips" else None
    candidates = []
    for candidate_elevation in elevations:
        rays = model.prepare_default_rays(device, elevation=candidate_elevation)
        lgm_input = torch.cat((normalized, rays), dim=1).unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            candidate_gaussians = model.forward_gaussians(lgm_input)
            candidate_rendered = _render_views(model, opt, device, candidate_gaussians, ANGLES, candidate_elevation)
        resized_inputs = F.interpolate(
            input_tensor, size=candidate_rendered.shape[-2:], mode="bilinear", align_corners=False
        )
        candidate_mrc, _ = compute_mrc(
            resized_inputs,
            candidate_rendered,
            ANGLES,
            metric=args.mrc_metric,
            lpips_model=lpips_model,
        )
        candidates.append(
            {
                "elevation": candidate_elevation,
                "gaussians": candidate_gaussians,
                "rendered": candidate_rendered,
                "mrc": candidate_mrc,
            }
        )
        print(
            f"Elevation {candidate_elevation:+.1f}\N{DEGREE SIGN}: MRC={candidate_mrc['mrc']:.6f}",
            flush=True,
        )

    best_candidate = min(candidates, key=lambda candidate: candidate["mrc"]["mrc"])
    selected_elevation = best_candidate["elevation"]
    gaussians = best_candidate["gaussians"]
    rendered_tensor = best_candidate["rendered"]
    mrc = best_candidate["mrc"]
    refinement = {"enabled": False, "steps": 0}
    if args.refine_steps:
        print(
            f"Refining the selected direct reconstruction for {args.refine_steps} steps at "
            f"{selected_elevation:+.1f}\N{DEGREE SIGN}...",
            flush=True,
        )
        gaussians, rendered_tensor, mrc, refinement = _refine_direct_gaussians(
            model,
            opt,
            device,
            gaussians,
            input_tensor,
            selected_elevation,
            mrc,
            args.mrc_metric,
            lpips_model,
            args.refine_steps,
            args.refine_learning_rate,
            args.refine_eval_every,
        )
    model.gs.save_ply(gaussians, args.output_dir / "reconstruction.ply")

    rendered_np = rendered_tensor.permute(0, 2, 3, 1).float().cpu().numpy()
    for angle, render in zip(ANGLES, rendered_np):
        _save_image(render, args.output_dir / f"render_view_{angle:03d}.png")
    _save_image(_make_grid(list(rendered_np)), args.output_dir / "render_grid.png")

    metadata = {
        "pipeline": "LGM-four-view-adapter-for-Carve3D-MRC",
        "checkpoint": str(checkpoint),
        "checkpoint_url": LGM_CHECKPOINT_URL,
        "lgm_preset": args.preset,
        "angles_degrees": list(ANGLES),
        "elevation_degrees": selected_elevation,
        "elevation_candidates_degrees": elevations,
        "elevation_search": [
            {"elevation_degrees": candidate["elevation"], "mrc": candidate["mrc"]}
            for candidate in candidates
        ],
        "render_size": args.render_size,
        "lgm_device": args.lgm_device,
        "preprocessing": {
            "remove_background": args.remove_background,
            "recenter_foreground": args.recenter,
        },
        "direct_refinement": refinement,
        "source": source_description,
        "mrc": mrc,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.orbit:
        _save_orbit(model, opt, device, gaussians, args.output_dir / "orbit.mp4", selected_elevation, args.orbit_frames)
    print(json.dumps(metadata, indent=2))
    print(f"\nCompleted. Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
