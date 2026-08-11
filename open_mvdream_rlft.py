#!/usr/bin/env python3
"""Runnable open replacement for Carve3D's unavailable RLFT components.

This implements the *algorithmic* RL loop from the paper with public models:

    prompt -> public MVDream 4 views -> public LGM -> same-pose renders
           -> bbox LPIPS MRC reward -> one on-policy LoRA policy-gradient step

It is not an exact reproduction of Carve3D.  The paper's Instant3D diffusion
checkpoint and sparse-view LRM are not public, and the released ``rewards.py``
leaves those functions as TODOs.  This runner uses MVDream and LGM only as an
open, executable substitute and writes its result metadata accordingly.

For Kaggle T4 x2, GPU 0 holds MVDream and its LoRA update; GPU 1 holds LGM and
the LPIPS MRC reward. Paper-scale 55-epoch training used 48 A100 80 GB GPUs
and is not a reasonable expectation for two T4s.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


ANGLES = (0, 90, 180, 270)
# MVDream's public pipeline output must be permuted to match LGM's canonical
# camera-ray order. This is the same convention used by full_pipeline/run.py.
MVDREAM_TO_LGM = (1, 2, 3, 0)
MVDREAM_CHECKPOINT = "ashawkey/mvdream-sd2.1-diffusers"
LGM_CHECKPOINT_URL = "https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors"


def silence_huggingface_progress() -> None:
    """Avoid tqdm/model-load spam while retaining failures and final metrics."""

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    try:
        from diffusers.utils import logging as diffusers_logging

        diffusers_logging.disable_progress_bar()
        diffusers_logging.set_verbosity_error()
    except (AttributeError, ImportError):
        pass
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
    except (AttributeError, ImportError):
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open MVDream + LGM MRC reward, on-policy LoRA RLFT. Not an exact unavailable Carve3D reproduction."
    )
    parser.add_argument("--lgm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/open-rlft-output"))
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="One training prompt. Repeat --prompt for more. Defaults to two conservative object prompts.",
    )
    parser.add_argument("--epochs", type=int, default=16, help="On-policy RL updates.")
    parser.add_argument("--samples-per-epoch", type=int, default=4, help="Same-prompt trajectories per update; at least 2 are required for advantages.")
    parser.add_argument(
        "--prompts-per-update",
        type=int,
        default=1,
        help="Distinct prompts in each update; samples-per-epoch must divide evenly and leave at least 2 samples per prompt.",
    )
    parser.add_argument("--num-steps", type=int, default=30, help="DDIM denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--eta", type=float, default=1.0, help="Stochastic DDIM eta; non-zero is required for policy log probabilities.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help=(
            "LoRA AdamW learning rate. 1e-5 is the conservative T4 small-batch default; "
            "the paper's 3e-4 was used with batch 768 on 48 A100s."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--reward-coeff", type=float, default=1.0)
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=0.2,
        help="Coefficient for per-prompt normalized trajectory KL to the frozen base MVDream policy.",
    )
    parser.add_argument(
        "--timestep-loss-reduction",
        choices=("mean", "sum"),
        default="mean",
        help="Mean is stable across DDIM step counts on small T4 batches; sum matches the paper equation literally.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--advantage-clip",
        type=float,
        default=5.0,
        help="Clamp the combined reward/KL advantage, matching the released Carve3D trainer default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--diffusion-device", type=int, default=0)
    parser.add_argument("--lgm-device", type=int, default=1)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--mrc-metric", choices=("lpips", "l1"), default="lpips")
    parser.add_argument("--final-prompt", default="a wooden chair, isolated studio product photograph, white background")
    parser.add_argument(
        "--validation-prompt",
        action="append",
        dest="validation_prompts",
        help="Held-out prompt for best-checkpoint selection. Repeat for multiple prompts.",
    )
    parser.add_argument("--validation-every", type=int, default=2, help="Validate every N RL updates.")
    parser.add_argument("--early-stop-patience", type=int, default=4, help="Stop after this many validations without a lower MRC.")
    parser.add_argument(
        "--min-validation-improvement",
        type=float,
        default=1e-4,
        help="Minimum absolute MRC reduction required to accept a transactional update.",
    )
    parser.add_argument(
        "--transactional-validation",
        action="store_true",
        help=(
            "Validate every update and restore the pre-update LoRA/optimizer state whenever validation MRC does not improve. "
            "Recommended for T4's small, noisy RL batches."
        ),
    )
    parser.add_argument(
        "--final-candidates",
        type=int,
        default=1,
        help="Generate this many fixed-seed final samples and retain the one with lowest MRC (at least 1).",
    )
    return parser.parse_args()


def default_prompts() -> list[str]:
    return [
        "a wooden chair, isolated studio product photograph, white background",
        "a ceramic teapot, isolated studio product photograph, white background",
    ]


def default_validation_prompts() -> list[str]:
    return [
        "a green desk fan, isolated studio product photograph, white background",
        "a canvas hiking backpack, isolated studio product photograph, white background",
    ]


class LoRALinear:  # Wrapped lazily so importing --help does not require PyTorch.
    """Factory-like wrapper implemented in ``make_lora_linear`` below."""


def make_lora_linear(base_layer: Any, rank: int, alpha: float) -> Any:
    """Add a small trainable LoRA residual without changing the base weights."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class _LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear) -> None:
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            self.rlft_lora_enabled = True
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            # Match Carve3D's mixed-precision recipe: the frozen base UNet is
            # fp16, while its trainable rank-4 LoRA weights stay fp32.
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        def forward(self, inputs):
            base_output = self.base(inputs)
            if not self.rlft_lora_enabled:
                return base_output
            # Explicitly disable autocast for this residual: otherwise CUDA
            # would silently cast both fp32 LoRA matrices back to fp16.
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                residual = functional.linear(functional.linear(inputs.float(), self.lora_a), self.lora_b)
            return base_output + (residual * self.scale).to(dtype=base_output.dtype)

    return _LoRALinear(base_layer)


@contextmanager
def disable_lora(unet: Any):
    """Temporarily expose the frozen base policy without a second UNet copy."""

    modules = [module for module in unet.modules() if hasattr(module, "rlft_lora_enabled")]
    previous = [module.rlft_lora_enabled for module in modules]
    for module in modules:
        module.rlft_lora_enabled = False
    try:
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.rlft_lora_enabled = enabled


def inject_attention_lora(unet: Any, rank: int, alpha: float) -> int:
    """Inject LoRA only in query/key/value/output attention projections."""

    import torch.nn as nn

    for parameter in unet.parameters():
        parameter.requires_grad_(False)

    replacements = 0

    def visit(module: Any, parent_name: str = "") -> None:
        nonlocal replacements
        for child_name, child in list(module.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            is_attention_projection = child_name in {"to_q", "to_k", "to_v"} or (
                child_name == "0" and parent_name.endswith("to_out")
            )
            if isinstance(child, nn.Linear) and is_attention_projection:
                setattr(module, child_name, make_lora_linear(child, rank, alpha))
                replacements += 1
            else:
                visit(child, full_name)

    visit(unet)
    if replacements == 0:
        raise RuntimeError("Could not find MVDream attention Linear layers for LoRA injection.")
    return replacements


@dataclass
class Transition:
    timestep: int
    latent: Any
    next_latent: Any
    behavior_log_prob: Any
    base_log_prob: Any
    is_stochastic: bool


@dataclass
class Trajectory:
    prompt: str
    seed: int
    prompt_embeddings: Any
    camera: Any
    transitions: list[Transition]
    ordered_images: np.ndarray
    mrc: float | None = None
    reward: float | None = None
    kl_to_base: float | None = None


def _check_devices(diffusion_device: int, lgm_device: int) -> tuple[Any, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Enable the Kaggle GPU accelerator.")
    count = torch.cuda.device_count()
    if not 0 <= diffusion_device < count or not 0 <= lgm_device < count:
        raise RuntimeError(f"Requested CUDA devices {diffusion_device} and {lgm_device}; detected {count} GPU(s).")
    if diffusion_device == lgm_device:
        raise RuntimeError("Use separate --diffusion-device and --lgm-device. T4 x2 should use 0 and 1.")
    for index in range(count):
        print(f"Detected GPU {index}: {torch.cuda.get_device_name(index)}")
    print(f"MVDream/RL uses GPU {diffusion_device}; LGM/MRC reward uses GPU {lgm_device}.")
    return torch.device(f"cuda:{diffusion_device}"), torch.device(f"cuda:{lgm_device}")


def load_mvdream_with_lora(lgm_root: Path, device: Any, rank: int, alpha: float) -> tuple[Any, int]:
    """Load the public MVDream checkpoint and freeze everything except LoRA."""

    if not (lgm_root / "mvdream" / "pipeline_mvdream.py").is_file():
        raise FileNotFoundError(f"LGM/MVDream source is missing at {lgm_root}; run scripts/setup_lgm_kaggle.sh first.")
    if str(lgm_root.resolve()) not in sys.path:
        sys.path.insert(0, str(lgm_root.resolve()))
    silence_huggingface_progress()
    import torch
    from mvdream.pipeline_mvdream import MVDreamPipeline

    pipe = MVDreamPipeline.from_pretrained(
        MVDREAM_CHECKPOINT, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    injected = inject_attention_lora(pipe.unet, rank, alpha)
    # ``pipe.to(device)`` above ran before LoRA injection. New Parameters are
    # created on CPU by PyTorch, so move the modified UNet once more; otherwise
    # the first attention projection mixes CUDA activations with CPU LoRA A/B.
    pipe.unet.to(device)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    # Eval mode does not disable autograd. Keeping dropout/training-only layers
    # disabled makes the replayed policy match the policy used for sampling.
    pipe.unet.eval()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    print(f"Loaded public MVDream with {injected} LoRA attention projections.")
    return pipe, injected


def _guided_noise(pipe: Any, latents: Any, timestep: int, prompt_embeddings: Any, camera: Any) -> Any:
    """Run MVDream's multi-view UNet with classifier-free guidance."""

    import torch

    number_of_frames = latents.shape[0]
    unconditional, conditional = prompt_embeddings.chunk(2)
    latent_model_input = pipe.scheduler.scale_model_input(torch.cat([latents, latents]), timestep)
    contexts = torch.cat([unconditional] * number_of_frames + [conditional] * number_of_frames)
    unet_inputs = {
        "x": latent_model_input,
        "timesteps": torch.full(
            (number_of_frames * 2,), timestep, dtype=latent_model_input.dtype, device=latents.device
        ),
        "context": contexts,
        "num_frames": number_of_frames,
        "camera": torch.cat([camera, camera]),
    }
    noise_prediction = pipe.unet.forward(**unet_inputs)
    noise_unconditional, noise_conditional = noise_prediction.chunk(2)
    return noise_unconditional + pipe._rlft_guidance_scale * (noise_conditional - noise_unconditional)


def sample_trajectory(
    pipe: Any,
    prompt: str,
    seed: int,
    device: Any,
    num_steps: int,
    guidance_scale: float,
    eta: float,
    elevation: float,
) -> Trajectory:
    """Sample a stochastic four-view trajectory and retain behaviour log-probs."""

    import torch
    from diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
    from mvdream.mv_unet import get_camera

    if eta <= 0:
        raise ValueError("--eta must be > 0 to compute stochastic DDIM policy log probabilities.")
    pipe._rlft_guidance_scale = guidance_scale
    generator = torch.Generator(device=device).manual_seed(seed)
    pipe.scheduler.set_timesteps(num_steps, device=device)
    was_training = pipe.unet.training
    pipe.unet.eval()
    with torch.no_grad():
        prompt_embeddings = pipe._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )
        _, conditional = prompt_embeddings.chunk(2)
        latents = pipe.prepare_latents(
            4,
            4,
            256,
            256,
            conditional.dtype,
            device,
            generator,
            None,
        )
        camera = get_camera(4, elevation=elevation).to(dtype=latents.dtype, device=device)
        transitions: list[Transition] = []
        for timestep in pipe.scheduler.timesteps:
            timestep_int = int(timestep.item())
            old_latents = latents
            noise_prediction = _guided_noise(pipe, old_latents, timestep_int, prompt_embeddings, camera)
            latents, behavior_log_prob = ddim_step_with_logprob(
                pipe.scheduler,
                noise_prediction,
                torch.full((4,), timestep_int, device=device, dtype=torch.long),
                old_latents,
                eta=eta,
                generator=generator,
            )
            # Carve3D regularizes against the frozen base model, not against
            # the just-sampled behaviour policy. Reuse the same UNet weights
            # with LoRA disabled so T4 does not need a second model copy.
            with disable_lora(pipe.unet):
                base_noise_prediction = _guided_noise(
                    pipe, old_latents, timestep_int, prompt_embeddings, camera
                )
                _, base_log_prob = ddim_step_with_logprob(
                    pipe.scheduler,
                    base_noise_prediction,
                    torch.full((4,), timestep_int, device=device, dtype=torch.long),
                    old_latents,
                    eta=eta,
                    prev_sample=latents,
                )
            is_stochastic = bool(behavior_log_prob.detach().abs().max().item() > 0)
            transitions.append(
                Transition(
                    timestep=timestep_int,
                    latent=old_latents.detach().cpu(),
                    next_latent=latents.detach().cpu(),
                    behavior_log_prob=behavior_log_prob.detach().cpu(),
                    base_log_prob=base_log_prob.detach().cpu(),
                    is_stochastic=is_stochastic,
                )
            )
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        decoded = ((decoded / 2) + 0.5).clamp(0, 1)
        ordered_images = decoded[list(MVDREAM_TO_LGM)].float().cpu().permute(0, 2, 3, 1).numpy()
    if was_training:
        pipe.unet.train()
    stochastic_transitions = [transition for transition in transitions if transition.is_stochastic]
    if not stochastic_transitions:
        raise RuntimeError("No stochastic DDIM transitions were sampled; eta must be positive.")
    kl_to_base = float(
        np.mean(
            [
                (transition.behavior_log_prob.float() - transition.base_log_prob.float()).mean().item()
                for transition in stochastic_transitions
            ]
        )
    )
    return Trajectory(
        prompt=prompt,
        seed=seed,
        prompt_embeddings=prompt_embeddings.detach().cpu(),
        camera=camera.detach().cpu(),
        transitions=transitions,
        ordered_images=ordered_images,
        kl_to_base=kl_to_base,
    )


class LgmMrcScorer:
    """Keep LGM and the LPIPS model resident on the reward GPU across RL samples."""

    def __init__(self, lgm_root: Path, device_index: int, render_size: int, metric: str, elevation: float) -> None:
        import torch
        import rembg
        from full_pipeline.run import _load_lgm

        self.metric = metric
        self.elevation = elevation
        checkpoint = lgm_root / "pretrained" / "model_fp16_fixrot.safetensors"
        self.model, self.opt, self.device = _load_lgm(lgm_root, checkpoint, "big", render_size, device_index)
        self.mrc_helpers = __import__("full_pipeline.mrc", fromlist=["compute_mrc"])
        self.lpips = self.mrc_helpers._lpips_model(self.device) if metric == "lpips" else None
        self.torch = torch

        # LGM's official text-to-3D path removes the background and recenters
        # every MVDream view before reconstruction. Feeding raw MVDream RGB to
        # LGM creates a large train/inference mismatch, so MRC mostly measures
        # background/crop failure instead of multi-view consistency. Force the
        # CPU provider to avoid Kaggle's noisy onnxruntime CUDA-provider probe.
        self.rembg = rembg
        self.background_session = rembg.new_session(providers=["CPUExecutionProvider"])

    def _prepare_lgm_views(self, ordered_images: np.ndarray) -> np.ndarray:
        """Apply the official LGM MVDream matte/recenter/white-bg preprocessing."""

        from kiui.op import recenter

        prepared: list[np.ndarray] = []
        for image in ordered_images:
            uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            rgba = self.rembg.remove(uint8, session=self.background_session)
            if rgba.ndim != 3 or rgba.shape[-1] != 4:
                raise RuntimeError(f"rembg returned an invalid image shape: {rgba.shape}")
            mask = rgba[..., 3] > 0
            if mask.any():
                rgba = recenter(rgba, mask, border_ratio=0.2)
            rgb = rgba[..., :3].astype(np.float32) / 255.0
            alpha = rgba[..., 3:4].astype(np.float32) / 255.0
            composite = rgb * alpha + (1.0 - alpha)
            square = Image.fromarray(np.clip(composite * 255, 0, 255).astype(np.uint8)).resize(
                (256, 256), Image.Resampling.LANCZOS
            )
            prepared.append(np.asarray(square, dtype=np.float32) / 255.0)
        return np.stack(prepared)

    def _prepared_tensor(self, ordered_images: np.ndarray) -> Any:
        """Resize already-matted RGB views for LGM reconstruction and MRC."""

        import torch.nn.functional as functional

        images = self.torch.from_numpy(ordered_images).permute(0, 3, 1, 2).float().to(self.device)
        images = functional.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        return images

    def _mrc(self, inputs: Any, rendered: Any) -> tuple[float, list[dict[str, Any]]]:
        import torch.nn.functional as functional

        values: list[float] = []
        details: list[dict[str, Any]] = []
        for angle, source, target in zip(ANGLES, inputs, rendered):
            crop = self.mrc_helpers._square_crop(source)
            source_crop = self.mrc_helpers._crop_and_resize(source, crop, 256)
            target_crop = self.mrc_helpers._crop_and_resize(target, crop, 256)
            l1 = functional.l1_loss(source_crop, target_crop).item()
            if self.metric == "lpips":
                value = self.lpips(source_crop * 2 - 1, target_crop * 2 - 1).mean().item()
                details.append({"view": angle, "lpips": value, "l1": l1, "crop": crop})
            else:
                value = l1
                details.append({"view": angle, "lpips": None, "l1": l1, "crop": crop})
            values.append(value)
        return float(sum(values) / len(values)), details

    def score(
        self, ordered_images: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Return MRC, prepared sources, LGM renders, and per-view metadata."""

        import torch.nn.functional as functional
        from full_pipeline.run import _render_views

        prepared_images = self._prepare_lgm_views(ordered_images)
        inputs = self._prepared_tensor(prepared_images)
        normalized = inputs.clone()
        mean = self.torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = self.torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        normalized.sub_(mean).div_(std)
        rays = self.model.prepare_default_rays(self.device, elevation=self.elevation)
        lgm_input = self.torch.cat((normalized, rays), dim=1).unsqueeze(0)
        with self.torch.no_grad(), self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
            gaussians = self.model.forward_gaussians(lgm_input)
            rendered = _render_views(self.model, self.opt, self.device, gaussians, ANGLES, self.elevation)
        resized_inputs = functional.interpolate(inputs, size=rendered.shape[-2:], mode="bilinear", align_corners=False)
        mrc, details = self._mrc(resized_inputs, rendered)
        rendered_images = rendered.permute(0, 2, 3, 1).float().cpu().numpy()
        return mrc, prepared_images, rendered_images, details


def _save_grid(images: np.ndarray, destination: Path) -> None:
    """Save a [0, 1] [4,H,W,3] image batch as the conventional 2x2 grid."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    top = np.concatenate((images[0], images[1]), axis=1)
    bottom = np.concatenate((images[2], images[3]), axis=1)
    grid = np.concatenate((top, bottom), axis=0)
    Image.fromarray(np.clip(grid * 255, 0, 255).astype(np.uint8)).save(destination)


def score_and_save(trajectory: Trajectory, scorer: LgmMrcScorer, destination: Path) -> None:
    mrc, prepared, rendered, details = scorer.score(trajectory.ordered_images)
    trajectory.mrc = mrc
    trajectory.reward = -mrc  # Paper's reward is negative MRC: lower reconstruction discrepancy is better.
    _save_grid(trajectory.ordered_images, destination / "source_grid.png")
    _save_grid(prepared, destination / "prepared_source_grid.png")
    _save_grid(rendered, destination / "render_grid.png")
    (destination / "reward.json").write_text(
        json.dumps(
            {
                "prompt": trajectory.prompt,
                "seed": trajectory.seed,
                "mrc": mrc,
                "reward_negative_mrc": trajectory.reward,
                "trajectory_kl_to_base": trajectory.kl_to_base,
                "mrc_metric": scorer.metric,
                "views": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def grouped_normalized_advantages(
    values: Sequence[float], group_ids: Sequence[str], epsilon: float = 1e-6
) -> list[float]:
    """Normalize values independently for each prompt in the current on-policy batch."""

    if len(values) != len(group_ids):
        raise ValueError("values and group_ids must have the same length")
    result = np.zeros(len(values), dtype=np.float32)
    for group_id in dict.fromkeys(group_ids):
        indexes = [index for index, candidate in enumerate(group_ids) if candidate == group_id]
        group = np.asarray([values[index] for index in indexes], dtype=np.float32)
        standard_deviation = float(group.std())
        normalized = np.zeros_like(group) if standard_deviation < epsilon else (group - group.mean()) / standard_deviation
        for index, value in zip(indexes, normalized):
            result[index] = value
    return result.tolist()


def replay_policy_gradient(
    pipe: Any,
    trajectories: Sequence[Trajectory],
    advantages: Sequence[float],
    device: Any,
    eta: float,
    max_grad_norm: float,
    optimizer: Any,
    timestep_loss_reduction: str,
) -> dict[str, Any]:
    """One pure on-policy LoRA update over sampled trajectories.

    ``behavior_log_prob`` is detached from the just-sampled policy.  The
    KL-to-base is already folded into ``advantages`` at trajectory level,
    matching Carve3D's pure on-policy score-function objective.
    """

    import torch
    from diffusers_patch.ddim_with_logprob import ddim_step_with_logprob

    optimizer.zero_grad(set_to_none=True)
    total_policy_loss = 0.0
    usable_steps = 0
    replay_errors: list[float] = []
    # Autograd remains enabled in eval mode. This prevents dropout or other
    # training-only behavior from changing the policy between sample/replay.
    pipe.unet.eval()
    stochastic_steps = sum(
        1 for trajectory in trajectories for transition in trajectory.transitions if transition.is_stochastic
    )
    if stochastic_steps == 0:
        raise RuntimeError("No stochastic DDIM actions were available for the RL update. Increase --num-steps.")
    if timestep_loss_reduction == "mean":
        denominator = stochastic_steps
    elif timestep_loss_reduction == "sum":
        denominator = max(1, len(trajectories))
    else:
        raise ValueError("timestep_loss_reduction must be 'mean' or 'sum'")
    for trajectory, advantage in zip(trajectories, advantages):
        prompt_embeddings = trajectory.prompt_embeddings.to(device=device, dtype=torch.float16)
        camera = trajectory.camera.to(device=device, dtype=torch.float16)
        for transition in trajectory.transitions:
            if not transition.is_stochastic:
                continue
            latents = transition.latent.to(device=device, dtype=torch.float16)
            next_latent = transition.next_latent.to(device=device, dtype=torch.float16)
            behavior_log_prob = transition.behavior_log_prob.to(device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_prediction = _guided_noise(pipe, latents, transition.timestep, prompt_embeddings, camera)
                _, current_log_prob = ddim_step_with_logprob(
                    pipe.scheduler,
                    noise_prediction,
                    torch.full((latents.shape[0],), transition.timestep, device=device, dtype=torch.long),
                    latents,
                    eta=eta,
                    prev_sample=next_latent,
                )
                log_prob = current_log_prob.float().mean()
                replay_errors.append(float((current_log_prob.detach().float() - behavior_log_prob).abs().mean().cpu()))
                policy_loss = -float(advantage) * log_prob / denominator
                policy_loss.backward()
            total_policy_loss += float(policy_loss.detach().cpu())
            usable_steps += 1
    parameters = [parameter for parameter in pipe.unet.parameters() if parameter.requires_grad]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm).detach().cpu())
    optimizer.step()
    return {
        "policy_loss": total_policy_loss,
        "grad_norm": grad_norm,
        "stochastic_steps": usable_steps,
        "timestep_loss_reduction": timestep_loss_reduction,
        "replay_logprob_mean_abs_error": float(np.mean(replay_errors)),
        "replay_logprob_max_abs_error": float(np.max(replay_errors)),
    }


def lora_state(unet: Any) -> dict[str, Any]:
    """Return an independent CPU copy of the small trainable LoRA state."""

    state: dict[str, Any] = {
        name: parameter.detach().cpu()
        for name, parameter in unet.named_parameters()
        if parameter.requires_grad and ("lora_a" in name or "lora_b" in name)
    }
    if not state:
        raise RuntimeError("No LoRA parameters were found to save.")
    return state


def save_lora(unet: Any, destination: Path) -> dict[str, Any]:
    """Save only trained LoRA tensors, never a multi-gigabyte base checkpoint."""

    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    state = lora_state(unet)
    torch.save(state, destination)
    return state


def restore_lora(unet: Any, state: dict[str, Any]) -> None:
    """Restore the validation-best LoRA weights before final generation."""

    parameters = dict(unet.named_parameters())
    for name, value in state.items():
        if name not in parameters:
            raise RuntimeError(f"Best LoRA checkpoint has no matching parameter: {name}")
        parameters[name].data.copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))


def write_training_progress(destination: Path, **progress: Any) -> None:
    """Persist a compact status record after each costly RL milestone."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def evaluate_prompts(
    pipe: Any,
    prompts: Sequence[str],
    scorer: LgmMrcScorer,
    destination: Path,
    seed: int,
    device: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate fixed-seed held-out prompts and return their mean MRC."""

    was_training = pipe.unet.training
    pipe.unet.eval()
    records: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        trajectory = sample_trajectory(
            pipe,
            prompt,
            seed + index,
            device,
            args.num_steps,
            args.guidance_scale,
            args.eta,
            args.elevation,
        )
        score_and_save(trajectory, scorer, destination / f"prompt_{index:02d}")
        records.append({"prompt": prompt, "seed": trajectory.seed, "mrc": trajectory.mrc})
    mean_mrc = float(np.mean([record["mrc"] for record in records]))
    result = {"mean_mrc": mean_mrc, "records": records}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if was_training:
        pipe.unet.train()
    return result


def evaluate_final_candidates(
    pipe: Any,
    scorer: LgmMrcScorer,
    device: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Score final fixed-seed samples and retain the best reconstruction-consistent one.

    This is deliberately reported as inference-time best-of-N selection.  It
    does not pretend that sampling several seeds is an RL improvement or a
    result reported by the original Carve3D paper.
    """

    if args.final_candidates < 1:
        raise ValueError("--final-candidates must be at least 1")
    candidates_root = args.output_dir / "final_candidates"
    records: list[dict[str, Any]] = []
    best_path: Path | None = None
    best_trajectory: Trajectory | None = None
    for index in range(args.final_candidates):
        seed = args.seed + 100_000 + index
        trajectory = sample_trajectory(
            pipe,
            args.final_prompt,
            seed,
            device,
            args.num_steps,
            args.guidance_scale,
            args.eta,
            args.elevation,
        )
        candidate_path = candidates_root / f"candidate_{index:02d}"
        score_and_save(trajectory, scorer, candidate_path)
        record = {"index": index, "seed": seed, "mrc": trajectory.mrc, "reward_negative_mrc": trajectory.reward}
        records.append(record)
        print(
            f"[Final {index + 1}/{args.final_candidates}] MRC={trajectory.mrc:.6f}",
            flush=True,
        )
        if best_trajectory is None or trajectory.mrc < best_trajectory.mrc:
            best_trajectory = trajectory
            best_path = candidate_path

    assert best_trajectory is not None and best_path is not None
    final_destination = args.output_dir / "final_evaluation"
    shutil.rmtree(final_destination, ignore_errors=True)
    shutil.copytree(best_path, final_destination)
    result = {
        "selection": "lowest_mrc_over_fixed_seed_candidates",
        "candidate_count": args.final_candidates,
        "prompt": args.final_prompt,
        "selected": min(records, key=lambda record: record["mrc"]),
        "candidates": records,
    }
    (final_destination / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.samples_per_epoch < 2:
        raise ValueError("--samples-per-epoch must be at least 2 so RL can compute a reward advantage")
    if args.prompts_per_update < 1:
        raise ValueError("--prompts-per-update must be at least 1")
    if args.samples_per_epoch % args.prompts_per_update != 0:
        raise ValueError("--samples-per-epoch must divide evenly by --prompts-per-update")
    samples_per_prompt = args.samples_per_epoch // args.prompts_per_update
    if samples_per_prompt < 2:
        raise ValueError("Each prompt needs at least two trajectories for per-prompt advantages")
    if args.num_steps < 2:
        raise ValueError("--num-steps must be at least 2")
    if args.validation_every < 1 or args.early_stop_patience < 1:
        raise ValueError("--validation-every and --early-stop-patience must be at least 1")
    if args.min_validation_improvement < 0:
        raise ValueError("--min-validation-improvement cannot be negative")
    if args.advantage_clip <= 0:
        raise ValueError("--advantage-clip must be positive")
    if args.final_candidates < 1:
        raise ValueError("--final-candidates must be at least 1")
    prompts = args.prompts or default_prompts()
    validation_prompts = args.validation_prompts or default_validation_prompts()
    if not prompts:
        raise ValueError("Supply at least one --prompt")

    diffusion_device, _ = _check_devices(args.diffusion_device, args.lgm_device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe, lora_projection_count = load_mvdream_with_lora(
        args.lgm_root, diffusion_device, args.lora_rank, args.lora_alpha
    )
    import torch

    trainable = [parameter for parameter in pipe.unet.parameters() if parameter.requires_grad]
    # PyTorch AdamW defaults to weight_decay=1e-2, one hundred times the value
    # reported by Carve3D. Specify the paper optimizer values explicitly.
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    scorer = LgmMrcScorer(args.lgm_root, args.lgm_device, args.render_size, args.mrc_metric, args.elevation)
    history: list[dict[str, Any]] = []
    validation_seed = args.seed + 1_000_000
    baseline_validation = evaluate_prompts(
        pipe,
        validation_prompts,
        scorer,
        args.output_dir / "validation" / "baseline",
        validation_seed,
        diffusion_device,
        args,
    )
    best_validation_mrc = baseline_validation["mean_mrc"]
    best_epoch = -1
    best_state = lora_state(pipe.unet)
    save_lora(pipe.unet, args.output_dir / "checkpoints" / "best_lora.pt")
    validations_without_improvement = 0
    progress_file = args.output_dir / "training_progress.json"
    write_training_progress(
        progress_file,
        status="running",
        stage="baseline_validation_complete",
        current_epoch=0,
        total_epochs=args.epochs,
        best_epoch=best_epoch,
        best_validation_mrc=best_validation_mrc,
        baseline_validation_mrc=baseline_validation["mean_mrc"],
    )
    print(f"[RL 0/{args.epochs}] baseline validation MRC={best_validation_mrc:.6f}", flush=True)

    for epoch in range(args.epochs):
        # In transactional mode the validation score is a hard trust-region
        # gate.  Snapshot AdamW as well as LoRA tensors: restoring only LoRA
        # would retain stale Adam moments and still push the next update in the
        # rejected direction.
        pre_update_state = lora_state(pipe.unet) if args.transactional_validation else None
        pre_update_optimizer_state = copy.deepcopy(optimizer.state_dict()) if args.transactional_validation else None
        trajectories: list[Trajectory] = []
        # Mix several prompts per update, but normalize reward/KL within each
        # prompt as Carve3D/DDPO requires. Cycling prompts across updates means
        # an object category is revisited instead of receiving one update only.
        epoch_prompts = [
            prompts[(epoch * args.prompts_per_update + index) % len(prompts)]
            for index in range(args.prompts_per_update)
        ]
        print(
            f"[RL {epoch + 1}/{args.epochs}] sampling {args.samples_per_epoch} trajectories "
            f"across {args.prompts_per_update} prompt(s)",
            flush=True,
        )
        for prompt_index, epoch_prompt in enumerate(epoch_prompts):
            print(
                f"[RL {epoch + 1}/{args.epochs}] prompt {prompt_index + 1}/{args.prompts_per_update}: "
                f"{epoch_prompt}",
                flush=True,
            )
            for prompt_sample_index in range(samples_per_prompt):
                sample_index = prompt_index * samples_per_prompt + prompt_sample_index
                seed = args.seed + epoch * args.samples_per_epoch + sample_index
                trajectory = sample_trajectory(
                    pipe,
                    epoch_prompt,
                    seed,
                    diffusion_device,
                    args.num_steps,
                    args.guidance_scale,
                    args.eta,
                    args.elevation,
                )
                score_and_save(
                    trajectory,
                    scorer,
                    args.output_dir / "epochs" / f"epoch_{epoch:03d}" / f"sample_{sample_index:03d}",
                )
                trajectories.append(trajectory)
                print(
                    f"[RL {epoch + 1}/{args.epochs}] sample {sample_index + 1}/{args.samples_per_epoch} "
                    f"MRC={trajectory.mrc:.6f} KL_base={trajectory.kl_to_base:.6f}",
                    flush=True,
                )
                write_training_progress(
                    progress_file,
                    status="running",
                    stage="sampling",
                    current_epoch=epoch + 1,
                    total_epochs=args.epochs,
                    current_sample=sample_index + 1,
                    samples_per_epoch=args.samples_per_epoch,
                    last_mrc=trajectory.mrc,
                    last_kl_to_base=trajectory.kl_to_base,
                    best_epoch=best_epoch,
                    best_validation_mrc=best_validation_mrc,
                )

        rewards = np.asarray([trajectory.reward for trajectory in trajectories], dtype=np.float32)
        kl_values = np.asarray([trajectory.kl_to_base for trajectory in trajectories], dtype=np.float32)
        group_ids = [trajectory.prompt for trajectory in trajectories]
        reward_advantages = grouped_normalized_advantages(rewards.tolist(), group_ids)
        kl_advantages = grouped_normalized_advantages(kl_values.tolist(), group_ids)
        unclipped_advantages = [
            args.reward_coeff * reward_advantage - args.kl_coeff * kl_advantage
            for reward_advantage, kl_advantage in zip(reward_advantages, kl_advantages)
        ]
        advantages = np.clip(
            np.asarray(unclipped_advantages, dtype=np.float32),
            -args.advantage_clip,
            args.advantage_clip,
        ).tolist()
        print(f"[RL {epoch + 1}/{args.epochs}] updating fp32 LoRA weights...", flush=True)
        update = replay_policy_gradient(
            pipe,
            trajectories,
            advantages,
            diffusion_device,
            args.eta,
            args.max_grad_norm,
            optimizer,
            args.timestep_loss_reduction,
        )
        epoch_record = {
            "epoch": epoch,
            "mean_mrc": float(np.mean([-reward for reward in rewards])),
            "mean_reward_negative_mrc": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "mean_kl_to_base": float(kl_values.mean()),
            "kl_to_base_std": float(kl_values.std()),
            "reward_advantages": reward_advantages,
            "kl_advantages": kl_advantages,
            "unclipped_combined_advantages": unclipped_advantages,
            "combined_advantages": advantages,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **update,
        }
        history.append(epoch_record)
        print(
            f"[RL {epoch + 1}/{args.epochs}] update complete: "
            f"mean_MRC={epoch_record['mean_mrc']:.6f}, mean_KL_base={epoch_record['mean_kl_to_base']:.6f}, "
            f"grad_norm={update['grad_norm']:.6f}, replay_error={update['replay_logprob_max_abs_error']:.2e}",
            flush=True,
        )
        should_validate = (
            args.transactional_validation
            or (epoch + 1) % args.validation_every == 0
            or epoch + 1 == args.epochs
        )
        if should_validate:
            validation = evaluate_prompts(
                pipe,
                validation_prompts,
                scorer,
                args.output_dir / "validation" / f"epoch_{epoch:03d}",
                validation_seed,
                diffusion_device,
                args,
            )
            epoch_record["validation_mrc"] = validation["mean_mrc"]
            validation_improvement = best_validation_mrc - validation["mean_mrc"]
            epoch_record["validation_improvement"] = validation_improvement
            improved = validation_improvement >= args.min_validation_improvement
            if improved:
                best_validation_mrc = validation["mean_mrc"]
                best_epoch = epoch
                best_state = save_lora(pipe.unet, args.output_dir / "checkpoints" / "best_lora.pt")
                validations_without_improvement = 0
                epoch_record["update_accepted"] = True
                print(
                    f"[RL {epoch + 1}/{args.epochs}] new best validation MRC={best_validation_mrc:.6f}; "
                    f"improvement={validation_improvement:.6f}",
                    flush=True,
                )
            else:
                validations_without_improvement += 1
                if args.transactional_validation:
                    assert pre_update_state is not None and pre_update_optimizer_state is not None
                    restore_lora(pipe.unet, pre_update_state)
                    optimizer.load_state_dict(pre_update_optimizer_state)
                    epoch_record["update_accepted"] = False
                    print(
                        f"[RL {epoch + 1}/{args.epochs}] rejected update and restored prior LoRA: "
                        f"validation MRC={validation['mean_mrc']:.6f}; best={best_validation_mrc:.6f}; "
                        f"required improvement={args.min_validation_improvement:.6f}",
                        flush=True,
                    )
                else:
                    epoch_record["update_accepted"] = True
                print(
                    f"[RL {epoch + 1}/{args.epochs}] validation MRC={validation['mean_mrc']:.6f}; "
                    f"best={best_validation_mrc:.6f}; no-improvement={validations_without_improvement}",
                    flush=True,
                )
                if validations_without_improvement >= args.early_stop_patience:
                    print("[RL] early stopping: validation MRC stopped improving.", flush=True)
                    write_training_progress(
                        progress_file,
                        status="early_stopped",
                        stage="training_complete",
                        current_epoch=epoch + 1,
                        total_epochs=args.epochs,
                        best_epoch=best_epoch,
                        best_validation_mrc=best_validation_mrc,
                    )
                    break
        # A transactional rejected candidate must never be presented as an
        # epoch checkpoint that a user could later load by mistake.
        if not args.transactional_validation or epoch_record.get("update_accepted", False):
            save_lora(pipe.unet, args.output_dir / "checkpoints" / f"lora_epoch_{epoch:03d}.pt")
        write_training_progress(
            progress_file,
            status="running",
            stage="epoch_complete",
            current_epoch=epoch + 1,
            total_epochs=args.epochs,
            mean_training_mrc=epoch_record["mean_mrc"],
            validation_mrc=epoch_record.get("validation_mrc"),
            best_epoch=best_epoch,
            best_validation_mrc=best_validation_mrc,
        )

    restore_lora(pipe.unet, best_state)
    pipe.unet.eval()
    final_selection = evaluate_final_candidates(pipe, scorer, diffusion_device, args)
    save_lora(pipe.unet, args.output_dir / "checkpoints" / "lora_final.pt")
    metadata = {
        "pipeline": "open-mvdream-lgm-mrc-on-policy-lora-rlft",
        "not_exact_carve3d": True,
        "reason": "The released Carve3D code and paper state that Instant3D and sparse-view LRM are unavailable.",
        "mvdream_checkpoint": MVDREAM_CHECKPOINT,
        "lgm_checkpoint_url": LGM_CHECKPOINT_URL,
        "mvdream_to_lgm_order": list(MVDREAM_TO_LGM),
        "rl": {
            "algorithm": "one replay/update per sampled on-policy batch; REINFORCE-style score function",
            "lora_rank": args.lora_rank,
            "reward_coeff": args.reward_coeff,
            "kl_coeff": args.kl_coeff,
            "kl_reference": "frozen base MVDream with LoRA disabled on every sampled transition",
            "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "prompts_per_update": args.prompts_per_update,
            "ddim_steps": args.num_steps,
            "eta": args.eta,
            "timestep_loss_reduction": args.timestep_loss_reduction,
            "advantage_clip": args.advantage_clip,
            "validation_every": args.validation_every,
            "early_stop_patience": args.early_stop_patience,
            "min_validation_improvement": args.min_validation_improvement,
            "learning_rate": args.learning_rate,
            "adamw": {"betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 1e-4},
            "transactional_validation": args.transactional_validation,
        },
        "devices": {"mvdream_rl": args.diffusion_device, "lgm_mrc": args.lgm_device},
        "mrc_metric": args.mrc_metric,
        "lora_projection_count": lora_projection_count,
        "baseline_validation": baseline_validation,
        "best_validation_mrc": best_validation_mrc,
        "best_epoch": best_epoch,
        "history": history,
        "final_evaluation": final_selection,
    }
    (args.output_dir / "rlft_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_training_progress(
        progress_file,
        status="completed",
        stage="final_evaluation_complete",
        current_epoch=len(history),
        total_epochs=args.epochs,
        best_epoch=best_epoch,
        best_validation_mrc=best_validation_mrc,
        final_mrc=final_selection["selected"]["mrc"],
    )
    print(f"\nCompleted open RLFT. Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
