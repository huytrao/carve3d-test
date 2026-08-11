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
the LPIPS MRC reward.  The default is a real one-epoch, two-sample RL smoke
run.  Paper-scale 55-epoch training used 48 A100 80 GB GPUs and is not a
reasonable expectation for two T4s.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    parser.add_argument("--epochs", type=int, default=16, help="On-policy RL updates. The quality preset uses 16 on T4 x2.")
    parser.add_argument("--samples-per-epoch", type=int, default=4, help="Same-prompt trajectories per update; at least 2 are required for advantages.")
    parser.add_argument("--num-steps", type=int, default=30, help="DDIM denoising steps. 30 is the practical T4 x2 quality setting.")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--eta", type=float, default=1.0, help="Stochastic DDIM eta; non-zero is required for policy log probabilities.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--kl-coeff", type=float, default=0.2, help="Approximate KL-to-sampling-policy penalty.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
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
    return parser.parse_args()


def default_prompts() -> list[str]:
    return [
        "a wooden chair, isolated studio product photograph, white background",
        "a ceramic teapot, isolated studio product photograph, white background",
    ]


def default_validation_prompts() -> list[str]:
    return [
        "a red toy car, isolated studio product photograph, white background",
        "a brass table lamp, isolated studio product photograph, white background",
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
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            # Match Carve3D's mixed-precision recipe: the frozen base UNet is
            # fp16, while its trainable rank-4 LoRA weights stay fp32.
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        def forward(self, inputs):
            base_output = self.base(inputs)
            # Explicitly disable autocast for this residual: otherwise CUDA
            # would silently cast both fp32 LoRA matrices back to fp16.
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                residual = functional.linear(functional.linear(inputs.float(), self.lora_a), self.lora_b)
            return base_output + (residual * self.scale).to(dtype=base_output.dtype)

    return _LoRALinear(base_layer)


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
    pipe.unet.train()
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
            transitions.append(
                Transition(
                    timestep=timestep_int,
                    latent=old_latents.detach().cpu(),
                    next_latent=latents.detach().cpu(),
                    behavior_log_prob=behavior_log_prob.detach().cpu(),
                )
            )
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        decoded = ((decoded / 2) + 0.5).clamp(0, 1)
        ordered_images = decoded[list(MVDREAM_TO_LGM)].float().cpu().permute(0, 2, 3, 1).numpy()
    return Trajectory(
        prompt=prompt,
        seed=seed,
        prompt_embeddings=prompt_embeddings.detach().cpu(),
        camera=camera.detach().cpu(),
        transitions=transitions,
        ordered_images=ordered_images,
    )


class LgmMrcScorer:
    """Keep LGM and the LPIPS model resident on the reward GPU across RL samples."""

    def __init__(self, lgm_root: Path, device_index: int, render_size: int, metric: str, elevation: float) -> None:
        import torch
        from full_pipeline.run import _load_lgm

        self.metric = metric
        self.elevation = elevation
        checkpoint = lgm_root / "pretrained" / "model_fp16_fixrot.safetensors"
        self.model, self.opt, self.device = _load_lgm(lgm_root, checkpoint, "big", render_size, device_index)
        self.mrc_helpers = __import__("full_pipeline.mrc", fromlist=["compute_mrc"])
        self.lpips = self.mrc_helpers._lpips_model(self.device) if metric == "lpips" else None
        self.torch = torch

    def _prepared_tensor(self, ordered_images: np.ndarray) -> Any:
        """Prepare generated RGB views without rembg; MRC itself finds white-background crops."""

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

    def score(self, ordered_images: np.ndarray) -> tuple[float, np.ndarray, list[dict[str, Any]]]:
        """Return MRC distance (lower is better), LGM renders, and per-view metadata."""

        import torch.nn.functional as functional
        from full_pipeline.run import _render_views

        inputs = self._prepared_tensor(ordered_images)
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
        return mrc, rendered_images, details


def _save_grid(images: np.ndarray, destination: Path) -> None:
    """Save a [0, 1] [4,H,W,3] image batch as the conventional 2x2 grid."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    top = np.concatenate((images[0], images[1]), axis=1)
    bottom = np.concatenate((images[2], images[3]), axis=1)
    grid = np.concatenate((top, bottom), axis=0)
    Image.fromarray(np.clip(grid * 255, 0, 255).astype(np.uint8)).save(destination)


def score_and_save(trajectory: Trajectory, scorer: LgmMrcScorer, destination: Path) -> None:
    mrc, rendered, details = scorer.score(trajectory.ordered_images)
    trajectory.mrc = mrc
    trajectory.reward = -mrc  # Paper's reward is negative MRC: lower reconstruction discrepancy is better.
    _save_grid(trajectory.ordered_images, destination / "source_grid.png")
    _save_grid(rendered, destination / "render_grid.png")
    (destination / "reward.json").write_text(
        json.dumps(
            {
                "prompt": trajectory.prompt,
                "seed": trajectory.seed,
                "mrc": mrc,
                "reward_negative_mrc": trajectory.reward,
                "mrc_metric": scorer.metric,
                "views": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def replay_policy_gradient(
    pipe: Any,
    trajectories: Sequence[Trajectory],
    advantages: Sequence[float],
    device: Any,
    eta: float,
    kl_coeff: float,
    max_grad_norm: float,
    optimizer: Any,
) -> dict[str, float]:
    """One pure on-policy LoRA update over sampled trajectories.

    ``behavior_log_prob`` is detached from the just-sampled policy.  The
    approximate KL term is the local quadratic KL estimator against that
    behaviour policy, keeping this tiny T4 run close to the base trajectory
    without retaining a second 2+ GB MVDream UNet on GPU 0.
    """

    import torch
    from diffusers_patch.ddim_with_logprob import ddim_step_with_logprob

    optimizer.zero_grad(set_to_none=True)
    total_policy_loss = 0.0
    total_kl = 0.0
    usable_steps = 0
    pipe.unet.train()
    denominator = max(1, len(trajectories))
    for trajectory, advantage in zip(trajectories, advantages):
        prompt_embeddings = trajectory.prompt_embeddings.to(device=device, dtype=torch.float16)
        camera = trajectory.camera.to(device=device, dtype=torch.float16)
        for transition in trajectory.transitions:
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
                if not current_log_prob.requires_grad:
                    # The final deterministic DDIM action has no probability density.
                    continue
                log_prob = current_log_prob.float().mean()
                delta = current_log_prob.float() - behavior_log_prob
                policy_loss = -float(advantage) * log_prob / denominator
                kl_loss = kl_coeff * 0.5 * delta.square().mean() / denominator
                (policy_loss + kl_loss).backward()
            total_policy_loss += float(policy_loss.detach().cpu())
            total_kl += float(kl_loss.detach().cpu())
            usable_steps += 1
    if usable_steps == 0:
        raise RuntimeError("No stochastic DDIM actions were available for the RL update. Increase --num-steps.")
    parameters = [parameter for parameter in pipe.unet.parameters() if parameter.requires_grad]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm).detach().cpu())
    optimizer.step()
    return {
        "policy_loss": total_policy_loss / usable_steps,
        "approx_kl_loss": total_kl / usable_steps,
        "grad_norm": grad_norm,
        "stochastic_steps": usable_steps,
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


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.samples_per_epoch < 2:
        raise ValueError("--samples-per-epoch must be at least 2 so RL can compute a reward advantage")
    if args.num_steps < 2:
        raise ValueError("--num-steps must be at least 2")
    if args.validation_every < 1 or args.early_stop_patience < 1:
        raise ValueError("--validation-every and --early-stop-patience must be at least 1")
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
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
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
    print(f"baseline validation MRC={best_validation_mrc:.6f}")

    for epoch in range(args.epochs):
        trajectories: list[Trajectory] = []
        # Score multiple random noises for the *same* prompt. This gives the
        # reward-normalized advantage a meaningful within-prompt comparison,
        # then cycles over the curated prompt list across epochs.
        epoch_prompt = prompts[epoch % len(prompts)]
        for sample_index in range(args.samples_per_epoch):
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
            score_and_save(trajectory, scorer, args.output_dir / "epochs" / f"epoch_{epoch:03d}" / f"sample_{sample_index:03d}")
            trajectories.append(trajectory)
            print(f"epoch={epoch} sample={sample_index} MRC={trajectory.mrc:.6f} reward={trajectory.reward:.6f}")

        rewards = np.asarray([trajectory.reward for trajectory in trajectories], dtype=np.float32)
        advantages = ((rewards - rewards.mean()) / (rewards.std() + 1e-6)).tolist()
        update = replay_policy_gradient(
            pipe,
            trajectories,
            advantages,
            diffusion_device,
            args.eta,
            args.kl_coeff,
            args.max_grad_norm,
            optimizer,
        )
        epoch_record = {
            "epoch": epoch,
            "mean_mrc": float(np.mean([-reward for reward in rewards])),
            "mean_reward_negative_mrc": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "advantages": advantages,
            **update,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2))
        save_lora(pipe.unet, args.output_dir / "checkpoints" / f"lora_epoch_{epoch:03d}.pt")

        should_validate = (epoch + 1) % args.validation_every == 0 or epoch + 1 == args.epochs
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
            if validation["mean_mrc"] < best_validation_mrc:
                best_validation_mrc = validation["mean_mrc"]
                best_epoch = epoch
                best_state = save_lora(pipe.unet, args.output_dir / "checkpoints" / "best_lora.pt")
                validations_without_improvement = 0
                print(f"new best validation MRC={best_validation_mrc:.6f} at epoch={epoch}")
            else:
                validations_without_improvement += 1
                print(
                    f"validation MRC={validation['mean_mrc']:.6f}; best={best_validation_mrc:.6f}; "
                    f"no-improvement validations={validations_without_improvement}"
                )
                if validations_without_improvement >= args.early_stop_patience:
                    print("Early stopping: validation MRC stopped improving.")
                    break

    restore_lora(pipe.unet, best_state)
    pipe.unet.eval()
    evaluation = sample_trajectory(
        pipe,
        args.final_prompt,
        args.seed + 100_000,
        diffusion_device,
        args.num_steps,
        args.guidance_scale,
        args.eta,
        args.elevation,
    )
    score_and_save(evaluation, scorer, args.output_dir / "final_evaluation")
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
            "kl_coeff": args.kl_coeff,
            "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "ddim_steps": args.num_steps,
            "eta": args.eta,
            "validation_every": args.validation_every,
            "early_stop_patience": args.early_stop_patience,
        },
        "devices": {"mvdream_rl": args.diffusion_device, "lgm_mrc": args.lgm_device},
        "mrc_metric": args.mrc_metric,
        "lora_projection_count": lora_projection_count,
        "baseline_validation": baseline_validation,
        "best_validation_mrc": best_validation_mrc,
        "best_epoch": best_epoch,
        "history": history,
        "final_evaluation": {
            "prompt": evaluation.prompt,
            "seed": evaluation.seed,
            "mrc": evaluation.mrc,
            "reward_negative_mrc": evaluation.reward,
        },
    }
    (args.output_dir / "rlft_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nCompleted open RLFT. Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
