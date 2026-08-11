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
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
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
        "--target-prompt",
        help=(
            "Primary object prompt for a target-specific RLFT experiment. It is used as the validation/final "
            "fallback, but it does not turn real input photographs into RL trajectories."
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help=(
            "One training prompt. Repeat for multiple prompt groups. At least --target-prompt or --prompt is "
            "required; there are intentionally no hidden chair/teapot defaults."
        ),
    )
    parser.add_argument("--epochs", type=int, default=16, help="On-policy RL updates.")
    parser.add_argument("--samples-per-epoch", type=int, default=4, help="Same-prompt trajectories per update; at least 2 are required for advantages.")
    parser.add_argument(
        "--prompts-per-update",
        type=int,
        default=1,
        help="Distinct prompts in each update; samples-per-epoch must divide evenly and leave at least 2 samples per prompt.",
    )
    parser.add_argument(
        "--curate-prompt-count",
        type=int,
        default=0,
        help=(
            "Before RL, score every explicit training prompt with the base policy and retain this many "
            "highest-MRC prompts. Zero disables the paper-style low-reward prompt curation."
        ),
    )
    parser.add_argument(
        "--curation-samples-per-prompt",
        type=int,
        default=4,
        help=(
            "Independent base-policy outputs averaged for each curation candidate. Carve3D Appendix C.1 uses 4; "
            "reducing this makes ranking faster but noisier."
        ),
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
    parser.add_argument(
        "--stat-buffer-epochs",
        type=int,
        default=3,
        help=(
            "Number of per-prompt sample windows retained when normalizing reward and base-KL advantages. "
            "The paper uses an approximately three-epoch window."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--diffusion-device", type=int, default=0)
    parser.add_argument("--lgm-device", type=int, default=1)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--mrc-metric", choices=("lpips", "l1"), default="lpips")
    parser.add_argument(
        "--overlap-reward",
        action="store_true",
        help=(
            "Pipeline GPU-0 MVDream sampling with GPU-1 LGM/MRC scoring. This reduces T4 x2 wall time "
            "without changing trajectories, rewards, or the on-policy update."
        ),
    )
    parser.add_argument(
        "--final-prompt",
        help="Prompt used for post-RL candidates. Defaults to --target-prompt, then the first training prompt.",
    )
    parser.add_argument(
        "--validation-prompt",
        action="append",
        dest="validation_prompts",
        help="Held-out prompt for best-checkpoint selection. Repeat for multiple prompts.",
    )
    parser.add_argument(
        "--validation-samples-per-prompt",
        type=int,
        default=1,
        help="Fixed seeds evaluated per held-out prompt. Use at least 2 for less noisy T4 checkpoint selection.",
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
            "This conservative diagnostic mode is not the paper algorithm; leave it disabled for paper-style on-policy training."
        ),
    )
    parser.add_argument(
        "--kl-early-stop-threshold",
        type=float,
        default=None,
        help=(
            "Stop and restore the best safe checkpoint when fixed-seed validation KL to the base policy reaches this value. "
            "The paper reports 3.2e-4 for Instant3D; that absolute value is architecture-dependent for MVDream."
        ),
    )
    parser.add_argument(
        "--final-candidates",
        type=int,
        default=1,
        help="Generate this many fixed-seed final samples and retain the one with lowest MRC (at least 1).",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class PromptConfiguration:
    """Resolved prompts with no unrelated implicit object categories."""

    training: list[str]
    validation: list[str]
    final: str
    target: str | None


def _nonempty_prompts(values: Sequence[str] | None, option: str) -> list[str]:
    prompts = [value.strip() for value in values or []]
    if any(not prompt for prompt in prompts):
        raise ValueError(f"{option} cannot be empty")
    return prompts


def resolve_prompt_configuration(args: argparse.Namespace) -> PromptConfiguration:
    """Resolve train/validation/final prompts without silently changing object class.

    A real four-view folder belongs to ``kaggle_import_four_views.py``. RLFT
    trains a text-conditioned MVDream policy and therefore still needs a text
    target plus model-sampled DDIM trajectories.
    """

    target_values = _nonempty_prompts([args.target_prompt] if args.target_prompt is not None else [], "--target-prompt")
    target = target_values[0] if target_values else None
    training = _nonempty_prompts(args.prompts, "--prompt")
    validation = _nonempty_prompts(args.validation_prompts, "--validation-prompt")
    final_values = _nonempty_prompts([args.final_prompt] if args.final_prompt is not None else [], "--final-prompt")

    if not training:
        if target is None:
            raise ValueError(
                "RLFT needs an explicit object description. Supply --target-prompt or at least one --prompt. "
                "For four external images, run kaggle_import_four_views.py instead."
            )
        training = [target]
    if not validation:
        # A target-specific run validates the actual requested category at
        # fixed held-out seeds. A multi-category run without a separate
        # validation list reuses its explicit categories rather than silently
        # substituting unrelated demo nouns.
        validation = [target] if target is not None else list(training)
    final = final_values[0] if final_values else (target or training[0])

    unique_training = list(dict.fromkeys(training))
    if len(unique_training) < args.prompts_per_update:
        raise ValueError(
            f"--prompts-per-update={args.prompts_per_update} needs at least that many distinct --prompt values; "
            f"received {len(unique_training)}. Use --prompts-per-update 1 for one target prompt."
        )
    return PromptConfiguration(
        training=training,
        validation=validation,
        final=final,
        target=target,
    )


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

        # CUDA's current device is thread-local. This matters when Kaggle T4x2
        # overlaps this GPU-1 reward call with MVDream sampling on GPU 0.
        self.torch.cuda.set_device(self.device)
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


def balanced_prompt_batches(
    prompts: Sequence[str], updates: int, prompts_per_update: int, seed: int
) -> list[list[str]]:
    """Build reproducible shuffled prompt batches with near-uniform coverage.

    The paper samples training prompts rather than walking a fixed difficulty
    ranking.  At T4 scale, purely random draws can omit a prompt for most of a
    short run, so shuffled cycles preserve randomness while keeping prompt
    counts within one occurrence of each other.
    """

    unique_prompts = list(dict.fromkeys(prompts))
    if updates < 1 or prompts_per_update < 1:
        raise ValueError("updates and prompts_per_update must be positive")
    if len(unique_prompts) < prompts_per_update:
        raise ValueError("not enough distinct prompts for one update")
    rng = np.random.default_rng(seed)
    pool: list[str] = []
    batches: list[list[str]] = []
    for _ in range(updates):
        batch: list[str] = []
        while len(batch) < prompts_per_update:
            if not pool:
                pool = [unique_prompts[index] for index in rng.permutation(len(unique_prompts))]
            candidate = pool.pop(0)
            if candidate in batch:
                pool.append(candidate)
                continue
            batch.append(candidate)
        batches.append(batch)
    return batches


class PerPromptRunningNormalizer:
    """Paper/DDPO per-prompt statistics retained across on-policy updates."""

    def __init__(self, buffer_size: int, min_count: int, epsilon: float = 1e-6) -> None:
        if buffer_size < 1 or min_count < 1:
            raise ValueError("buffer_size and min_count must be positive")
        if min_count > buffer_size:
            raise ValueError("min_count cannot exceed buffer_size")
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.epsilon = epsilon
        self.stats: dict[str, deque[float]] = {}

    def update(self, values: Sequence[float], group_ids: Sequence[str]) -> list[float]:
        """Append the current batch, then normalize from each prompt's running window."""

        if len(values) != len(group_ids):
            raise ValueError("values and group_ids must have the same length")
        values_array = np.asarray(values, dtype=np.float32)
        result = np.empty(len(values_array), dtype=np.float32)
        for group_id in dict.fromkeys(group_ids):
            indexes = [index for index, candidate in enumerate(group_ids) if candidate == group_id]
            current = values_array[indexes]
            history = self.stats.setdefault(group_id, deque(maxlen=self.buffer_size))
            history.extend(float(value) for value in current)
            if len(history) < self.min_count:
                mean = float(values_array.mean())
                standard_deviation = float(values_array.std())
            else:
                history_array = np.asarray(history, dtype=np.float32)
                mean = float(history_array.mean())
                standard_deviation = float(history_array.std())
            denominator = standard_deviation + self.epsilon
            result[indexes] = (current - mean) / denominator
        return result.tolist()

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            group_id: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
            }
            for group_id, values in self.stats.items()
        }


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
    samples_per_prompt: int = 1,
    label: str = "Validation",
) -> dict[str, Any]:
    """Evaluate prompts over fixed independent seeds and aggregate per prompt.

    Appendix C.1 ranks curation prompts by the average of four base-policy
    outputs.  The same aggregation also makes small held-out validation less
    sensitive to one unusually easy or difficult seed.
    """

    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be at least 1")

    was_training = pipe.unet.training
    pipe.unet.eval()
    records: list[dict[str, Any]] = []
    executor_context = ThreadPoolExecutor(max_workers=1) if args.overlap_reward else nullcontext(None)
    pending_scores: list[tuple[int, int, Trajectory, Future[Any]]] = []

    def finish_score(
        prompt_index: int,
        sample_index: int,
        trajectory: Trajectory,
        future: Future[Any] | None = None,
    ) -> None:
        if future is not None:
            future.result()
        records.append(
            {
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "prompt": trajectory.prompt,
                "seed": trajectory.seed,
                "mrc": trajectory.mrc,
                "kl_to_base": trajectory.kl_to_base,
            }
        )
        print(
            f"[{label}] prompt {prompt_index + 1}/{len(prompts)} sample "
            f"{sample_index + 1}/{samples_per_prompt} MRC={trajectory.mrc:.6f} "
            f"KL_base={trajectory.kl_to_base:.6f}",
            flush=True,
        )

    with executor_context as reward_executor:
        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(samples_per_prompt):
                trajectory_seed = seed + prompt_index * samples_per_prompt + sample_index
                trajectory = sample_trajectory(
                    pipe,
                    prompt,
                    trajectory_seed,
                    device,
                    args.num_steps,
                    args.guidance_scale,
                    args.eta,
                    args.elevation,
                )
                sample_destination = (
                    destination / f"prompt_{prompt_index:02d}" / f"sample_{sample_index:02d}"
                )
                if reward_executor is None:
                    score_and_save(trajectory, scorer, sample_destination)
                    finish_score(prompt_index, sample_index, trajectory)
                else:
                    future = reward_executor.submit(score_and_save, trajectory, scorer, sample_destination)
                    pending_scores.append((prompt_index, sample_index, trajectory, future))
                    if len(pending_scores) > 1:
                        finish_score(*pending_scores.pop(0))
        while pending_scores:
            finish_score(*pending_scores.pop(0))

    records.sort(key=lambda record: (record["prompt_index"], record["sample_index"]))
    prompt_records: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_samples = [record for record in records if record["prompt_index"] == prompt_index]
        prompt_records.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "mean_mrc": float(np.mean([record["mrc"] for record in prompt_samples])),
                "std_mrc": float(np.std([record["mrc"] for record in prompt_samples])),
                "mean_kl_to_base": float(np.mean([record["kl_to_base"] for record in prompt_samples])),
                "samples": prompt_samples,
            }
        )
    mean_mrc = float(np.mean([record["mrc"] for record in records]))
    mean_kl_to_base = float(np.mean([record["kl_to_base"] for record in records]))
    result = {
        "samples_per_prompt": samples_per_prompt,
        "mean_mrc": mean_mrc,
        "mean_kl_to_base": mean_kl_to_base,
        "prompt_records": prompt_records,
        "records": records,
    }
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
    *,
    candidates_directory: str = "final_candidates",
    publish_best: bool = True,
    label: str = "Final",
) -> dict[str, Any]:
    """Score final fixed-seed samples and retain the best reconstruction-consistent one.

    This is deliberately reported as inference-time best-of-N selection.  It
    does not pretend that sampling several seeds is an RL improvement or a
    result reported by the original Carve3D paper.
    """

    if args.final_candidates < 1:
        raise ValueError("--final-candidates must be at least 1")
    candidates_root = args.output_dir / candidates_directory
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
            f"[{label} {index + 1}/{args.final_candidates}] MRC={trajectory.mrc:.6f}",
            flush=True,
        )
        if best_trajectory is None or trajectory.mrc < best_trajectory.mrc:
            best_trajectory = trajectory
            best_path = candidate_path

    assert best_trajectory is not None and best_path is not None
    result = {
        "selection": "lowest_mrc_over_fixed_seed_candidates",
        "candidate_count": args.final_candidates,
        "prompt": args.final_prompt,
        "mean_mrc": float(np.mean([record["mrc"] for record in records])),
        "std_mrc": float(np.std([record["mrc"] for record in records])),
        "selected": min(records, key=lambda record: record["mrc"]),
        "candidates": records,
    }
    candidates_root.mkdir(parents=True, exist_ok=True)
    (candidates_root / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if publish_best:
        final_destination = args.output_dir / "final_evaluation"
        shutil.rmtree(final_destination, ignore_errors=True)
        shutil.copytree(best_path, final_destination)
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
    if args.curate_prompt_count < 0:
        raise ValueError("--curate-prompt-count cannot be negative")
    if args.curation_samples_per_prompt < 1:
        raise ValueError("--curation-samples-per-prompt must be at least 1")
    if args.samples_per_epoch % args.prompts_per_update != 0:
        raise ValueError("--samples-per-epoch must divide evenly by --prompts-per-update")
    samples_per_prompt = args.samples_per_epoch // args.prompts_per_update
    if samples_per_prompt < 2:
        raise ValueError("Each prompt needs at least two trajectories for per-prompt advantages")
    if args.num_steps < 2:
        raise ValueError("--num-steps must be at least 2")
    if args.validation_every < 1 or args.early_stop_patience < 1:
        raise ValueError("--validation-every and --early-stop-patience must be at least 1")
    if args.validation_samples_per_prompt < 1:
        raise ValueError("--validation-samples-per-prompt must be at least 1")
    if args.min_validation_improvement < 0:
        raise ValueError("--min-validation-improvement cannot be negative")
    if args.advantage_clip <= 0:
        raise ValueError("--advantage-clip must be positive")
    if args.stat_buffer_epochs < 1:
        raise ValueError("--stat-buffer-epochs must be at least 1")
    if args.kl_early_stop_threshold is not None and args.kl_early_stop_threshold <= 0:
        raise ValueError("--kl-early-stop-threshold must be positive")
    if args.final_candidates < 1:
        raise ValueError("--final-candidates must be at least 1")
    prompt_configuration = resolve_prompt_configuration(args)
    prompts = prompt_configuration.training
    validation_prompts = prompt_configuration.validation
    # Downstream final-candidate helpers consume the resolved value from args.
    args.final_prompt = prompt_configuration.final
    print("Resolved RLFT prompts (these condition MVDream; they are not captions read from input images):")
    for index, prompt in enumerate(prompts, start=1):
        print(f"  train {index}: {prompt}")
    for index, prompt in enumerate(validation_prompts, start=1):
        print(f"  validation {index}: {prompt}")
    print(f"  final: {args.final_prompt}", flush=True)

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
    curation: dict[str, Any] = {"enabled": False, "candidates": list(prompts), "selected": list(prompts)}
    if args.curate_prompt_count:
        curation_candidates = list(dict.fromkeys(prompts))
        if args.curate_prompt_count < args.prompts_per_update:
            raise ValueError("--curate-prompt-count must be at least --prompts-per-update")
        if args.curate_prompt_count > len(curation_candidates):
            raise ValueError("--curate-prompt-count cannot exceed the number of distinct training prompts")
        print(
            f"[Curation] scoring {len(curation_candidates)} base-policy prompts; selecting "
            f"{args.curate_prompt_count} with highest mean MRC over "
            f"{args.curation_samples_per_prompt} samples/prompt...",
            flush=True,
        )
        curation_evaluation = evaluate_prompts(
            pipe,
            curation_candidates,
            scorer,
            args.output_dir / "curation" / "base_policy",
            args.seed + 2_000_000,
            diffusion_device,
            args,
            samples_per_prompt=args.curation_samples_per_prompt,
            label="Curation",
        )
        ranked = sorted(
            curation_evaluation["prompt_records"],
            key=lambda record: record["mean_mrc"],
            reverse=True,
        )
        prompts = [record["prompt"] for record in ranked[: args.curate_prompt_count]]
        curation = {
            "enabled": True,
            "criterion": "highest base-policy mean MRC over independent outputs (lowest mean reward)",
            "samples_per_candidate": args.curation_samples_per_prompt,
            "candidates": curation_evaluation["prompt_records"],
            "selected": list(prompts),
        }
        for index, record in enumerate(ranked, start=1):
            marker = "selected" if record["prompt"] in prompts else "not selected"
            print(
                f"[Curation] rank {index}: mean_MRC={record['mean_mrc']:.6f} "
                f"std={record['std_mrc']:.6f} ({marker}) {record['prompt']}",
                flush=True,
            )
    prompt_batches = balanced_prompt_batches(
        prompts,
        args.epochs,
        args.prompts_per_update,
        args.seed + 17_003,
    )
    prompt_occurrences = {
        prompt: sum(prompt in batch for batch in prompt_batches)
        for prompt in prompts
    }
    print(
        "[Prompt schedule] balanced shuffled curriculum: "
        + ", ".join(f"{count}x {prompt}" for prompt, count in prompt_occurrences.items()),
        flush=True,
    )
    stat_buffer_size = samples_per_prompt * args.stat_buffer_epochs
    reward_stat_tracker = PerPromptRunningNormalizer(stat_buffer_size, samples_per_prompt)
    kl_stat_tracker = PerPromptRunningNormalizer(stat_buffer_size, samples_per_prompt)
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
        samples_per_prompt=args.validation_samples_per_prompt,
        label="Baseline validation",
    )
    baseline_final_selection = evaluate_final_candidates(
        pipe,
        scorer,
        diffusion_device,
        args,
        candidates_directory="baseline_final_candidates",
        publish_best=False,
        label="Baseline final",
    )
    best_validation_mrc = baseline_validation["mean_mrc"]
    best_epoch = -1
    best_state = lora_state(pipe.unet)
    save_lora(pipe.unet, args.output_dir / "checkpoints" / "best_lora.pt")
    validations_without_improvement = 0
    training_stop_reason = "max_epochs"
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
    print(
        f"[RL 0/{args.epochs}] baseline validation MRC={best_validation_mrc:.6f} "
        f"KL_base={baseline_validation['mean_kl_to_base']:.6f}",
        flush=True,
    )

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
        epoch_prompts = prompt_batches[epoch]
        print(
            f"[RL {epoch + 1}/{args.epochs}] sampling {args.samples_per_epoch} trajectories "
            f"across {args.prompts_per_update} prompt(s)",
            flush=True,
        )
        executor_context = ThreadPoolExecutor(max_workers=1) if args.overlap_reward else nullcontext(None)
        pending_scores: list[tuple[int, int, int, Trajectory, Future[Any]]] = []

        def finish_score(
            sample_index: int,
            prompt_index: int,
            prompt_sample_index: int,
            trajectory: Trajectory,
            future: Future[Any] | None = None,
        ) -> None:
            if future is not None:
                future.result()
            trajectories.append(trajectory)
            completed = len(trajectories)
            print(
                f"[RL {epoch + 1}/{args.epochs}] sample {completed}/{args.samples_per_epoch} "
                f"prompt={prompt_index + 1}/{args.prompts_per_update} "
                f"trajectory={prompt_sample_index + 1}/{samples_per_prompt} seed={trajectory.seed} "
                f"MRC={trajectory.mrc:.6f} KL_base={trajectory.kl_to_base:.6f}",
                flush=True,
            )
            write_training_progress(
                progress_file,
                status="running",
                stage="sampling_and_reward",
                current_epoch=epoch + 1,
                total_epochs=args.epochs,
                current_sample=completed,
                samples_per_epoch=args.samples_per_epoch,
                last_sample_index=sample_index,
                last_mrc=trajectory.mrc,
                last_kl_to_base=trajectory.kl_to_base,
                best_epoch=best_epoch,
                best_validation_mrc=best_validation_mrc,
            )

        with executor_context as reward_executor:
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
                    destination = (
                        args.output_dir / "epochs" / f"epoch_{epoch:03d}" / f"sample_{sample_index:03d}"
                    )
                    if reward_executor is None:
                        score_and_save(trajectory, scorer, destination)
                        finish_score(sample_index, prompt_index, prompt_sample_index, trajectory)
                    else:
                        future = reward_executor.submit(score_and_save, trajectory, scorer, destination)
                        pending_scores.append(
                            (sample_index, prompt_index, prompt_sample_index, trajectory, future)
                        )
                        # Keep at most one score queued behind the active GPU-1
                        # job while GPU 0 samples the next trajectory.
                        if len(pending_scores) > 1:
                            finish_score(*pending_scores.pop(0))
            while pending_scores:
                finish_score(*pending_scores.pop(0))

        rewards = np.asarray([trajectory.reward for trajectory in trajectories], dtype=np.float32)
        kl_values = np.asarray([trajectory.kl_to_base for trajectory in trajectories], dtype=np.float32)
        group_ids = [trajectory.prompt for trajectory in trajectories]
        # Paper Eqs. (6), (9), and Appendix C.2: normalize reward and base-KL
        # separately per prompt using a running window of roughly three prompt
        # appearances, instead of discarding history after every tiny T4 batch.
        reward_advantages = reward_stat_tracker.update(rewards.tolist(), group_ids)
        kl_advantages = kl_stat_tracker.update(kl_values.tolist(), group_ids)
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
            "stat_buffer_size_per_prompt": stat_buffer_size,
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
                samples_per_prompt=args.validation_samples_per_prompt,
                label=f"Validation {epoch + 1}",
            )
            epoch_record["validation_mrc"] = validation["mean_mrc"]
            epoch_record["validation_kl_to_base"] = validation["mean_kl_to_base"]
            kl_limit_reached = (
                args.kl_early_stop_threshold is not None
                and validation["mean_kl_to_base"] >= args.kl_early_stop_threshold
            )
            if kl_limit_reached:
                training_stop_reason = "validation_kl_threshold"
                epoch_record["update_accepted"] = False
                epoch_record["early_stop_reason"] = "validation_kl_threshold"
                print(
                    f"[RL] paper-style KL early stopping: validation KL_base="
                    f"{validation['mean_kl_to_base']:.6f} reached threshold="
                    f"{args.kl_early_stop_threshold:.6f}. Restoring best safe LoRA.",
                    flush=True,
                )
                write_training_progress(
                    progress_file,
                    status="early_stopped",
                    stage="training_complete",
                    reason="validation_kl_threshold",
                    current_epoch=epoch + 1,
                    total_epochs=args.epochs,
                    best_epoch=best_epoch,
                    best_validation_mrc=best_validation_mrc,
                    validation_kl_to_base=validation["mean_kl_to_base"],
                )
                break
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
                    training_stop_reason = "validation_mrc_plateau"
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
    final_selection["paired_mean_mrc_improvement"] = (
        baseline_final_selection["mean_mrc"] - final_selection["mean_mrc"]
    )
    for selection_path in (
        args.output_dir / "final_candidates" / "selection.json",
        args.output_dir / "final_evaluation" / "selection.json",
    ):
        selection_path.write_text(json.dumps(final_selection, indent=2), encoding="utf-8")
    print(
        f"[Final comparison] same-seed mean MRC: base={baseline_final_selection['mean_mrc']:.6f} "
        f"post_RL={final_selection['mean_mrc']:.6f} "
        f"improvement={final_selection['paired_mean_mrc_improvement']:.6f}",
        flush=True,
    )
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
            "prompt_curation": curation,
            "prompt_schedule": {
                "method": "seeded balanced shuffled cycles",
                "occurrences": prompt_occurrences,
                "batches": prompt_batches,
            },
            "curation_samples_per_prompt": args.curation_samples_per_prompt,
            "ddim_steps": args.num_steps,
            "eta": args.eta,
            "timestep_loss_reduction": args.timestep_loss_reduction,
            "advantage_clip": args.advantage_clip,
            "per_prompt_stat_tracking": {
                "buffer_epochs": args.stat_buffer_epochs,
                "buffer_size": stat_buffer_size,
                "min_count": samples_per_prompt,
                "reward": reward_stat_tracker.summary(),
                "kl_to_base": kl_stat_tracker.summary(),
            },
            "validation_every": args.validation_every,
            "validation_samples_per_prompt": args.validation_samples_per_prompt,
            "early_stop_patience": args.early_stop_patience,
            "kl_early_stop_threshold": args.kl_early_stop_threshold,
            "min_validation_improvement": args.min_validation_improvement,
            "learning_rate": args.learning_rate,
            "adamw": {"betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 1e-4},
            "transactional_validation": args.transactional_validation,
        },
        "devices": {"mvdream_rl": args.diffusion_device, "lgm_mrc": args.lgm_device},
        "overlap_reward": args.overlap_reward,
        "mrc_metric": args.mrc_metric,
        "lora_projection_count": lora_projection_count,
        "prompts": {
            "target": prompt_configuration.target,
            "training": prompts,
            "validation": validation_prompts,
            "final": args.final_prompt,
        },
        "baseline_validation": baseline_validation,
        "baseline_final_selection": baseline_final_selection,
        "best_validation_mrc": best_validation_mrc,
        "best_epoch": best_epoch,
        "training_stop_reason": training_stop_reason,
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
        final_mean_mrc=final_selection["mean_mrc"],
        paired_mean_mrc_improvement=final_selection["paired_mean_mrc_improvement"],
        training_stop_reason=training_stop_reason,
    )
    print(f"\nCompleted open RLFT. Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
