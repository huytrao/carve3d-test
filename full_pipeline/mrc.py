"""Multi-view reconstruction consistency (MRC) evaluation helpers.

The Carve3D paper defines MRC as the perceptual distance between each input
view and a rendering of the reconstructed representation at the same camera.
This module keeps that definition independent from the reconstruction backend.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class ViewScore:
    """Metric result for one camera view."""

    view: int
    lpips: float | None
    l1: float
    crop: list[int]


def _square_crop(image: torch.Tensor, threshold: float = 0.95, padding: int = 5) -> list[int]:
    """Return a padded square foreground box as ``[left, top, right, bottom]``.

    Inputs in this project use a white background, the convention in the
    Carve3D reward code.  Returning the full image for an empty mask is a
    deliberate safe fallback for real-world images with a failed matte.
    """

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected CHW RGB tensor, got {tuple(image.shape)}")
    _, height, width = image.shape
    mask = (image < threshold).any(dim=0)
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        return [0, 0, width, height]

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    side = max(right - left, bottom - top)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    left = center_x - side // 2 - padding
    top = center_y - side // 2 - padding
    right = left + side + 2 * padding
    bottom = top + side + 2 * padding

    # Shift instead of only clipping where possible.  This preserves a square
    # crop near the image border.
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    return [max(0, left), max(0, top), min(width, right), min(height, bottom)]


def _crop_and_resize(image: torch.Tensor, crop: list[int], size: int) -> torch.Tensor:
    left, top, right, bottom = crop
    cropped = image[:, top:bottom, left:right].unsqueeze(0)
    if cropped.shape[-1] < 2 or cropped.shape[-2] < 2:
        cropped = image.unsqueeze(0)
    return F.interpolate(cropped, (size, size), mode="bilinear", align_corners=False)


def _lpips_model(device: torch.device):
    try:
        import lpips
    except ImportError as exc:  # pragma: no cover - documented runtime error
        raise RuntimeError(
            "LPIPS is required for MRC. Run `pip install lpips` or use "
            "`--mrc-metric l1` only for a smoke test."
        ) from exc
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def compute_mrc(
    inputs: torch.Tensor,
    rendered: torch.Tensor,
    angles: Iterable[int] = (0, 90, 180, 270),
    metric: str = "lpips",
    crop_size: int = 256,
    lpips_model=None,
) -> tuple[dict, list[ViewScore]]:
    """Compute Carve3D-style MRC between four source and rendered views.

    Args:
        inputs: RGB tensor ``[4, 3, H, W]`` in ``[0, 1]``.
        rendered: RGB tensor with the same view order in ``[0, 1]``.
        metric: ``lpips`` (the paper metric) or ``l1`` for a dependency-free
            smoke test.  LPIPS is reported as a distance: lower is better.
    """

    if inputs.ndim != 4 or rendered.ndim != 4 or inputs.shape[:2] != (4, 3) or rendered.shape[:2] != (4, 3):
        raise ValueError("inputs and rendered must both be [4, 3, H, W]")
    if metric not in {"lpips", "l1"}:
        raise ValueError("metric must be 'lpips' or 'l1'")

    device = rendered.device
    source = inputs.to(device=device, dtype=torch.float32)
    target = rendered.to(device=device, dtype=torch.float32)
    if metric == "lpips" and lpips_model is None:
        lpips_model = _lpips_model(device)
    scores: list[ViewScore] = []

    with torch.no_grad():
        for angle, source_view, rendered_view in zip(angles, source, target):
            crop = _square_crop(source_view)
            source_crop = _crop_and_resize(source_view, crop, crop_size)
            rendered_crop = _crop_and_resize(rendered_view, crop, crop_size)
            l1 = F.l1_loss(source_crop, rendered_crop).item()
            lpips_value = None
            if lpips_model is not None:
                lpips_value = lpips_model(source_crop * 2 - 1, rendered_crop * 2 - 1).mean().item()
            scores.append(ViewScore(view=int(angle), lpips=lpips_value, l1=l1, crop=crop))

    summary = {
        "metric": metric,
        "mrc": sum(score.lpips if metric == "lpips" else score.l1 for score in scores) / len(scores),
        "lower_is_better": True,
        "views": [asdict(score) for score in scores],
    }
    return summary, scores
