# Changelog

All material pipeline changes are recorded here.

## Unreleased — `import_direct_four_views`

### Added

- `kaggle_import_four_views.py`: direct Kaggle launcher for four real camera
  images. It validates `view_000`, `view_090`, `view_180`, and `view_270`
  before downloading or compiling LGM.
- `run_import_four_views.py`: saved command with the project's Kaggle dataset
  path and output folder prefilled.
- `code_import_four_views.txt`: a single, self-contained Kaggle Code-cell
  version that clones this branch and starts the direct-image pipeline.

### Behavior

- The direct path is `4 real images -> LGM Gaussian reconstruction -> same-pose
  renders -> LPIPS MRC`. It does not load MVDream or accept a prompt.
- On a T4 x2 Kaggle session, LGM uses GPU 1. GPU 0 remains free because one LGM
  inference job is not data-parallel across both GPUs.
- The launcher supports both the supplied input path
  `/kaggle/input/datasets/traoanhuy/carve3d/views` and Kaggle's usual mounted
  path `/kaggle/input/carve3d/views` when no `--input-dir` is supplied.
- It accepts both padded names such as `view_090.png` and common non-padded
  names such as `view_90.png`, then passes the validated paths in fixed camera
  order to the reconstructor.

## Unreleased — `open_rlft_full_pipeline`

### Added

- `open_mvdream_rlft.py`: executable open-model RLFT replacement using public
  MVDream, public LGM, foreground-crop LPIPS MRC reward, an on-policy
  score-function LoRA update, approximate KL control, checkpoint saving, and
  post-RL evaluation across two T4 GPUs.
- `docs/OPEN_RLFT.md`: exact paper-versus-public-replacement matrix and the
  paper-scale compute limitation.

### Changed

- `code.txt` now runs direct four-photo reconstruction/MRC followed by the
  public RLFT cycle, instead of the former prompt-only MVDream smoke test.
- The DDIM log-probability helper handles the deterministic final DDIM action
  without producing NaN/Inf during an RL update.

## Earlier work — `implement_full_pipeline`

- Added a reproducible open LGM adapter for real four-view reconstruction,
  same-camera rendering, and MRC.
- Added an MVDream prompt baseline for smoke testing only. It remains separate
  from this direct-four-image path and is not an exact replacement for the
  unreleased Carve3D multiview model.
