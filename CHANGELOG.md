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
- The open RLFT LoRA now follows the paper's applicable mixed-precision recipe:
  rank 4 fp32 LoRA weights on a frozen fp16 base UNet, with the paper's `3e-4`
  AdamW learning-rate default and `0.2` KL coefficient.
- Move newly injected fp32 LoRA tensors to the MVDream CUDA device before the
  first sample, fixing the Kaggle CPU/CUDA matrix-device mismatch.
- Add an initial T4 x2 quality profile: 16
  updates, four same-prompt trajectories per update, 30 DDIM steps, held-out
  validation, best-LoRA restoration, and early stopping on validation MRC.
- Add concise live `[RL epoch/total]` progress lines and a continuously updated
  `training_progress.json` file for Kaggle monitoring.
- Replace the unsafe T4 quality profile with a conservative small-batch LoRA
  learning rate (`1e-5`), eight trajectories per update, transactional
  validation rollback, and fixed-seed best-of-eight final selection. Rejected
  LoRA updates now restore both weights and AdamW moments instead of degrading
  the next update.
- Add optional direct-four-view elevation search. The Kaggle profile evaluates
  -10/-5/0/+5/+10 degrees and keeps the reconstruction with lowest same-pose
  LPIPS MRC, recording every candidate in `metrics.json`.
- Replace the direct-gradient Gaussian refinement path after its CUDA
  rasterizer backward kernel caused an illegal-memory-access crash on Kaggle
  T4. The direct quality path now uses only safe forward rendering: a global
  RGB affine fit plus modest opacity/scale candidates, retaining the
  lowest-LPIPS-MRC Gaussian result and printing `[Appearance]` progress.
- Show a labelled 2×2 preview of the exact 000/090/180/270 source files at
  the start of the one-cell Kaggle run, before LGM is loaded or calibration
  begins.
- Accept the obsolete `--refine-*` flags used by previously copied Kaggle
  cells, automatically mapping them to safe forward-only appearance search so
  the old cell does not fail argument parsing.
- Re-enable the actual public-MVDream LoRA RLFT stage by default after direct
  four-photo reconstruction, while documenting that it writes a separate
  prompt-model experiment rather than changing the exported direct PLY.
- Replace the ineffective KL-to-behaviour quadratic with Carve3D's sampled
  trajectory KL against frozen base MVDream (`log p_current - log p_base`),
  normalized per prompt and folded into the score-function advantage.
- Compute DDIM Gaussian log probabilities in fp32 on the exact fp16 action
  returned to the next denoising step, average the small-batch T4 loss over
  stochastic timesteps, keep sample/replay in UNet eval mode, and print replay
  log-probability error to expose policy mismatches and gradient spikes.
- Use all eight T4 trajectories for one prompt per update, clip advantages at
  5, and schedule selected prompts through seeded shuffled cycles. Sample logs
  now include the prompt slot, within-prompt trajectory index, and seed so
  overlapped GPU reward output cannot be mistaken for an uneven batch.
- Match the public LGM text pipeline by removing MVDream backgrounds,
  recentering foregrounds, and compositing on white before reward
  reconstruction. The reusable rembg session is pinned to ONNX CPU to avoid
  Kaggle CUDA-provider warning spam.
- Set AdamW betas, epsilon, and weight decay explicitly; the previous implicit
  PyTorch weight decay (`1e-2`) was 100x the paper's reported `1e-4`.
- Add a detailed v3 non-convergence diagnosis to `docs/OPEN_RLFT.md` and move
  the one-cell run to a fresh `carve3d-open-rlft-output-v4` directory.
- Keep external four-image reconstruction as stage A, then continue into the
  requested prompt-RL experiment by default. Its train/validation/final prompts
  consistently describe the steel staircase shown by the source views; users
  can still set `RUN_PROMPT_RLFT=False` for a direct-only run.
- Remove hidden chair/teapot RLFT defaults. `open_mvdream_rlft.py` now requires
  an explicit `--target-prompt` or `--prompt`, resolves validation/final prompts
  from that same configuration, and fails before model loading if the prompt
  setup is missing or inconsistent.
- Add a paper-style Kaggle T4 x2 profile: general low-reward-style training
  prompts with the steel staircase held out, rank-4 fp32 LoRA, guarded
  `7.5e-5` learning rate, persistent three-appearance reward/KL statistics, and
  paper-style validation-KL early stopping. Per-update transactional rollback
  is no longer used by the one-cell profile.
- Recreate the unreleased Appendix-C.1 dataset recipe as 30 checked-in
  "complex but not too creative" candidates. Rank every candidate by the mean
  of four base-policy outputs, as specified by the paper, then train the ten
  with highest MRC/lowest reward for three balanced appearances each.
- Validate four held-out prompts with two fixed seeds each and compare base
  versus best-LoRA final MRC on the same four seeds. The paired mean is reported
  separately from inference-time best-of-four selection.
- Pipeline GPU-0 MVDream sampling with GPU-1 LGM/MRC scoring through
  `--overlap-reward`, increasing dual-T4 utilization without changing the
  sampled trajectories or on-policy objective.
- Document the exact paper/main-versus-public substitutions and convergence
  criteria in `docs/PAPER_T4X2.md`.

## Earlier work — `implement_full_pipeline`

- Added a reproducible open LGM adapter for real four-view reconstruction,
  same-camera rendering, and MRC.
- Added an MVDream prompt baseline for smoke testing only. It remains separate
  from this direct-four-image path and is not an exact replacement for the
  unreleased Carve3D multiview model.
