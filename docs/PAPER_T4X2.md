# Paper-style RLFT profile for Kaggle T4 x2

The original `main` branch is the authoritative algorithm skeleton, but it is
not executable end-to-end: `config/dgx.py` points to an unavailable Instant3D
checkpoint and prompt function, while `rewards.py` leaves the sparse-view LRM
and renderer as TODOs. This profile preserves the released paper algorithm
where public components permit it and names every necessary substitution.

| Component | Carve3D paper / `main` | Kaggle T4 x2 profile |
| --- | --- | --- |
| Policy | Instant3D SDXL multi-view, unavailable | Public MVDream SD 2.1 |
| Reconstruction reward | Unreleased sparse-view LRM/NeRF | Public LGM Gaussian reconstruction |
| Objective | Pure on-policy REINFORCE, one update per sampled batch | Same |
| Reward | Negative foreground-bbox LPIPS MRC | Same definition, using LGM renders |
| KL | Mean sampled log-policy ratio to frozen base | Same, LoRA disabled for the base pass |
| Advantage | Per-prompt normalized reward minus `0.2 ×` normalized KL | Same |
| Statistics | First epoch fills stats without an update; window size 76 (~3 epochs) | Same no-update warmup; window covers 3 prompt appearances |
| LoRA | UNet attention, rank 4, fp32 | Same |
| Optimizer | AdamW, `3e-4`, `(0.9, 0.999)`, eps `1e-8`, wd `1e-4` | Same except lr `7.5e-5` |
| Batch | 768 over 48 A100 80 GB | 8 same-prompt trajectories per update over T4 x2 |
| Denoising | 100 Instant3D steps, CFG 5 | 30 MVDream steps, CFG 5 |
| Prompt preparation | Generate 100 "complex but not too creative" candidates, average 4 base outputs per prompt, retain 30 lowest-reward prompts | 100 recreated candidates; T4 `100×1 -> top 30×3 more`; retain 10 lowest-reward prompts |
| Duration | 55 epochs / 16.5 hours | 55 paper epochs: epoch 1 stat warmup, then at most 54 updates |
| Validation/test | 415 DreamFusion prompts, 4 outputs each for reported evaluation | 4 prompts × 2 seeds for monitoring; optional `--test-prompt-file` executes 4-seed base/post protocol |
| KL stop | `3.2e-4` on Instant3D | Logged and enabled, but not numerically transferable across architectures |
| Checkpoint | Stop at empirically safe KL | Restore `paper_last_safe`, the latest validated policy below the KL threshold |

The former `3e-5`, 12-update profile left base KL near `1e-5` and changed the
held-out MRC only slightly. The T4 profile uses `7.5e-5` with 8 trajectories
from one prompt per update, a `3.2e-4` KL guard, and last-safe checkpoint
restoration. This learning rate is an engineering adaptation, not a value
reported by the paper.

The exact training prompt dataset is still absent from the authors' repository.
`prompt_sets/paper_style_t4.json` therefore records a transparent recreation of
Appendix C.1 rather than claiming to contain the authors' private prompts. Its
phrases are concise compositional object descriptions; repeated style suffixes
such as "studio product photograph, white background" are not baked into every
prompt.

## Fast convergence safeguards

- The one-cell profile skips the independent direct-photo reconstruction by
  default, so Kaggle starts RLFT immediately after dependency/model setup. Set
  `RUN_DIRECT_RECONSTRUCTION=True` only when that separate PLY is required.
- The checked-in dataset now contains 100 unique candidates. The default T4
  successive pass gives all 100 one sample, shortlists 30, then adds three
  samples so every final-ranking prompt has the paper's four-output mean. Set
  `RL_CURATION_PREFILTER_COUNT=0` to score all 100 four times exactly, at a cost
  of 400 curation rollouts.
- The ten lowest-reward prompts are trained. The paper's scaling experiment
  found that a size of ten still generalized and that optimal data size grows
  with batch size; using all 30 with a batch of only eight would leave most
  prompts statistically undersampled.
- Each update contains eight trajectories from one prompt. A seeded shuffled
  schedule keeps coverage balanced across the 54 optimizer updates.
- GPU 0 samples the next MVDream trajectory while GPU 1 scores the previous
  trajectory with LGM/MRC. This changes wall time only, not the update.
- Epoch 1 samples every selected prompt, fills reward/KL statistics, and does
  not call the optimizer, matching the control flow in `carve3d_train.py`.
  Statistics then persist for three prompt appearances.
- Transactional rollback is disabled. It was not part of the paper and could
  keep restoring the zero-LoRA base after every small update.
- Two fixed seeds for each of four held-out prompts monitor MRC and KL. The T4
  runner disables MRC-plateau stopping and restores the latest checkpoint still
  below the KL threshold, following the paper's stopping criterion.
- Sampled training KL is checked every epoch before the optimizer step. A batch
  at or above the threshold is discarded immediately, so the less frequent
  validation pass cannot hide a KL overshoot.
- The final report compares base and last-safe LoRA MRC on the same four seeds. This
  paired mean is the improvement signal; the best-of-four sample is only an
  inference-time output selection.
- `--test-prompt-file prompts.txt --test-samples-per-prompt 4` optionally runs
  a disjoint test set before and after RL with identical seeds. Supplying all
  415 DreamFusion prompts reproduces the paper's evaluation count but adds
  3,320 T4 rollouts across the two policies.
- Replay error, gradient norm, training MRC, validation MRC, and base KL are
  written to `rlft_metrics.json` and `training_progress.json`.

There is no valid promise that a two-T4 run will converge to the paper's
reported result. Treat it as converged only when held-out MRC decreases over
the base checkpoint, replay error remains small, gradients are finite, and KL
stays below the configured threshold. Training MRC alone is not sufficient.

## Remaining irreducible differences

No configuration change can remove these differences:

- Instant3D-10K and its SDXL multiview pipeline are not released; MVDream SD2.1
  therefore uses 256-pixel views and its official 30 denoising steps instead of
  Instant3D's four 512-pixel tiles and 100 steps.
- The sparse-view NeRF LRM is not released; LGM Gaussian reconstruction is the
  reward backend.
- The authors' exact GPT-4-generated training strings remain a repository TODO;
  `paper_style_t4.json` is explicitly a recipe-compatible recreation.
- Batch 768 on 48 A100-80GB GPUs and fixed learning rate `3e-4` cannot be made
  numerically equivalent to sequential batch 8 on two T4s. The T4 profile uses
  `7.5e-5`, while preserving the optimizer, LoRA, reward, advantage, and KL
  definitions.

These limitations are also serialized under `paper_parity` in every
`rlft_metrics.json`; results are never labeled as an exact Carve3DM reproduction.

Without early KL stopping, the default v8 profile requests 806 generated
rollouts: 190 successive-curation, 80 statistic-warmup, 432 training, 96
baseline/periodic-validation, and 8 final paired-comparison outputs. This count
does not include an optional external paper test file. Actual T4 wall time must
be measured on Kaggle; no local T4 runtime is available in this repository.
