# Paper-faithful RLFT profile for Kaggle T4 x2

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
| Statistics | Per-prompt window of about three epochs | Same three-appearance window |
| LoRA | UNet attention, rank 4, fp32 | Same |
| Optimizer | AdamW, `3e-4`, `(0.9, 0.999)`, eps `1e-8`, wd `1e-4` | Same except lr `3e-5` |
| Batch | 768 over 48 A100 80 GB | 8 trajectories over T4 x2 |
| Denoising | 100 Instant3D steps, CFG 5 | 30 MVDream steps, CFG 5 |
| Training set | 30 curated low-reward prompts | Score 8 explicit candidates once; retain the 4 highest-MRC prompts |
| Duration | 55 epochs / 16.5 hours | At most 12 updates with early stopping |
| KL stop | `3.2e-4` on Instant3D | Logged and enabled, but not numerically transferable across architectures |

The T4 learning rate is the paper rate scaled by the square root of the batch
ratio: `3e-4 × sqrt(8 / 768) ≈ 3e-5`. Linear scaling would be safer but much
slower; retaining `3e-4` with batch 8 produces high-variance destructive
updates. This is an engineering adaptation, not a claim reproduced by the
paper.

## Fast convergence safeguards

- The one-cell profile skips the independent direct-photo reconstruction by
  default, so Kaggle starts RLFT immediately after dependency/model setup. Set
  `RUN_DIRECT_RECONSTRUCTION=True` only when that separate PLY is required.
- A base-policy curation pass ranks the eight candidate prompts by MRC and
  trains only the four lowest-reward prompts, following Appendix C.1 at reduced
  T4 scale. The ranking is saved under `curation/base_policy`.
- GPU 0 samples the next MVDream trajectory while GPU 1 scores the previous
  trajectory with LGM/MRC. This changes wall time only, not the update.
- Reward and KL statistics persist for three prompt appearances. The former
  batch-local normalization threw away history and amplified T4 variance.
- Transactional rollback is disabled. It was not part of the paper and could
  keep restoring the zero-LoRA base after every small update.
- Fixed-seed held-out MRC selects the best safe LoRA. Validation KL stops the
  run before excessive distribution shift; MRC plateau stopping avoids paying
  for updates with no measured improvement.
- Replay error, gradient norm, training MRC, validation MRC, and base KL are
  written to `rlft_metrics.json` and `training_progress.json`.

There is no valid promise that a two-T4 run will converge to the paper's
reported result. Treat it as converged only when held-out MRC decreases over
the base checkpoint, replay error remains small, gradients are finite, and KL
stays below the configured threshold. Training MRC alone is not sufficient.
