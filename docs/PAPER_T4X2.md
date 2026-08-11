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
| Statistics | Per-prompt window of about three epochs | Same three-appearance window |
| LoRA | UNet attention, rank 4, fp32 | Same |
| Optimizer | AdamW, `3e-4`, `(0.9, 0.999)`, eps `1e-8`, wd `1e-4` | Same except lr `7.5e-5` |
| Batch | 768 over 48 A100 80 GB | 8 same-prompt trajectories per update over T4 x2 |
| Denoising | 100 Instant3D steps, CFG 5 | 30 MVDream steps, CFG 5 |
| Prompt preparation | Generate 100 "complex but not too creative" candidates, average 4 base outputs per prompt, retain 30 lowest-reward prompts | 30 recreated candidates, average 4 outputs exactly as Appendix C.1, retain 10 lowest-reward prompts |
| Duration | 55 epochs / 16.5 hours | At most 30 updates; each selected prompt appears 3 times |
| Validation | 415 DreamFusion prompts, 4 outputs each for reported evaluation | 4 held-out prompts, 2 fixed outputs each for checkpoint selection |
| KL stop | `3.2e-4` on Instant3D | Logged and enabled, but not numerically transferable across architectures |

The former `3e-5`, 12-update profile left base KL near `1e-5` and changed the
held-out MRC only slightly. The T4 profile uses `7.5e-5` with 8 trajectories
from one prompt per update, a `3.2e-4` KL guard, and best-validation checkpoint
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
- A base-policy curation pass ranks 30 candidates by mean MRC across four
  independent outputs and trains the ten lowest-reward prompts. The paper found
  that even a training-set size of ten generalized; ten also lets a short T4
  run gather useful same-prompt statistics.
- Each update contains eight trajectories from one prompt. A seeded shuffled
  schedule gives every selected prompt exactly three appearances in 30 updates,
  instead of splitting the tiny batch into noisy groups of four.
- GPU 0 samples the next MVDream trajectory while GPU 1 scores the previous
  trajectory with LGM/MRC. This changes wall time only, not the update.
- Reward and KL statistics persist for three prompt appearances. The former
  batch-local normalization threw away history and amplified T4 variance.
- Transactional rollback is disabled. It was not part of the paper and could
  keep restoring the zero-LoRA base after every small update.
- Two fixed seeds for each of four held-out prompts select the best safe LoRA.
  Validation KL stops the
  run before excessive distribution shift; MRC plateau stopping avoids paying
  for updates with no measured improvement.
- The final report compares base and best-LoRA MRC on the same four seeds. This
  paired mean is the improvement signal; the best-of-four sample is only an
  inference-time output selection.
- Replay error, gradient norm, training MRC, validation MRC, and base KL are
  written to `rlft_metrics.json` and `training_progress.json`.

There is no valid promise that a two-T4 run will converge to the paper's
reported result. Treat it as converged only when held-out MRC decreases over
the base checkpoint, replay error remains small, gradients are finite, and KL
stays below the configured threshold. Training MRC alone is not sufficient.
