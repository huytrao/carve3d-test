# Open RLFT reproduction

`code.txt` runs the direct external-image workflow that can be reproduced with
public models on Kaggle T4 x2 and contains an optional prompt-RLFT experiment.
It is deliberately not labelled an exact Carve3D reproduction.

| Paper component | Paper/released-code status | Runnable public replacement |
| --- | --- | --- |
| Instant3D-10K 1024px four-view SDXL | Not released | `ashawkey/mvdream-sd2.1-diffusers` |
| Instant3D sparse-view LRM NeRF | Not released | official LGM `model_fp16_fixrot.safetensors` |
| MRC | `rewards.py` has TODO render functions | LGM same-camera render + foreground-crop LPIPS |
| RLFT | released training file imports unavailable prompt/model components | on-policy MVDream LoRA score-function update |

The paper's Appendix C.2 states that it leaves the sparse-view LRM and
Instant3D as abstract functions because they do not plan to release code.
That is also visible locally: `config/dgx.py` points to
`/path/to/your_multiview_diffusion_model`, `carve3d_train.py` imports a missing
root `prompts.py`, and `rewards.py` leaves `nerf_render` and the LRM model as
TODOs.

## What runs

1. Four captured views are validated in 0/90/180/270 degree order, reconstructed
   with public LGM, rendered at the same poses, then measured with LPIPS MRC.
2. Public MVDream samples a stochastic four-view DDIM trajectory on GPU 0.
3. Public LGM renders the corresponding reconstruction on GPU 1. Its foreground
   crop LPIPS distance is MRC; the RL reward is `-MRC`.
4. For every sampled transition, the runner also evaluates the same action
   under frozen base MVDream (LoRA temporarily disabled). The trajectory KL is
   the timestep-average `log p_current - log p_base`, matching paper Eq. (8).
5. Reward and base-KL are normalized independently inside each prompt group.
   Their combined advantage drives exactly one score-function (pure on-policy)
   LoRA update per sampled batch, then only LoRA tensors are saved.
6. MVDream outputs are background-matted, recentered, and composited on white
   before LGM, matching LGM's official text-to-3D preprocessing.
7. A fresh post-RL prompt sample is reconstructed and evaluated.

Steps 2–7 run only when `RUN_PROMPT_RLFT=True`; the paper-style one-cell profile
enables that switch and disables the unrelated direct reconstruction stage by
default so RL logs start after setup. External photographs are valid LGM/MRC inputs
in step 1, but cannot be substituted for MVDream policy samples in steps 2–7:
policy-gradient replay requires the DDIM states, actions, and log probabilities
that produced each sample.

When its RLFT switch is enabled, `code.txt` uses the paper-faithful T4 x2
profile: at most 12 updates, two prompts per update, four stochastic
trajectories per prompt, and the official public MVDream/LGM setting of 30
DDIM steps. A reduced Appendix-C.1 curation pass scores eight candidates and
retains the four highest-MRC (lowest-reward) prompts. Timestep losses are averaged so their scale does not grow with the
step count. Reward and KL normalization retain a three-appearance per-prompt
window, matching the paper's approximately three-epoch tracker. Fixed-seed
validation selects the best LoRA; paper-style KL and validation-plateau early
stopping replace the former per-update transactional rollback. Four fixed final
seeds are reported explicitly as inference-time best-of-N selection.

The paper trained for 55 epochs on 48 A100 80GB GPUs, with batch size 768,
taking 16.5 hours. Increasing this Kaggle profile further increases cost
sharply and still does not make the MVDream/LGM substitute numerically
comparable to Carve3DM.

The LoRA part follows the paper's reported recipe where it is applicable:
rank 4, frozen fp16 base networks, fp32 LoRA UNet weights, AdamW learning rate
`3e-4`, and a `0.2` KL coefficient. That `3e-4` was for their batch-768
training; this T4 small-batch profile uses square-root batch scaling,
`3e-4 × sqrt(8/768) ≈ 3e-5`, and keeps the same `0.2` KL coefficient. AdamW
explicitly uses the paper's betas, epsilon,
and `1e-4` weight decay; relying on PyTorch defaults would use `1e-2` weight
decay. The source MVDream/LGM replacement has a different architecture and
reward implementation, so this is not a claim that the resulting weights are
compatible with the unavailable Instant3D model.

## Why the v3 T4 run did not converge

The reported v3 validation moved from `0.923640` to `0.922821`: only
`0.000819`, or about `0.089%`. The epoch-3 gradient norm simultaneously jumped
from roughly `0.009` to `0.311`, more than 30x, while that update improved
validation by only `0.000025`. That pattern is gradient variance, not useful
convergence.

There were several concrete causes:

- The old “KL” was a quadratic difference from the just-sampled behaviour
  policy. Before the single on-policy update those policies are identical, so
  it contributed essentially zero and did not regularize toward base MVDream.
- Each update contained one prompt, and each of eight prompts appeared only
  once. Therefore training means from different epochs measured different
  object difficulty and were not a learning curve.
- DDIM Gaussian density used implicit mixed precision and scored the
  pre-return sample, while replay received the sample after its fp16 cast. At
  late timesteps that rounding is large relative to the small variance, so the
  supposedly identical sample/replay policies can disagree. V4 explicitly
  evaluates fp32 density on the exact returned action.
- The score fed raw MVDream images to LGM even though official LGM inference
  removes backgrounds and recenters objects. Much of the reward therefore
  measured input-distribution mismatch rather than view consistency.
- Validation used only two prompts and one was nearly duplicated in training
  (`small red toy car` versus `red toy car`). Improvements of a few `1e-5`
  were accepted even though they are below a useful decision margin.
- The run collected 64 trajectories. The paper used batch 768 for 55 epochs,
  about 42,240 trajectories, with a different SDXL/Instant3D model and 48
  A100s. Absolute MRC values and convergence rates are not comparable.

The current implementation corrects the algorithmic issues above, prints
`KL_base`, `grad_norm`, and `replay_error`, uses disjoint validation categories,
and pipelines MVDream sampling with LGM reward computation across both T4s. It
still cannot guarantee a large visual improvement at T4 scale; fixed validation
MRC, not per-epoch training MRC, is the acceptance signal. See
`docs/PAPER_T4X2.md` for the complete mapping.

## Real four-photo calibration

The default Kaggle run now prioritizes the requested real-image task. It
tries a small common-elevation grid, starts from LGM's best feed-forward
Gaussian result, calculates a bounded global RGB affine fit from the same four
camera views, and searches that fit plus modest opacity/scale candidates.
Every candidate uses the forward renderer and LPIPS MRC; only the lowest-MRC
Gaussian state is exported. The console lines beginning `[Appearance]` are the
progress to judge, not the unrelated prompt-RL validation metric.

The first direct-gradient refinement implementation is intentionally not used:
the `diff-gaussian-rasterization` CUDA backward kernel produced an illegal
memory-access error on Kaggle T4/Python 3.12. Forward-only calibration is less
expressive, but it is reproducible on this environment and never invokes the
failing kernel.

Optional prompt RL runs after direct reconstruction in `code.txt` as a
separate algorithmic experiment. It fine-tunes the public MVDream prompt model
and still cannot change a reconstruction initialized from the user's four real
images; its checkpoint and MRC are written under the separate RLFT output
directory. The runner now rejects a missing object prompt instead of silently
falling back to unrelated demo nouns. `--target-prompt` is also used as the
validation/final fallback, while repeated `--prompt` values can provide
same-domain wording variants for per-prompt advantages.
