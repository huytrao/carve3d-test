# Open RLFT reproduction

`code.txt` runs the complete workflow that can be reproduced with public
models on Kaggle T4 x2. It is deliberately not labelled an exact Carve3D
reproduction.

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
4. The runner normalizes rewards into advantages and makes exactly one
   score-function (pure on-policy) LoRA update per sampled batch. It records an
   approximate KL-to-behaviour-policy penalty, then saves only LoRA tensors.
5. A fresh post-RL prompt sample is reconstructed and evaluated.

`code.txt` uses the safe quality profile for T4 x2: eight updates, eight
same-prompt stochastic trajectories per update, 40 DDIM steps, and
transactional held-out validation after every update. An update is accepted
only if its fixed-seed validation MRC is lower; otherwise the LoRA tensors and
AdamW state are restored. It also selects the lowest-MRC result across eight
fixed final seeds, recorded explicitly as inference-time best-of-N selection.
This prevents the small noisy batches from silently accumulating worse LoRA
updates; it does not make the selection a paper result.

The paper trained for 55 epochs on 48 A100 80GB GPUs, with batch size 768,
taking 16.5 hours. Increasing this Kaggle profile further increases cost
sharply and still does not make the MVDream/LGM substitute numerically
comparable to Carve3DM.

The LoRA part follows the paper's reported recipe where it is applicable:
rank 4, frozen fp16 base networks, fp32 LoRA UNet weights, AdamW learning rate
`3e-4`, and a `0.2` KL coefficient. That `3e-4` was for their batch-768
training; this T4 small-batch profile deliberately uses `1e-5` and keeps the
same `0.2` KL coefficient. The source MVDream/LGM replacement has a different
architecture and reward implementation, so this is not a claim that the
resulting weights are compatible with the unavailable Instant3D model.

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

Prompt RL remains in `code.txt` behind `RUN_PROMPT_RLFT = False` for
algorithmic research. It is disabled by default because it fine-tunes the
public MVDream prompt model and cannot change a reconstruction initialized
from the user's four real images.
