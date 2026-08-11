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

`code.txt` uses the strongest practical T4 x2 profile: 16 RL updates, four
same-prompt stochastic trajectories per update, 30 DDIM steps, held-out
fixed-seed validation every two updates, and early stopping. It retains
`best_lora.pt` by lowest validation MRC, then restores it for the final result.
This demonstrates the real data path and update, not the paper's training
scale. The paper trained for 55 epochs on 48 A100 80GB GPUs, with batch size
768, taking 16.5 hours. Increasing `RL_EPOCHS` and `RL_DDIM_STEPS` further
increases cost sharply and still does not make the MVDream/LGM substitute
numerically comparable to Carve3DM.

The LoRA part follows the paper's reported recipe where it is applicable:
rank 4, frozen fp16 base networks, fp32 LoRA UNet weights, AdamW learning rate
`3e-4`, and a `0.2` KL coefficient. The source MVDream/LGM replacement has a
different architecture and reward implementation, so this is not a claim that
the resulting weights are compatible with the unavailable Instant3D model.
