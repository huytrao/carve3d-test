# Checkpoint research and design choices

## What the released Carve3D repository omits

The upstream README says that it does not contain the SDXL multi-view diffusion
model from Instant3D nor a sparse-view reconstruction model.  Its MRC reward
also has explicit TODOs for the reconstructor and renderer.  Therefore there is
no official, runnable checkpoint that can turn the stock repository into the
paper's exact `Carve3DM` model.

The implementation in this branch does not fabricate that missing artifact. It
uses public model artifacts to provide the complete *operational* pipeline and
labels the prompt route as a baseline.

## Selected models

| Role | Selected artifact | Why it is used | Runtime / license |
| --- | --- | --- | --- |
| Four-view reconstructor | [3DTopia/LGM](https://github.com/3DTopia/LGM), commit `fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c` | Its model consumes exactly four fixed views and predicts Gaussian splats; its official renderer can render at the same camera poses required by MRC. | Official README estimates about 10 GB VRAM. Code MIT. |
| Reconstruction checkpoint | [`ashawkey/LGM` `model_fp16_fixrot.safetensors`](https://huggingface.co/ashawkey/LGM/blob/main/model_fp16_fixrot.safetensors) | Official fixed-rotation inference checkpoint; LGM's README explicitly recommends this filename following their rotation bug fix. | 830 MB, SHA-256 `744d6324656342c64f871308e73db97f0eb51858d94329b30090e986a6d050ab`, MIT. |
| Prompt-to-four-view baseline | [`ashawkey/mvdream-sd2.1-diffusers`](https://huggingface.co/ashawkey/mvdream-sd2.1-diffusers/tree/main) | Public diffusers-format four-view MVDream model already used by the official LGM code. | 2.59 GB, OpenRAIL. |

## Alternatives evaluated

* [OpenLRM](https://github.com/3DTopia/OpenLRM) is a strong open reconstruction
  candidate and publishes `openlrm-mix-*-1.1` checkpoints, but its documented
  inference API accepts **one** input image. It would need a trained fusion
  adapter to consume the four real views without discarding information.
* LGM's stock `infer.py` also accepts a directory, but it treats every image as
  a separate single-image job and uses ImageDream to synthesize new views. The
  `full_pipeline/run.py` adapter is intentionally different: it builds LGM's
  `[view_000, view_090, view_180, view_270]` tensor directly and attaches the
  matching official ray embeddings.
* The exact Instant3D/Carve3D diffusion checkpoints are not released, as noted
  by Carve3D. Replacing them silently would make an invalid reproduction claim.

## Reproducibility and input contract

`scripts/setup_lgm_kaggle.sh` pins source code, checks for CUDA, downloads the
named checkpoint, and records the checkpoint path in `metrics.json`. The runner
requires all four named input views to prevent accidental filesystem ordering.
It also records the view azimuths, elevation, preprocessing switches, source
paths/prompt, MRC metric and per-view scores in `metrics.json`.

MRC is computed in the same spirit as the Carve3D reward: foreground crop from
the source view, render at the corresponding source camera, and average LPIPS
over four views. Lower LPIPS MRC means the reconstructor explains the supplied
multi-view images more consistently.  `--mrc-metric l1` is exposed only as a
smoke-test fallback and never presented as paper MRC.
