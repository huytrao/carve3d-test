# Carve3D
Code for CVPR 2024 paper, *Carve3D: Improving Multi-view Reconstruction Consistency for Diffusion Models with RL Finetuning*, by Desai Xie, Jiahao Li, Hao Tan, Xin Sun, Zhixin Shu, Yi Zhou, Sai Bi, Sören Pirk, Arie E. Kaufman, a collaboration between Adobe Research, Stony Brook University, TTIC, and Kiel University.  

<img src="https://github.com/desaixie/carve3d/assets/32966374/acb3c6ba-19cd-478f-a923-286c4c4f9a73" width="550"/>

[Project Website](https://desaixie.github.io/carve-3d/)  [ArXiv Paper](https://arxiv.org/abs/2312.13980)

This repository does not contain the implementation of the SDXL-based multiview diffusion model and the sparse-view large reconstruction model proposed in [Instant3D](https://jiahao.ai/instant3d/).
To implement the full Carve3D pipeline, an open-sourced sparse-view reconstruction model is needed, such as [OpenLRM](https://github.com/3DTopia/OpenLRM), [GRM](https://github.com/justimyhxu/grm), [LGM](https://github.com/3DTopia/LGM), etc. 
Pull requests are welcome!

## Runnable full pipeline (four images or a prompt)

This fork adds an executable, reproducible replacement for the non-released
components above.  It uses the official [LGM](https://github.com/3DTopia/LGM)
multi-view Gaussian reconstructer and its public
[`model_fp16_fixrot.safetensors`](https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors)
checkpoint.  It performs the entire evaluation path:

```
4 real views (0/90/180/270) ──> LGM Gaussian reconstruction ──> same-pose renders ──> LPIPS MRC
prompt ──> MVDream public checkpoint ──> 4 views ──> LGM ──> same-pose renders ──> LPIPS MRC
```

The prompt route is an **open-checkpoint baseline**, using
`ashawkey/mvdream-sd2.1-diffusers`; it is not a claim that the unpublished
Carve3D/Instant3D diffusion checkpoint has been reproduced.  The four-image
route is the direct reconstruction/MRC experiment and is the recommended way
to evaluate real captures.

### Kaggle quick start

Open [full_pipeline_run_in_kaggle.ipynb](full_pipeline_run_in_kaggle.ipynb),
turn on *Internet* and a GPU accelerator, then set `INPUT_DIR` to a Kaggle
dataset containing exactly these four views:

```
view_000.png   # front, azimuth 0 degrees
view_090.png   # right, azimuth 90 degrees
view_180.png   # back, azimuth 180 degrees
view_270.png   # left, azimuth 270 degrees
```

The notebook installs the CUDA rasterizer, pins LGM source to commit
`fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c`, downloads the public checkpoint,
and creates `reconstruction.ply`, render grids, an orbit video, and
`metrics.json`.  LGM reports roughly 10 GB VRAM for its official inference
stack; this direct-four-view adapter does not load ImageDream, but a 16 GB
Kaggle GPU remains the practical target.

The Kaggle setup intentionally skips `nvdiffrast`: it is only necessary for
LGM's separate `.ply` → `.glb` mesh conversion and currently fails to build on
Kaggle Python 3.12. The runnable pipeline still writes `reconstruction.ply`,
same-camera renders, orbit video, and MRC. Set `INSTALL_NVDIFFRAST=1` only in
an environment where you need mesh conversion and have a compatible build.

The setup pins `kiui==0.2.3`, the version compatible with LGM's 2024 source.
This avoids the `NameError: Union` raised by recent `kiui` releases on Kaggle
Python 3.12. The next notebook run automatically reapplies setup if an older
dependency stamp is found.

### Kaggle direct import: this project's four real photographs

For the direct-image route, use the dedicated branch and launcher instead of
the prompt files. It reads real captures only; no MVDream checkpoint is loaded.
The four camera names and order are mandatory:

```
view_000.png   # front
view_090.png   # right
view_180.png   # back
view_270.png   # left
```

Your supplied Kaggle dataset directory is already the default:

```bash
python run_import_four_views.py
```

This resolves `/kaggle/input/datasets/traoanhuy/carve3d/views` first and then
Kaggle's normal mount path `/kaggle/input/carve3d/views`. To set a different
input folder without editing the file:

```bash
CARVE3D_INPUT_DIR=/kaggle/input/your-dataset/views python run_import_four_views.py
```

For a literal copy/paste Kaggle Code cell, open
[`code_import_four_views.txt`](code_import_four_views.txt), paste its entire
contents into one cell, and run it. The launcher checks all four filenames
before installing dependencies, reports exactly which view is missing, and
uses GPU 1 for LGM automatically when T4 x2 is enabled. LGM inference itself
is one model on one GPU, so GPU 0 is intentionally left free rather than
pretending that the reconstruction is distributed across both cards.

### Local command

```bash
# First install LGM and the official checkpoint (requires CUDA, nvcc, and internet).
bash scripts/setup_lgm_kaggle.sh ./LGM

python full_pipeline/run.py \
  --lgm-root ./LGM \
  --input-dir /path/to/four_views \
  --output-dir outputs/chair
```

For a public-checkpoint prompt baseline:

```bash
python full_pipeline/run.py \
  --lgm-root ./LGM \
  --prompt "a wooden chair" \
  --output-dir outputs/chair_prompt
```

For Kaggle's **T4 x2** accelerator, run
`kaggle_full_pipeline_prompt.py` instead. It automatically uses GPU 0 for
MVDream and GPU 1 for LGM/MRC, and runs the LGM/checkpoint bootstrap on its
first invocation. Pass `--no-bootstrap` only after setup has completed.

For the simplest reproducible Kaggle prompt run, use the saved command in
`run_full_pipeline.py`:

```bash
python run_full_pipeline.py
```

Override its prompt without editing code with
`CARVE3D_PROMPT="a ceramic teapot, studio product photograph, centered object, white background" python run_full_pipeline.py`.

To upload one Python file to Kaggle (or paste its entire contents into one Code
cell), use `kaggle_full_pipeline_prompt_notebook.py`. Edit its `PROMPT` and
`OUTPUT_DIR` constants, then run it.

For the most direct Kaggle experience, upload
`kaggle_full_pipeline_prompt.ipynb`: it contains the required Markdown and one
Code cell that passes the prompt explicitly rather than relying on notebook
`sys.argv`.

### Prompt-only smoke test

Before a full reconstruction run, test only the public MVDream prompt stage:

```bash
python test_prompt_only.py \
  --lgm-root /kaggle/working/LGM \
  --prompt "a wooden chair, studio product photograph, centered object, white background" \
  --num-steps 20 \
  --output-dir /kaggle/working/prompt-smoke-test
```

It saves four canonical `view_000.png` … `view_270.png` images and
`prompt_grid.png`, without loading LGM's reconstruction model or computing
MRC. The output can be inspected before running the full prompt pipeline.

Use `--render-size 256` to conserve VRAM, `--no-orbit` for a faster smoke
test, or `--mrc-metric l1` only when LPIPS model weights cannot be downloaded.
Lower MRC is better.  The precise source order and all checkpoint metadata are
recorded in `metrics.json`.

Checkpoint selection, alternatives considered, and reproduction limitations are
documented in [docs/RESEARCH.md](docs/RESEARCH.md).
Direct checkpoint links and download behavior (without committing binaries) are
listed in [CHECKPOINT_DOWNLOADS.md](CHECKPOINT_DOWNLOADS.md).

## Release TODOs
- [ ] training and testing text prompt dataset
- [x] SDXL LoRA training code adapted from [diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py) and [DDPO](https://github.com/kvablack/ddpo-pytorch)
  - [ ] a multi-view diffusion model (e.g. [MVDream](https://github.com/bytedance/MVDream)), replacing Instant3D's finetuned SDXL. Pull requests are welcomed!
- [x] Multi-view Reconstruction Consistency (MRC) metric
  - [ ] an open-sourced multi-view/sparse-view reconstruction model. Pull requests are welcomed!

## Installation
```
cd ddpo
pip install -e .
```

![method](https://desaixie.github.io/carve-3d/static/images/figure_overview.png)

## Training
```
bash carve3d_train.sh
```

## Configurations
`carve3d_train.py` uses configuration files. 
The base configuration file is `config/base.py`. 
The final training config reported in the paper is in `carve3d_train()` in `config/dgx.py`.

## Improvements on DDPO (Section 4.2 of the paper)
The training code `carve3d_train.py`, configurations in `config/`, and DDPO's change to diffusers library codes `diffusers_patch/` are based on [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch).
`carve3d_train.py` is also based on [diffusers sdxl lora train example](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py).
Our improvement to DDPO include pure on-policy training, and KL-divergence regularization.

### Pure On-policy Training
In the original DDPO, total sample batch size, `sample.batch_size * sample.num_batches_per_epoch * num_nodes * 8`, is 2x the total training batch size, `train.batch_size * train.gradient_accumulation_steps * num_nodes * 8`, so that in each epoch of sampling, there will be 2 updates, each on half of the samples.

Our modification is to simply set them to be equal, so that in each epoch of sampling, there will be only 1 update, using all of the samples.
We didn't remove the importance sampling ratio, but I think removing it shouldn't make a difference.

### KL-divergence Regularization
Our KL-divergence Regularization is enabled by default with `config.kl_penalty = True`, `config.kl_in_reward = True`, `config.kl_per_prompt_stat_tracking = True`, and `config.kl_normalized_coeff = 0.2`.
The coefficient could be tuned, as a trade-off between focusing on optimizing more for higher reward or more for lower KL-divergence.


## Hyperparameter Tuning
The main hyperparameters to tune are 
+ KL-divergence regularization coefficient `config.kl_normalized_coeff`, 
+ training data size `config.train_size`, 
+ total sample batch size, `sample.batch_size * sample.num_batches_per_epoch * num_nodes * 8`, 
+ and total train batch size, `train.batch_size * train.gradient_accumulation_steps * num_nodes * 8`.

Keep total sample batch size and total train batch size equal to use our pure on-policy training.
Otherwise, set total sample batch size to be 2x (or 3x, etc.) of total train batch size to use the PPO multi-round update.
Tune the total batch size and the training data size according to Section 4.3 of the paper.

## Citation
If you find this code useful, please consider citing:
```bibtex
@inproceedings{xie2024carve3d,
  title={Carve3d: Improving multi-view reconstruction consistency for diffusion models with rl finetuning},
  author={Xie, Desai and Li, Jiahao and Tan, Hao and Sun, Xin and Shu, Zhixin and Zhou, Yi and Bi, Sai and Pirk, S{\"o}ren and Kaufman, Arie E},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6369--6379},
  year={2024}
}

@inproceedings{black2024training,
title={Training Diffusion Models with Reinforcement Learning},
author={Kevin Black and Michael Janner and Yilun Du and Ilya Kostrikov and Sergey Levine},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=YCWjhGrJFD}
}
```
