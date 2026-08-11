#!/usr/bin/env bash
# Install the official LGM reconstructer and its public checkpoint for Kaggle.
# Usage: bash scripts/setup_lgm_kaggle.sh /kaggle/working/LGM
set -euo pipefail

LGM_DIR="${1:-/kaggle/working/LGM}"
LGM_REPO="https://github.com/3DTopia/LGM.git"
LGM_REVISION="fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c"
RASTER_DIR="${LGM_DIR}/diff-gaussian-rasterization"
CHECKPOINT_DIR="${LGM_DIR}/pretrained"
CHECKPOINT="${CHECKPOINT_DIR}/model_fp16_fixrot.safetensors"
CHECKPOINT_URL="https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors"
CHECKPOINT_SHA256="744d6324656342c64f871308e73db97f0eb51858d94329b30090e986a6d050ab"

if [ ! -d "${LGM_DIR}/.git" ]; then
  git clone "${LGM_REPO}" "${LGM_DIR}"
fi
git -C "${LGM_DIR}" fetch --depth 1 origin "${LGM_REVISION}"
git -C "${LGM_DIR}" checkout --detach "${LGM_REVISION}"

# Kaggle already has a CUDA-enabled PyTorch. Avoid replacing it before compiling
# the CUDA extensions against that exact build.
python -m pip install --upgrade pip
# Do not let pip replace Kaggle's CUDA PyTorch while adding xFormers; the
# Gaussian rasterizer must compile against the already-installed PyTorch ABI.
python -m pip install --upgrade --no-deps xformers || echo "xFormers install skipped; inference can still run without memory-efficient attention."
python -m pip install -r "${LGM_DIR}/requirements.txt"

if [ ! -d "${RASTER_DIR}/.git" ]; then
  git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization "${RASTER_DIR}"
fi
python -m pip install --no-build-isolation "${RASTER_DIR}"

# nvdiffrast is used only by LGM's separate `convert.py` mesh-export utility;
# LGM reconstruction, Gaussian `.ply` export, same-pose rendering, and MRC do
# not import it.  It currently has no build path for Kaggle's Python 3.12, so
# keep it opt-in instead of making the runnable pipeline fail during setup.
if [ "${INSTALL_NVDIFFRAST:-0}" = "1" ]; then
  python -m pip install git+https://github.com/NVlabs/nvdiffrast
else
  echo "Skipping optional nvdiffrast (only needed for convert.py -> .glb mesh export)."
fi

mkdir -p "${CHECKPOINT_DIR}"
if [ ! -s "${CHECKPOINT}" ]; then
  wget --show-progress -O "${CHECKPOINT}" "${CHECKPOINT_URL}"
fi
echo "${CHECKPOINT_SHA256}  ${CHECKPOINT}" | sha256sum --check --status

python - <<'PY'
import torch
assert torch.cuda.is_available(), "Enable a GPU accelerator in Kaggle before running the pipeline."
print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
PY
echo "LGM installed at ${LGM_DIR}"
echo "Checkpoint ready: ${CHECKPOINT}"
