#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DFORMER_ROOT="${DFORMER_ROOT:-/defaultShare/archive/yinbowen/Houjd/DFormer-Jittor}"
PYTHON_BIN="${PYTHON_BIN:-/defaultShare/archive/yinbowen/Houjd/envs/jittordet/bin/python}"
GPUS="${GPUS:-1}"
STRICT_LOAD="${STRICT_LOAD:-1}"
MS_SCALES="${MS_SCALES:-}"
MAX_ITERS="${MAX_ITERS:-0}"

echo "[INFO] nk-mmseg root: ${ROOT_DIR}"
echo "[INFO] DFormer-Jittor root: ${DFORMER_ROOT}"
echo "[INFO] Python bin: ${PYTHON_BIN}"
echo "[INFO] STRICT_LOAD: ${STRICT_LOAD}"
if [ -n "${MS_SCALES}" ]; then
  echo "[INFO] MS_SCALES: ${MS_SCALES}"
else
  echo "[INFO] MS_SCALES: <default in eval.py>"
fi
echo "[INFO] MAX_ITERS: ${MAX_ITERS}"

if [ ! -d "${DFORMER_ROOT}" ]; then
  echo "[ERROR] DFormer-Jittor not found: ${DFORMER_ROOT}"
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[WARN] ${PYTHON_BIN} not executable, fallback to python"
  PYTHON_BIN="python"
fi

cd "${ROOT_DIR}"

# # Link dataset/checkpoints from DFormer-Jittor for reproducible evaluation.
# if [ ! -e "${ROOT_DIR}/datasets" ]; then
#   ln -s "${DFORMER_ROOT}/datasets" "${ROOT_DIR}/datasets"
#   echo "[INFO] linked datasets -> ${DFORMER_ROOT}/datasets"
# fi

if [ ! -e "${ROOT_DIR}/checkpoints" ]; then
  ln -s "${DFORMER_ROOT}/checkpoints" "${ROOT_DIR}/checkpoints"
  echo "[INFO] linked checkpoints -> ${DFORMER_ROOT}/checkpoints"
fi

echo "[INFO] Evaluate NYUDepthv2 DFormer-Large (target mIoU ~= 55.8)"
STRICT_ARGS=()
if [ "${STRICT_LOAD}" = "1" ]; then
  STRICT_ARGS+=(--strict-load)
fi
SCALE_ARGS=()
if [ -n "${MS_SCALES}" ]; then
  # shellcheck disable=SC2206
  _scales=(${MS_SCALES})
  SCALE_ARGS+=(--scales "${_scales[@]}")
fi
ITER_ARGS=()
if [ "${MAX_ITERS}" -gt 0 ]; then
  ITER_ARGS+=(--max-iters="${MAX_ITERS}")
fi

${PYTHON_BIN} tools/test.py \
  --config=configs/dformer/dformer_large_8xb8-500e_nyudepthv2-480x640.py \
  --gpus="${GPUS}" \
  --checkpoint="${ROOT_DIR}/checkpoints/trained/NYUv2_DFormer_Large.pth" \
  --mode=val \
  "${STRICT_ARGS[@]}" \
  --multi_scale \
  "${SCALE_ARGS[@]}" \
  "${ITER_ARGS[@]}" \
  --flip \
  --sliding \
  --verbose

echo "[INFO] Evaluate NYUDepthv2 DFormerv2-Large (target mIoU ~= 57.1)"
${PYTHON_BIN} tools/test.py \
  --config=configs/dformer/dformerv2_l_8xb16-500e_nyudepthv2-480x640.py \
  --gpus="${GPUS}" \
  --checkpoint="${ROOT_DIR}/checkpoints/trained/DFormerv2_Large_NYU.pth" \
  --mode=val \
  "${STRICT_ARGS[@]}" \
  --multi_scale \
  "${SCALE_ARGS[@]}" \
  "${ITER_ARGS[@]}" \
  --flip \
  --sliding \
  --verbose
