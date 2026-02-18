#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/defaultShare/archive/yinbowen/Houjd/envs/jittordet/bin/python}"
GPU_IDS="${GPU_IDS:-0}"
MAX_ITERS="${MAX_ITERS:-0}"
STRICT_LOAD="${STRICT_LOAD:-1}"
RUN_TAG="${RUN_TAG:-full_$(date +%Y%m%d_%H%M%S)}"
MODEL_FILTER="${MODEL_FILTER:-}"
LOG_DIR="${ROOT_DIR}/work_dirs/repro_all_logs/${RUN_TAG}"
SUMMARY_CSV="${LOG_DIR}/summary.csv"
SUMMARY_MD="${LOG_DIR}/summary.md"

mkdir -p "${LOG_DIR}"

echo "run_tag,${RUN_TAG}" > "${SUMMARY_CSV}"
echo "model,dataset,config,checkpoint,target_miou,actual_miou,delta,status,log_file" >> "${SUMMARY_CSV}"

cat > "${SUMMARY_MD}" <<EOF
# Reproduce DFormer/DFormerv2 (All Models)

- run_tag: ${RUN_TAG}
- root_dir: ${ROOT_DIR}
- python: ${PYTHON_BIN}
- gpu_ids: ${GPU_IDS}
- max_iters: ${MAX_ITERS}
- strict_load: ${STRICT_LOAD}
- model_filter: ${MODEL_FILTER:-<none>}

| model | dataset | target_mIoU(%) | actual_mIoU(%) | delta | status | log |
|---|---|---:|---:|---:|---|---|
EOF

COMMON_ARGS=(
  "--mode" "val"
  "--multi_scale"
  "--flip"
  "--sliding"
  "--verbose"
)
if [[ "${STRICT_LOAD}" == "1" ]]; then
  COMMON_ARGS+=("--strict-load")
fi
if [[ "${MAX_ITERS}" -gt 0 ]]; then
  COMMON_ARGS+=("--max-iters=${MAX_ITERS}")
fi

# format: model|dataset|config|checkpoint|target_miou
MODELS=(
  "NYUv2_DFormer_Tiny|NYUDepthv2|configs/dformer/dformer_tiny_8xb8-500e_nyudepthv2-480x640.py|checkpoints/trained/NYUv2_DFormer_Tiny.pth|"
  "NYUv2_DFormer_Small|NYUDepthv2|configs/dformer/dformer_small_8xb8-500e_nyudepthv2-480x640.py|checkpoints/trained/NYUv2_DFormer_Small.pth|52.3"
  "NYUv2_DFormer_Base|NYUDepthv2|configs/dformer/dformer_base_8xb8-500e_nyudepthv2-480x640.py|checkpoints/trained/NYUv2_DFormer_Base.pth|54.1"
  "NYUv2_DFormer_Large|NYUDepthv2|configs/dformer/dformer_large_8xb8-500e_nyudepthv2-480x640.py|checkpoints/trained/NYUv2_DFormer_Large.pth|55.8"
  "NYUv2_DFormerv2_Small|NYUDepthv2|configs/dformer/dformerv2_s_8xb4-500e_nyudepthv2-480x640.py|checkpoints/trained/DFormerv2_Small_NYU.pth|53.7"
  "NYUv2_DFormerv2_Base|NYUDepthv2|configs/dformer/dformerv2_b_8xb16-500e_nyudepthv2-480x640.py|checkpoints/trained/DFormerv2_Base_NYU.pth|55.3"
  "NYUv2_DFormerv2_Large|NYUDepthv2|configs/dformer/dformerv2_l_8xb16-500e_nyudepthv2-480x640.py|checkpoints/trained/DFormerv2_Large_NYU.pth|57.1"
  "SUNRGBD_DFormer_Tiny|SUNRGBD|configs/dformer/dformer_tiny_8xb16-300e_sunrgbd-480x480.py|checkpoints/trained/SUNRGBD_DFormer_Tiny.pth|"
  "SUNRGBD_DFormer_Small|SUNRGBD|configs/dformer/dformer_small_8xb16-300e_sunrgbd-480x480.py|checkpoints/trained/SUNRGBD_DFormer_Small.pth|"
  "SUNRGBD_DFormer_Base|SUNRGBD|configs/dformer/dformer_base_8xb16-300e_sunrgbd-480x480.py|checkpoints/trained/SUNRGBD_DFormer_Base.pth|"
  "SUNRGBD_DFormer_Large|SUNRGBD|configs/dformer/dformer_large_8xb16-300e_sunrgbd-480x480.py|checkpoints/trained/SUNRGBD_DFormer_Large.pth|"
  "SUNRGBD_DFormerv2_Small|SUNRGBD|configs/dformer/dformerv2_s_8xb16-300e_sunrgbd-480x480.py|checkpoints/trained/DFormerv2_Small_SUNRGBD.pth|"
  "SUNRGBD_DFormerv2_Base|SUNRGBD|configs/dformer/dformerv2_b_8xb16-300e_sunrgbd-480x480.py|checkpoints/trained/DFormerv2_Base_SUNRGBD.pth|"
  "SUNRGBD_DFormerv2_Large|SUNRGBD|configs/dformer/dformerv2_l_8xb16-300e_sunrgbd-480x480.py|checkpoints/trained/DFormerv2_Large_SUNRGBD.pth|"
)

extract_miou_percent() {
  local log_file="$1"
  "${PYTHON_BIN}" - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, 'r', encoding='utf-8', errors='ignore').read()
vals = [float(x) for x in re.findall(r'\bmIoU:\s*([0-9]+(?:\.[0-9]+)?)', text)]
if not vals:
    print("")
    raise SystemExit(0)
v = vals[-1]
if v <= 1.5:
    v *= 100.0
print(f"{v:.4f}")
PY
}

append_summary() {
  local model="$1"
  local dataset="$2"
  local config="$3"
  local ckpt="$4"
  local target="$5"
  local actual="$6"
  local delta="$7"
  local status="$8"
  local log_file="$9"

  echo "${model},${dataset},${config},${ckpt},${target},${actual},${delta},${status},${log_file}" >> "${SUMMARY_CSV}"
  echo "| ${model} | ${dataset} | ${target:-N/A} | ${actual:-N/A} | ${delta:-N/A} | ${status} | \`${log_file}\` |" >> "${SUMMARY_MD}"
}

echo "[INFO] Running full reproduction to: ${LOG_DIR}"

idx=0
IFS=',' read -r -a gpu_arr <<< "${GPU_IDS}"
gpu_count="${#gpu_arr[@]}"

for item in "${MODELS[@]}"; do
  IFS='|' read -r model dataset config ckpt target <<< "${item}"
  log_file="${LOG_DIR}/${model}.log"

  if [[ -n "${MODEL_FILTER}" ]]; then
    if [[ ! "${model}" =~ ${MODEL_FILTER} ]]; then
      continue
    fi
  fi

  gpu="${gpu_arr[$((idx % gpu_count))]}"
  idx=$((idx + 1))

  echo "[INFO] =================================================="
  echo "[INFO] model=${model} dataset=${dataset} gpu=${gpu}"
  echo "[INFO] config=${config}"
  echo "[INFO] checkpoint=${ckpt}"
  echo "[INFO] log=${log_file}"

  if [[ ! -f "${ROOT_DIR}/${ckpt}" ]]; then
    echo "[ERROR] checkpoint not found: ${ROOT_DIR}/${ckpt}" | tee "${log_file}"
    append_summary "${model}" "${dataset}" "${config}" "${ckpt}" "${target}" "" "" "missing_checkpoint" "${log_file}"
    continue
  fi
  if [[ ! -s "${ROOT_DIR}/${ckpt}" ]]; then
    echo "[ERROR] checkpoint is empty: ${ROOT_DIR}/${ckpt}" | tee "${log_file}"
    append_summary "${model}" "${dataset}" "${config}" "${ckpt}" "${target}" "" "" "empty_checkpoint" "${log_file}"
    continue
  fi

  set +e
  (
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1
    "${PYTHON_BIN}" tools/test.py \
      --config "${config}" \
      --checkpoint "${ckpt}" \
      "${COMMON_ARGS[@]}"
  ) 2>&1 | tee "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  set -e

  actual="$(extract_miou_percent "${log_file}")"
  status="ok"
  delta=""
  if [[ "${exit_code}" -ne 0 ]]; then
    status="failed_exit_${exit_code}"
  elif [[ -z "${actual}" ]]; then
    status="failed_no_miou"
  elif [[ -n "${target}" ]]; then
    delta="$("${PYTHON_BIN}" - "${target}" "${actual}" <<'PY'
import sys
t = float(sys.argv[1])
a = float(sys.argv[2])
print(f"{a - t:+.4f}")
PY
)"
  fi

  append_summary "${model}" "${dataset}" "${config}" "${ckpt}" "${target}" "${actual}" "${delta}" "${status}" "${log_file}"
done

echo "[INFO] Done."
echo "[INFO] Summary CSV: ${SUMMARY_CSV}"
echo "[INFO] Summary MD : ${SUMMARY_MD}"
