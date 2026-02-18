#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL_SEC="${INTERVAL_SEC:-120}"
HEART_LOG="${HEART_LOG:-${ROOT_DIR}/work_dirs/repro_all_logs/heartbeat.log}"
NYU_TAG="${NYU_TAG:-paper_full_nyu}"
SUN_TAG="${SUN_TAG:-paper_full_sun}"
NYU_LOG="${ROOT_DIR}/work_dirs/repro_all_logs/${NYU_TAG}.driver.log"
SUN_LOG="${ROOT_DIR}/work_dirs/repro_all_logs/${SUN_TAG}.driver.log"

mkdir -p "$(dirname "${HEART_LOG}")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

last_iter() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "n/a"
    return
  fi
  local line
  line="$(grep -oE 'Validation Iter: [0-9]+ / [0-9]+' "${file}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo "n/a"
  else
    echo "${line#Validation Iter: }"
  fi
}

while true; do
  {
    echo "[$(ts)] heartbeat"
    echo "nyu_tag=${NYU_TAG}"
    echo "sun_tag=${SUN_TAG}"
    echo "nyu_iter=$(last_iter "${NYU_LOG}")"
    echo "sun_iter=$(last_iter "${SUN_LOG}")"
    if [[ -f "${NYU_LOG}" ]]; then
      echo "nyu_log_size_bytes=$(stat -c '%s' "${NYU_LOG}")"
    fi
    if [[ -f "${SUN_LOG}" ]]; then
      echo "sun_log_size_bytes=$(stat -c '%s' "${SUN_LOG}")"
    fi
    echo "-- active reproduce processes --"
    pgrep -af "tools/reproduce_all_dformer_paper_eval.sh|tools/test.py --config configs/dformer|tools/test.py --config=configs/dformer" || true
    echo "-- gpu compute apps --"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader || true
    echo
  } >> "${HEART_LOG}"
  sleep "${INTERVAL_SEC}"
done
