#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/defaultShare/archive/yinbowen/Houjd/envs/jittordet/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.4}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

NYU_TAG="${NYU_TAG:-paper_full_nyu}"
SUN_TAG="${SUN_TAG:-paper_full_sun}"
CHECK_SEC="${CHECK_SEC:-120}"
MAX_ITERS="${MAX_ITERS:-0}"
STRICT_LOAD="${STRICT_LOAD:-1}"
NYU_EXPECTED_COUNT="${NYU_EXPECTED_COUNT:-7}"
SUN_EXPECTED_COUNT="${SUN_EXPECTED_COUNT:-7}"

NYU_DRIVER_LOG="${ROOT_DIR}/work_dirs/repro_all_logs/${NYU_TAG}.driver.log"
SUN_DRIVER_LOG="${ROOT_DIR}/work_dirs/repro_all_logs/${SUN_TAG}.driver.log"
WATCH_LOG="${ROOT_DIR}/work_dirs/repro_all_logs/watchdog.log"

mkdir -p "$(dirname "${WATCH_LOG}")"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

is_done() {
  local tag="$1"
  local expected_count="$2"
  local summary_csv="${ROOT_DIR}/work_dirs/repro_all_logs/${tag}/summary.csv"
  [[ -f "${summary_csv}" ]] || return 1
  local total_rows done_rows
  total_rows=$(awk -F',' 'NR>2 {c++} END{print c+0}' "${summary_csv}")
  done_rows=$(awk -F',' 'NR>2 && $8 ~ /^ok$/ {c++} END{print c+0}' "${summary_csv}")
  [[ "${total_rows}" -ge "${expected_count}" && "${done_rows}" -ge "${expected_count}" ]]
}

spawn_job() {
  local tag="$1"
  local filter="$2"
  local gpu="$3"
  local driver_log="$4"
  (
    cd "${ROOT_DIR}"
    RUN_TAG="${tag}" \
    MODEL_FILTER="${filter}" \
    GPU_IDS="${gpu}" \
    MAX_ITERS="${MAX_ITERS}" \
    STRICT_LOAD="${STRICT_LOAD}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    /bin/bash tools/reproduce_all_dformer_paper_eval.sh
  ) >> "${driver_log}" 2>&1 &
  echo $!
}

ensure_job() {
  local tag="$1"
  local filter="$2"
  local gpu="$3"
  local driver_log="$4"
  local expected_count="$5"
  local pattern="RUN_TAG=${tag} MODEL_FILTER=${filter} GPU_IDS=${gpu}"

  if is_done "${tag}" "${expected_count}"; then
    echo "[$(timestamp)] ${tag} already completed." | tee -a "${WATCH_LOG}"
    return 0
  fi

  if pgrep -af "${pattern}" >/dev/null 2>&1; then
    return 0
  fi

  local pid
  pid=$(spawn_job "${tag}" "${filter}" "${gpu}" "${driver_log}")
  echo "[$(timestamp)] restarted ${tag} on GPU ${gpu}, pid=${pid}" | tee -a "${WATCH_LOG}"
}

echo "[$(timestamp)] watchdog started: NYU=${NYU_TAG}, SUN=${SUN_TAG}" | tee -a "${WATCH_LOG}"

while true; do
  ensure_job "${NYU_TAG}" "NYUv2" "0" "${NYU_DRIVER_LOG}" "${NYU_EXPECTED_COUNT}"
  ensure_job "${SUN_TAG}" "SUNRGBD" "1" "${SUN_DRIVER_LOG}" "${SUN_EXPECTED_COUNT}"

  nyu_done=0
  sun_done=0
  is_done "${NYU_TAG}" "${NYU_EXPECTED_COUNT}" && nyu_done=1 || true
  is_done "${SUN_TAG}" "${SUN_EXPECTED_COUNT}" && sun_done=1 || true
  echo "[$(timestamp)] heartbeat: nyu_done=${nyu_done}, sun_done=${sun_done}" | tee -a "${WATCH_LOG}"

  if [[ "${nyu_done}" -eq 1 && "${sun_done}" -eq 1 ]]; then
    echo "[$(timestamp)] all completed." | tee -a "${WATCH_LOG}"
    break
  fi
  sleep "${CHECK_SEC}"
done
