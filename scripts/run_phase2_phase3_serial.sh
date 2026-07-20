#!/usr/bin/env bash

# Resume Phase 2 and Phase 3 in strict sequence while retaining every launcher,
# server, and pair-result log. Deliberately avoids strict shell modes so a
# failure returns control to the caller or tmux pane for inspection.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
PYTHON="${LAPLACE_PYTHON:-${ROOT}/.venv/bin/python}"
LAUNCHER="${LAPLACE_ABLATION_LAUNCHER:-${ROOT}/scripts/run_multilanguage_dual_model_ablation.sh}"
SERVER_MANAGER="${LAPLACE_SERVER_MANAGER:-${ROOT}/scripts/manage_multilanguage_model_servers.sh}"
CONFIG="${LAPLACE_ABLATION_CONFIG:-${ROOT}/codex_a6000/experiments/multilanguage_dual_model_ablation_v1/experiment.json}"
OUTPUT_ROOT="${LAPLACE_ABLATION_OUTPUT_ROOT:-${ROOT}/outputs/a6000_agent_team/experiments/multilanguage_dual_model_ablation_v1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${OUTPUT_ROOT}/serial_launcher_logs"
LOG_PATH="${LOG_DIR}/phase2_phase3_${STAMP}.log"

export LAPLACE_ABLATION_BASE_REVISION="${LAPLACE_ABLATION_BASE_REVISION:-21e51e855b73971fcc6b2ea5ed8764319a5774fc}"
export LAPLACE_ABLATION_HELD_OUT_ROOT="${LAPLACE_ABLATION_HELD_OUT_ROOT:-/home/giando/work/laplace-evaluator/multilanguage_dual_model_ablation_v1_pack_v3_20260715}"
export LAPLACE_VLLM_EXECUTABLE="${LAPLACE_VLLM_EXECUTABLE:-${ROOT}/.venv-vllm-cu129/bin/vllm}"
export LAPLACE_FFMPEG_LIBRARY_PATH="${LAPLACE_FFMPEG_LIBRARY_PATH:-${ROOT}/.runtime/ffmpeg7/lib}"
export LD_LIBRARY_PATH="${LAPLACE_FFMPEG_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset LAPLACE_SERVER_OWNER_TOKEN
unset LAPLACE_SERVER_MANAGEMENT_MODE

mkdir -p "${LOG_DIR}" 2>/dev/null
exec > >(tee -a "${LOG_PATH}") 2>&1

run_command() {
  echo "SERIAL_COMMAND=$(printf '%q ' "$@")"
  "$@"
  COMMAND_STATUS="$?"
  if [ "${COMMAND_STATUS}" -ne 0 ]; then
    echo "Serial command failed with status ${COMMAND_STATUS}. Partial results and logs were retained."
  fi
  return "${COMMAND_STATUS}"
}

phase_complete() {
  run_command "${PYTHON}" -m research_workspace.multilanguage_ablation phase-status \
    --phase "$1" --require-complete --config "${CONFIG}"
}

stop_managed_server() {
  run_command "${SERVER_MANAGER}" "stop-$1"
}

run_serial_phases() {
  echo "Verifying the fingerprint-compatible Phase 1 checkpoint."
  phase_complete phase1 || return "$?"

  echo "Dry-running the selective Phase 2 resume plan. No result file is modified."
  run_command "${PYTHON}" -m research_workspace.multilanguage_ablation \
    selective-retry-plan --phase phase2 --config "${CONFIG}" || return "$?"

  echo "Running or resuming Phase 2. Compatible completed pairs will be skipped."
  run_command "${LAUNCHER}" phase2 managed
  PHASE_RESULT="$?"
  stop_managed_server phase2
  STOP_RESULT="$?"
  if [ "${PHASE_RESULT}" -ne 0 ]; then
    return "${PHASE_RESULT}"
  fi
  if [ "${STOP_RESULT}" -ne 0 ]; then
    return "${STOP_RESULT}"
  fi
  phase_complete phase2 || return "$?"

  echo "Phase 2 is complete and its server is stopped; running or resuming Phase 3."
  run_command "${LAUNCHER}" phase3 managed
  PHASE_RESULT="$?"
  stop_managed_server phase3
  STOP_RESULT="$?"
  if [ "${PHASE_RESULT}" -ne 0 ]; then
    return "${PHASE_RESULT}"
  fi
  if [ "${STOP_RESULT}" -ne 0 ]; then
    return "${STOP_RESULT}"
  fi
  phase_complete phase3 || return "$?"

  echo "Both phases are complete; finalizing the single experiment report."
  run_command "${LAUNCHER}" merge
}

STARTUP_OK=1
if [ ! -x "${PYTHON}" ]; then
  echo "Configured control-plane Python is not executable: ${PYTHON}"
  STARTUP_OK=0
fi
if [ ! -x "${LAUNCHER}" ]; then
  echo "Ablation launcher is not executable: ${LAUNCHER}"
  STARTUP_OK=0
fi
if [ ! -x "${SERVER_MANAGER}" ]; then
  echo "Server manager is not executable: ${SERVER_MANAGER}"
  STARTUP_OK=0
fi
if [ ! -x "${LAPLACE_VLLM_EXECUTABLE}" ]; then
  echo "CUDA-12-compatible vLLM is not executable: ${LAPLACE_VLLM_EXECUTABLE}"
  STARTUP_OK=0
fi
if [ ! -d "${LAPLACE_FFMPEG_LIBRARY_PATH}" ]; then
  echo "FFmpeg library directory is missing: ${LAPLACE_FFMPEG_LIBRARY_PATH}"
  STARTUP_OK=0
fi
if [ ! -f "${LAPLACE_ABLATION_HELD_OUT_ROOT}/manifest.json" ]; then
  echo "Held-out pack manifest is missing: ${LAPLACE_ABLATION_HELD_OUT_ROOT}/manifest.json"
  STARTUP_OK=0
fi

FINAL_STATUS=2
if [ "${STARTUP_OK}" -eq 1 ]; then
  cd "${ROOT}" || STARTUP_OK=0
fi
if [ "${STARTUP_OK}" -eq 1 ]; then
  echo "Laplace Phase 2 -> Phase 3 serial run: ${STAMP}"
  echo "Base revision: ${LAPLACE_ABLATION_BASE_REVISION}"
  echo "vLLM executable: ${LAPLACE_VLLM_EXECUTABLE}"
  echo "FFmpeg libraries: ${LAPLACE_FFMPEG_LIBRARY_PATH}"
  echo "Held-out pack: ${LAPLACE_ABLATION_HELD_OUT_ROOT}"
  run_serial_phases
  FINAL_STATUS="$?"
fi

echo "Serial launcher log: ${LOG_PATH}"
echo "Result, server, and launcher logs remain under ${OUTPUT_ROOT}."
echo "The shell remains available; rerun this command to resume compatible incomplete pairs."
finish_with_status() {
  return "$1"
}
finish_with_status "${FINAL_STATUS}"
