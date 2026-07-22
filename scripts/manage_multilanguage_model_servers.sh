#!/usr/bin/env bash

# Model-server lifecycle is deliberately external to the experiment runner.
# This script never uses strict shell modes and always returns control to the
# invoking shell or tmux pane after reporting an error.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
STATE_DIR="${ROOT}/outputs/a6000_agent_team/model_servers"
PYTHON="${ROOT}/.venv/bin/python"
OWNER_TOKEN="${LAPLACE_SERVER_OWNER_TOKEN:-}"
VLLM_OVERRIDE="${LAPLACE_VLLM_EXECUTABLE:-}"
FFMPEG_LIBRARY_PATH="${LAPLACE_FFMPEG_LIBRARY_PATH:-${ROOT}/.runtime/ffmpeg7/lib}"

if [ -d "${FFMPEG_LIBRARY_PATH}" ]; then
  export LD_LIBRARY_PATH="${FFMPEG_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

mkdir -p "${STATE_DIR}" 2>/dev/null

profile_values() {
  if [ ! -x "${PYTHON}" ]; then
    echo "Missing Laplace control-plane Python: ${PYTHON}"
    return 2
  fi
  mapfile -t PROFILE_DATA < <(
    "${PYTHON}" "${ROOT}/scripts/manage_multilanguage_models.py" server-profile --artifact "$1"
  )
  if [ "${#PROFILE_DATA[@]}" -lt 8 ]; then
    echo "Invalid or unknown server profile: $1"
    return 2
  fi
  VLLM="${PROFILE_DATA[0]}"
  if [ -n "${VLLM_OVERRIDE}" ]; then
    case "${VLLM_OVERRIDE}" in
      /*) VLLM="${VLLM_OVERRIDE}" ;;
      *)
        echo "LAPLACE_VLLM_EXECUTABLE must be an absolute path: ${VLLM_OVERRIDE}"
        return 2
        ;;
    esac
  fi
  VLLM_BIN_DIR="$(dirname "${VLLM}")"
  case ":${PATH}:" in
    *":${VLLM_BIN_DIR}:"*) ;;
    *) export PATH="${VLLM_BIN_DIR}:${PATH}" ;;
  esac
  MODEL_PATH="${PROFILE_DATA[1]}"
  SERVED_MODEL="${PROFILE_DATA[2]}"
  PORT="${PROFILE_DATA[3]}"
  GPU_FRACTION="${PROFILE_DATA[4]}"
  MAX_MODEL_LEN="${PROFILE_DATA[5]}"
  MAX_SEQS="${PROFILE_DATA[6]}"
  EXTRA_ARGS=("${PROFILE_DATA[@]:7}")
  PID_FILE="${STATE_DIR}/${1}.pid"
  OWNER_FILE="${PID_FILE}.owner"
  LOG_FILE="${STATE_DIR}/${1}_$(date -u +%Y%m%dT%H%M%SZ).log"
}

pid_is_owned() {
  profile_values "$1" || return 2
  if [ ! -f "${PID_FILE}" ]; then
    return 1
  fi
  PID="$(sed -n '1p' "${PID_FILE}" 2>/dev/null)"
  case "${PID}" in
    ''|*[!0-9]*) return 1 ;;
  esac
  if [ ! -r "/proc/${PID}/cmdline" ]; then
    return 1
  fi
  CMDLINE="$(tr '\0' ' ' < "/proc/${PID}/cmdline" 2>/dev/null)"
  case "${CMDLINE}" in
    *"${MODEL_PATH}"*"--host 127.0.0.1"*"--port ${PORT}"*) return 0 ;;
    *) return 1 ;;
  esac
}

start_profile() {
  PROFILE="$1"
  profile_values "${PROFILE}" || return 2
  if pid_is_owned "${PROFILE}"; then
    echo "${PROFILE} is already running with PID ${PID}."
    return 0
  fi
  if [ ! -x "${VLLM}" ]; then
    echo "Missing vLLM executable: ${VLLM}"
    echo "Prepare the pinned serving environment before retrying."
    return 2
  fi
  READY_OUTPUT="$(
    "${PYTHON}" "${ROOT}/scripts/manage_multilanguage_models.py" ready --artifact "${PROFILE}" 2>&1
  )"
  if [ "$?" -ne 0 ]; then
    echo "Model artifact is not hash- and metadata-verified: ${MODEL_PATH}"
    echo "${READY_OUTPUT}"
    echo "Preparation commands: ${PYTHON} ${ROOT}/scripts/manage_multilanguage_models.py commands"
    return 2
  fi
  echo "Starting ${PROFILE}; log: ${LOG_FILE}"
  nohup "${VLLM}" serve "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_SEQS}" \
    --gpu-memory-utilization "${GPU_FRACTION}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    "${EXTRA_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
  PID=$!
  printf '%s\n' "${PID}" >"${PID_FILE}"
  sleep 1
  if pid_is_owned "${PROFILE}"; then
    if [ -n "${OWNER_TOKEN}" ]; then
      printf '%s\n%s\n' "${OWNER_TOKEN}" "${PID}" >"${OWNER_FILE}"
      chmod 600 "${OWNER_FILE}" 2>/dev/null
    fi
    echo "${PROFILE} started with PID ${PID}."
  else
    echo "${PROFILE} did not remain running; inspect ${LOG_FILE}."
    return 2
  fi
  return 0
}

stop_profile() {
  PROFILE="$1"
  profile_values "${PROFILE}" || return 2
  if ! pid_is_owned "${PROFILE}"; then
    echo "No owned ${PROFILE} server is running; no process was signalled."
    return 0
  fi
  if [ -n "${OWNER_TOKEN}" ]; then
    RECORDED_TOKEN="$(sed -n '1p' "${OWNER_FILE}" 2>/dev/null)"
    RECORDED_PID="$(sed -n '2p' "${OWNER_FILE}" 2>/dev/null)"
    if [ "${RECORDED_TOKEN}" != "${OWNER_TOKEN}" ] || [ "${RECORDED_PID}" != "${PID}" ]; then
      echo "${PROFILE} was not started by orchestration token ${OWNER_TOKEN}; no process was signalled."
      return 2
    fi
  fi
  echo "Stopping owned ${PROFILE} PID ${PID}."
  kill -TERM "${PID}" 2>/dev/null
  COUNT=0
  while [ -r "/proc/${PID}/cmdline" ] && [ "${COUNT}" -lt 30 ]; do
    sleep 1
    COUNT=$((COUNT + 1))
  done
  if [ -r "/proc/${PID}/cmdline" ]; then
    echo "PID ${PID} did not stop after 30 seconds; it was not force-killed."
    return 2
  else
    rm -f "${PID_FILE}" "${OWNER_FILE}"
    echo "${PROFILE} stopped."
  fi
  return 0
}

status_profile() {
  PROFILE="$1"
  if pid_is_owned "${PROFILE}"; then
    echo "${PROFILE}: RUNNING pid=${PID} port=${PORT} model=${SERVED_MODEL}"
  else
    echo "${PROFILE}: NOT_RUNNING"
  fi
  return 0
}

require_profile_running() {
  PROFILE="$1"
  if pid_is_owned "${PROFILE}"; then
    echo "${PROFILE}: RUNNING pid=${PID} port=${PORT} model=${SERVED_MODEL}"
    return 0
  fi
  echo "${PROFILE}: NOT_RUNNING"
  return 2
}

command_profile() {
  PROFILE="$1"
  profile_values "${PROFILE}" || return 2
  printf 'profile=%s\nexecutable=%s\npid_file=%s\nlog_file=%s\ncommand=' \
    "${PROFILE}" "${VLLM}" "${PID_FILE}" "${LOG_FILE}"
  printf '%q ' "${VLLM}" serve "${MODEL_PATH}" \
    --host 127.0.0.1 --port "${PORT}" --served-model-name "${SERVED_MODEL}" \
    --tensor-parallel-size 1 --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_SEQS}" --gpu-memory-utilization "${GPU_FRACTION}" \
    --enable-prefix-caching --enable-chunked-prefill "${EXTRA_ARGS[@]}"
  printf '\n'
  return 0
}

start_phase3_profiles() {
  # vLLM profiles non-KV memory independently for each process. On a shared
  # GPU, the smaller worker must reserve its KV cache before the main model;
  # otherwise the worker can observe the already-resident main allocation as
  # its own profiling overhead and reject a configuration that fits jointly.
  if pid_is_owned phase2_main; then
    echo "Restarting owned phase2_main after the RTL worker for deterministic Phase 3 memory profiling."
    stop_profile phase2_main || return 2
  fi
  start_profile phase2_rtl_worker || return 2
  start_profile phase2_main
  PHASE3_MAIN_STATUS="$?"
  if [ "${PHASE3_MAIN_STATUS}" -ne 0 ]; then
    stop_profile phase2_rtl_worker
    return "${PHASE3_MAIN_STATUS}"
  fi
  return 0
}

ACTION="${1:-help}"
ACTION_STATUS=2
case "${ACTION}" in
  start-phase1) start_profile phase1_main; ACTION_STATUS="$?" ;;
  start-phase2-main) start_profile phase2_main; ACTION_STATUS="$?" ;;
  start-phase2) start_profile phase2_main; ACTION_STATUS="$?" ;;
  start-phase2-worker) start_profile phase2_rtl_worker; ACTION_STATUS="$?" ;;
  start-phase3-main) start_profile phase2_main; ACTION_STATUS="$?" ;;
  start-phase3-worker) start_profile phase2_rtl_worker; ACTION_STATUS="$?" ;;
  start-phase3) start_phase3_profiles; ACTION_STATUS="$?" ;;
  command-phase1) command_profile phase1_main; ACTION_STATUS="$?" ;;
  command-phase2-main) command_profile phase2_main; ACTION_STATUS="$?" ;;
  command-phase2-worker) command_profile phase2_rtl_worker; ACTION_STATUS="$?" ;;
  command-phase3-main) command_profile phase2_main; ACTION_STATUS="$?" ;;
  command-phase3-worker) command_profile phase2_rtl_worker; ACTION_STATUS="$?" ;;
  check-phase1) require_profile_running phase1_main; ACTION_STATUS="$?" ;;
  check-phase2) require_profile_running phase2_main; ACTION_STATUS="$?" ;;
  check-phase3)
    require_profile_running phase2_main
    ACTION_STATUS="$?"
    if [ "${ACTION_STATUS}" -eq 0 ]; then
      require_profile_running phase2_rtl_worker
      ACTION_STATUS="$?"
    fi
    ;;
  stop-phase1) stop_profile phase1_main; ACTION_STATUS="$?" ;;
  stop-phase2-main) stop_profile phase2_main; ACTION_STATUS="$?" ;;
  stop-phase2) stop_profile phase2_main; ACTION_STATUS="$?" ;;
  stop-phase2-worker) stop_profile phase2_rtl_worker; ACTION_STATUS="$?" ;;
  stop-phase3-main) stop_profile phase2_main; ACTION_STATUS="$?" ;;
  stop-phase3-worker) stop_profile phase2_rtl_worker; ACTION_STATUS="$?" ;;
  stop-phase3)
    stop_profile phase2_rtl_worker
    WORKER_STOP_STATUS="$?"
    stop_profile phase2_main
    MAIN_STOP_STATUS="$?"
    ACTION_STATUS="${WORKER_STOP_STATUS}"
    if [ "${ACTION_STATUS}" -eq 0 ]; then
      ACTION_STATUS="${MAIN_STOP_STATUS}"
    fi
    ;;
  status)
    status_profile phase1_main
    status_profile phase2_main
    status_profile phase2_rtl_worker
    ACTION_STATUS=0
    ;;
  health-phase1)
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/manage_multilanguage_models.py" endpoint --artifact phase1_main
    ACTION_STATUS="$?"
    ;;
  health-phase2)
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/manage_multilanguage_models.py" endpoint --artifact phase2_main
    ACTION_STATUS="$?"
    ;;
  health-phase3)
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/manage_multilanguage_models.py" endpoint --artifact phase2_main
    ACTION_STATUS="$?"
    if [ "${ACTION_STATUS}" -eq 0 ]; then
      "${ROOT}/.venv/bin/python" "${ROOT}/scripts/manage_multilanguage_models.py" endpoint --artifact phase2_rtl_worker
      ACTION_STATUS="$?"
    fi
    ;;
  gpu)
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/manage_multilanguage_models.py" gpu
    ACTION_STATUS="$?"
    ;;
  *)
    echo "Usage: $0 {start-phase1|start-phase2|start-phase3|start-phase3-worker|command-phase1|command-phase2-main|command-phase3-main|command-phase3-worker|check-phase1|check-phase2|check-phase3|stop-phase1|stop-phase2|stop-phase3|status|health-phase1|health-phase2|health-phase3|gpu}"
    ;;
esac

# Propagate lifecycle failures to orchestration without using strict shell modes.
# When invoked from an interactive shell or a tmux pane, returning from this
# script leaves that caller available for inspection.
finish_with_status() {
  return "$1"
}
finish_with_status "${ACTION_STATUS}"
