#!/usr/bin/env bash

# Three-phase launcher for the one logical multilingual ablation. It avoids
# strict shell modes and explicit exits so failures remain visible in tmux.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
ACTION="${1:-}"
MODE="${2:-}"
PYTHON="${LAPLACE_PYTHON:-${ROOT}/.venv/bin/python}"
CONFIG="${LAPLACE_ABLATION_CONFIG:-${ROOT}/codex_a6000/experiments/multilanguage_dual_model_ablation_v1/experiment.json}"
SERVER_MANAGER="${LAPLACE_SERVER_MANAGER:-${ROOT}/scripts/manage_multilanguage_model_servers.sh}"
OUTPUT_ROOT="${LAPLACE_ABLATION_OUTPUT_ROOT:-${ROOT}/outputs/a6000_agent_team/experiments/multilanguage_dual_model_ablation_v1}"
CORPUS_ROOT="${OUTPUT_ROOT}/corpus"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${OUTPUT_ROOT}/launcher_logs"
LOG_PATH="${LOG_DIR}/${STAMP}.log"
ORCHESTRATION_DIR="${OUTPUT_ROOT}/orchestration"
TOKEN_FILE="${ORCHESTRATION_DIR}/managed_owner_token"
LIFECYCLE_LOG="${ORCHESTRATION_DIR}/server_lifecycle_${STAMP}.log"
SERVER_READY_TIMEOUT_SECONDS="${LAPLACE_SERVER_READY_TIMEOUT_SECONDS:-1800}"
SERVER_READY_POLL_SECONDS="${LAPLACE_SERVER_READY_POLL_SECONDS:-10}"

if [ -d "${ROOT}/.tools/multilanguage/bin" ]; then
  export PATH="${ROOT}/.tools/multilanguage/bin:${PATH}"
fi
mkdir -p "${LOG_DIR}" "${ORCHESTRATION_DIR}" 2>/dev/null
exec > >(tee -a "${LOG_PATH}") 2>&1

log_lifecycle() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LIFECYCLE_LOG}"
}

run_control() {
  echo "CONTROL_COMMAND=$(printf '%q ' "$@")"
  "$@"
  RESULT="$?"
  if [ "${RESULT}" -ne 0 ]; then
    echo "Control command failed with status ${RESULT}; the next phase will not start."
  fi
  return "${RESULT}"
}

phase_complete() {
  PHASE="$1"
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation phase-status \
    --phase "${PHASE}" --require-complete --config "${CONFIG}"
}

validate_phase_offline() {
  PHASE="$1"
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
    "validate-${PHASE}" --config "${CONFIG}" || return "$?"
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
    validate-manifest --config "${CONFIG}" || return "$?"
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
    validate-corpus --config "${CONFIG}" --corpus-overlay "${CORPUS_ROOT}" || return "$?"
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
    validate-heldout --config "${CONFIG}" || return "$?"
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
    plan-only --phase "${PHASE}" --config "${CONFIG}"
}

wait_for_managed_endpoint() {
  PHASE="$1"
  WAIT_STARTED="$(date +%s)"
  while true; do
    LAPLACE_SERVER_OWNER_TOKEN="${OWNER_TOKEN}" "${SERVER_MANAGER}" "check-${PHASE}"
    SERVER_RESULT="$?"
    if [ "${SERVER_RESULT}" -ne 0 ]; then
      echo "${PHASE} server stopped before its endpoint became ready."
      log_lifecycle "phase=${PHASE} server_running=false"
      return 2
    fi
    echo "CONTROL_COMMAND=$(printf '%q ' "${PYTHON}" -m research_workspace.multilanguage_ablation validate-runtime --phase "${PHASE}" --config "${CONFIG}")"
    "${PYTHON}" -m research_workspace.multilanguage_ablation \
      validate-runtime --phase "${PHASE}" --config "${CONFIG}"
    RESULT="$?"
    if [ "${RESULT}" -eq 0 ]; then
      echo "CONTROL_COMMAND=$(printf '%q ' "${PYTHON}" -m research_workspace.multilanguage_ablation smoke-runtime --phase "${PHASE}" --config "${CONFIG}")"
      "${PYTHON}" -m research_workspace.multilanguage_ablation \
        smoke-runtime --phase "${PHASE}" --config "${CONFIG}"
      RESULT="$?"
      if [ "${RESULT}" -eq 0 ]; then
        log_lifecycle "phase=${PHASE} endpoint_ready=true smoke_passed=true"
        return 0
      fi
      log_lifecycle "phase=${PHASE} endpoint_ready=true smoke_passed=false"
    fi
    WAIT_NOW="$(date +%s)"
    WAIT_ELAPSED=$((WAIT_NOW - WAIT_STARTED))
    if [ "${WAIT_ELAPSED}" -ge "${SERVER_READY_TIMEOUT_SECONDS}" ]; then
      echo "${PHASE} endpoints were not ready within ${SERVER_READY_TIMEOUT_SECONDS} seconds."
      log_lifecycle "phase=${PHASE} endpoint_ready=false elapsed_seconds=${WAIT_ELAPSED}"
      return 2
    fi
    echo "${PHASE} endpoint is not ready after ${WAIT_ELAPSED}s; retrying in ${SERVER_READY_POLL_SECONDS}s."
    sleep "${SERVER_READY_POLL_SECONDS}"
  done
}

run_phase_after_runtime_validation() {
  PHASE="$1"
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
    "run-${PHASE}" --config "${CONFIG}" || return "$?"
  phase_complete "${PHASE}"
}

run_external_phase_with_endpoint() {
  PHASE="$1"
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
    validate-runtime --phase "${PHASE}" --config "${CONFIG}" || return "$?"
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
    smoke-runtime --phase "${PHASE}" --config "${CONFIG}" || return "$?"
  run_phase_after_runtime_validation "${PHASE}"
}

load_owner_token() {
  if [ -s "${TOKEN_FILE}" ]; then
    OWNER_TOKEN="$(sed -n '1p' "${TOKEN_FILE}" 2>/dev/null)"
  else
    OWNER_TOKEN="laplace-${STAMP}-$$"
    printf '%s\n' "${OWNER_TOKEN}" >"${TOKEN_FILE}"
    chmod 600 "${TOKEN_FILE}" 2>/dev/null
  fi
  export LAPLACE_SERVER_OWNER_TOKEN="${OWNER_TOKEN}"
  export LAPLACE_SERVER_MANAGEMENT_MODE="managed"
  log_lifecycle "owner_token_loaded token=${OWNER_TOKEN}"
}

manage_server() {
  SERVER_ACTION="$1"
  log_lifecycle "server_action=${SERVER_ACTION}"
  LAPLACE_SERVER_OWNER_TOKEN="${OWNER_TOKEN}" "${SERVER_MANAGER}" "${SERVER_ACTION}"
  return "$?"
}

cleanup_owner_token() {
  rm -f "${TOKEN_FILE}" 2>/dev/null
  log_lifecycle "managed_flow_cleanup_complete"
}

run_external_phase() {
  PHASE="$1"
  export LAPLACE_SERVER_MANAGEMENT_MODE="external"
  unset LAPLACE_SERVER_OWNER_TOKEN
  if phase_complete "${PHASE}"; then
    echo "${PHASE} is already complete and fingerprint-compatible; no task-arm pair was repeated."
    return 0
  fi
  validate_phase_offline "${PHASE}" || return "$?"
  run_external_phase_with_endpoint "${PHASE}"
}

run_managed_phase() {
  PHASE="$1"
  load_owner_token
  if phase_complete "${PHASE}"; then
    echo "${PHASE} is already complete and fingerprint-compatible; no server was started."
    cleanup_owner_token
    return 0
  fi
  validate_phase_offline "${PHASE}" || return "$?"
  manage_server "start-${PHASE}"
  RESULT="$?"
  if [ "${RESULT}" -eq 0 ]; then
    wait_for_managed_endpoint "${PHASE}"
    RESULT="$?"
  fi
  if [ "${RESULT}" -eq 0 ]; then
    run_phase_after_runtime_validation "${PHASE}"
    RESULT="$?"
  fi
  manage_server "stop-${PHASE}"
  STOP_RESULT="$?"
  if [ "${RESULT}" -eq 0 ]; then
    RESULT="${STOP_RESULT}"
  fi
  if [ "${RESULT}" -eq 0 ]; then
    cleanup_owner_token
  fi
  return "${RESULT}"
}

run_all_managed() {
  load_owner_token

  if ! phase_complete phase1; then
    validate_phase_offline phase1 || return "$?"
    manage_server start-phase1
    RESULT="$?"
    if [ "${RESULT}" -eq 0 ]; then
      wait_for_managed_endpoint phase1
      RESULT="$?"
    fi
    if [ "${RESULT}" -eq 0 ]; then
      run_phase_after_runtime_validation phase1
      RESULT="$?"
    fi
    manage_server stop-phase1
    STOP_RESULT="$?"
    if [ "${RESULT}" -eq 0 ]; then
      RESULT="${STOP_RESULT}"
    fi
    if [ "${RESULT}" -ne 0 ]; then
      log_lifecycle "phase1_failed next_phase_blocked=true"
      return "${RESULT}"
    fi
  else
    echo "Phase 1 is already complete; skipping all Arm-A task pairs."
    manage_server stop-phase1
  fi

  phase_complete phase1 || return "$?"
  if ! phase_complete phase2; then
    validate_phase_offline phase2 || return "$?"
    manage_server start-phase2
    RESULT="$?"
    if [ "${RESULT}" -eq 0 ]; then
      wait_for_managed_endpoint phase2
      RESULT="$?"
    fi
    if [ "${RESULT}" -eq 0 ]; then
      run_phase_after_runtime_validation phase2
      RESULT="$?"
    fi
    if [ "${RESULT}" -ne 0 ]; then
      manage_server stop-phase2
      log_lifecycle "phase2_failed next_phase_blocked=true"
      return "${RESULT}"
    fi
  else
    echo "Phase 2 is already complete; skipping all Arm-B task pairs."
  fi

  phase_complete phase2 || return "$?"
  if ! phase_complete phase3; then
    validate_phase_offline phase3
    RESULT="$?"
    if [ "${RESULT}" -ne 0 ]; then
      manage_server stop-phase2
      return "${RESULT}"
    fi
    # start-phase3 initializes CodeV before Qwen3.6 so vLLM profiles the shared
    # GPU memory deterministically; an owned Phase-2 Qwen process is restarted.
    manage_server start-phase3
    RESULT="$?"
    if [ "${RESULT}" -eq 0 ]; then
      wait_for_managed_endpoint phase3
      RESULT="$?"
    fi
    if [ "${RESULT}" -eq 0 ]; then
      run_phase_after_runtime_validation phase3
      RESULT="$?"
    fi
    manage_server stop-phase3
    STOP_RESULT="$?"
    if [ "${RESULT}" -eq 0 ]; then
      RESULT="${STOP_RESULT}"
    fi
    if [ "${RESULT}" -ne 0 ]; then
      log_lifecycle "phase3_failed merge_blocked=true"
      return "${RESULT}"
    fi
  else
    echo "Phase 3 is already complete; skipping all Arm-C task pairs."
    manage_server stop-phase3
  fi

  phase_complete phase3 || return "$?"
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
    merge-report --config "${CONFIG}" || return "$?"
  cleanup_owner_token
  return 0
}

run_all_external() {
  export LAPLACE_SERVER_MANAGEMENT_MODE="external"
  unset LAPLACE_SERVER_OWNER_TOKEN
  echo "External mode never starts, switches or stops model servers."
  echo "Ensure the correct endpoint is active before each phase, or run the individual phase commands."
  for PHASE in phase1 phase2 phase3; do
    run_external_phase "${PHASE}" || return "$?"
  done
  run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
    merge-report --config "${CONFIG}"
}

STARTUP_OK=1
case "${ACTION}" in
  phase1|phase2|phase3|all|status|merge) ;;
  *)
    echo "Usage: $0 {phase1|phase2|phase3|all|status|merge} [external|managed]"
    STARTUP_OK=0
    ;;
esac
if [ ! -x "${PYTHON}" ]; then
  echo "Configured Laplace Python is not executable: ${PYTHON}"
  STARTUP_OK=0
fi
if [ -z "${LAPLACE_ABLATION_BASE_REVISION:-}" ]; then
  echo "LAPLACE_ABLATION_BASE_REVISION must name the reviewed clean checkpoint commit."
  STARTUP_OK=0
fi
if [ "${ACTION}" != "status" ] && [ "${ACTION}" != "merge" ] \
  && [ -z "${LAPLACE_ABLATION_HELD_OUT_ROOT:-}" ]; then
  echo "LAPLACE_ABLATION_HELD_OUT_ROOT must name the evaluator-owned pack outside the repository."
  STARTUP_OK=0
fi

FINAL_STATUS=2
if [ "${STARTUP_OK}" -eq 1 ]; then
  cd "${ROOT}" || STARTUP_OK=0
fi

if [ "${STARTUP_OK}" -eq 1 ]; then
  echo "Laplace three-phase ablation launcher: ${STAMP}"
  echo "Configuration: ${CONFIG}"
  if [ "${ACTION}" == "status" ]; then
    run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
      phase-status --config "${CONFIG}"
    FINAL_STATUS="$?"
  elif [ "${ACTION}" == "merge" ]; then
    run_control "${PYTHON}" -m research_workspace.multilanguage_ablation \
      merge-report --config "${CONFIG}"
    FINAL_STATUS="$?"
  elif [ "${ACTION}" == "all" ]; then
    SELECTED_MODE="${MODE:-managed}"
    if [ "${SELECTED_MODE}" == "managed" ]; then
      run_all_managed
      FINAL_STATUS="$?"
    elif [ "${SELECTED_MODE}" == "external" ]; then
      run_all_external
      FINAL_STATUS="$?"
    else
      echo "Mode must be external or managed."
      FINAL_STATUS=2
    fi
  else
    SELECTED_MODE="${MODE:-external}"
    if [ "${SELECTED_MODE}" == "managed" ]; then
      run_managed_phase "${ACTION}"
      FINAL_STATUS="$?"
    elif [ "${SELECTED_MODE}" == "external" ]; then
      run_external_phase "${ACTION}"
      FINAL_STATUS="$?"
    else
      echo "Mode must be external or managed."
      FINAL_STATUS=2
    fi
  fi
fi

echo "Launcher log: ${LOG_PATH}"
echo "The shell remains available; inspect status before retrying an interrupted phase."
finish_with_status() {
  return "$1"
}
finish_with_status "${FINAL_STATUS}"
