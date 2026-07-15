#!/usr/bin/env bash

# Reproducible user-local deterministic toolchain for the multilingual benchmark.
# No sudo, system-package changes, strict shell mode, or calling-shell exit is used.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
PREFIX="${LAPLACE_TOOLCHAIN_PREFIX:-${ROOT}/.tools/multilanguage}"
ACTION="${1:-report}"
LOG_DIR="${ROOT}/outputs/a6000_agent_team/tool_bootstrap"
LOG_FILE="${LOG_DIR}/bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log"
CONDA="$(command -v conda 2>/dev/null)"

mkdir -p "${LOG_DIR}" 2>/dev/null

tool_version() {
  NAME="$1"
  CANDIDATE="$(command -v "${NAME}" 2>/dev/null)"
  if [ -x "${PREFIX}/bin/${NAME}" ]; then
    CANDIDATE="${PREFIX}/bin/${NAME}"
  fi
  if [ -n "${CANDIDATE}" ]; then
    FIRST_LINE="$(${CANDIDATE} --version 2>&1 | sed -n '1p')"
    if [ "${NAME}" = "iverilog" ] || [ "${NAME}" = "vvp" ]; then
      FIRST_LINE="$(${CANDIDATE} -V 2>&1 | sed -n '1p')"
    fi
    echo "${NAME}: AVAILABLE path=${CANDIDATE} version=${FIRST_LINE}"
  else
    echo "${NAME}: MISSING"
  fi
}

report() {
  echo "Laplace multilingual deterministic toolchain"
  echo "isolated_prefix=${PREFIX}"
  for NAME in gcc clang cmake ctest iverilog vvp yosys verilator cppcheck; do
    tool_version "${NAME}"
  done
  echo "required_profile=clang 18.1.8, compiler-rt 18.1.8, CMake/CTest 3.30.5, Verilator 5.032"
  echo "activation=export PATH=\"${PREFIX}/bin:\$PATH\""
}

install_tools() {
  if [ -z "${CONDA}" ] || [ ! -x "${CONDA}" ]; then
    echo "Conda is unavailable; no installation was attempted."
    echo "Install a user-local conda-compatible solver, then rerun this command."
    return 0
  fi
  echo "Installing the pinned tool profile under ${PREFIX}."
  echo "This may download substantial packages and is intended as an explicit user action."
  CONDA_ACTION="create"
  if [ -f "${PREFIX}/conda-meta/history" ]; then
    CONDA_ACTION="install"
  fi
  "${CONDA}" "${CONDA_ACTION}" --yes --prefix "${PREFIX}" --channel conda-forge --strict-channel-priority \
    clang=18.1.8 clangxx=18.1.8 compiler-rt=18.1.8 \
    cmake=3.30.5 verilator=5.032 cppcheck=2.14.1 ninja=1.12.1 \
    2>&1 | tee -a "${LOG_FILE}"
  RESULT="${PIPESTATUS[0]}"
  if [ "${RESULT}" -ne 0 ]; then
    echo "Toolchain setup did not complete (conda status ${RESULT}); inspect ${LOG_FILE}."
    return 0
  fi
  echo "Toolchain setup completed."
  report
  return 0
}

case "${ACTION}" in
  report) report ;;
  install) install_tools ;;
  command)
    echo "${CONDA:-conda} create --yes --prefix ${PREFIX} --channel conda-forge --strict-channel-priority clang=18.1.8 clangxx=18.1.8 compiler-rt=18.1.8 cmake=3.30.5 verilator=5.032 cppcheck=2.14.1 ninja=1.12.1"
    ;;
  *)
    echo "Usage: $0 {report|install|command}"
    ;;
esac

# Always leave an interactive shell or tmux pane open after diagnostics.
true
