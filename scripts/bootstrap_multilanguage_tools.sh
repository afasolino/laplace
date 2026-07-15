#!/usr/bin/env bash

# Reproducible user-local deterministic toolchain for the multilingual benchmark.
# No sudo, system-package changes, strict shell mode, or calling-shell exit is used.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
PREFIX="${LAPLACE_TOOLCHAIN_PREFIX:-${ROOT}/.tools/multilanguage}"
PKGS_DIR="${PREFIX}/.conda-pkgs"
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
  echo "isolated_package_cache=${PKGS_DIR}"
  for NAME in gcc clang cmake ctest iverilog vvp yosys verilator cppcheck; do
    tool_version "${NAME}"
  done
  echo "required_profile=clang 18.1.8, compiler-rt 18.1.8, CMake/CTest 3.30.5, Verilator 5.032"
  echo "activation=export PATH=\"${PREFIX}/bin:\$PATH\""
}

run_probe() {
  LABEL="$1"
  shift
  echo "COMMAND[${LABEL}]=$(printf '%q ' "$@")" | tee -a "${PROBE_LOG}"
  "$@" >>"${PROBE_LOG}" 2>&1
  STATUS="$?"
  echo "RETURN_CODE[${LABEL}]=${STATUS}" | tee -a "${PROBE_LOG}"
  return "${STATUS}"
}

probe_tools() {
  PROBE_ROOT="${LOG_DIR}/probes_$(date -u +%Y%m%dT%H%M%SZ)"
  PROBE_LOG="${PROBE_ROOT}/probe.log"
  mkdir -p "${PROBE_ROOT}" 2>/dev/null
  export PATH="${PREFIX}/bin:${PATH}"
  echo "Laplace multilingual functional probes" | tee -a "${PROBE_LOG}"
  echo "isolated_prefix=${PREFIX}" | tee -a "${PROBE_LOG}"

  cat >"${PROBE_ROOT}/asan.c" <<'EOF'
#include <stdlib.h>
int main(void) {
    int *values = malloc(2 * sizeof(*values));
    if (values == NULL) return 2;
    values[2] = 7;
    free(values);
    return 0;
}
EOF
  run_probe asan_compile "${PREFIX}/bin/clang" -fsanitize=address -fno-omit-frame-pointer \
    "${PROBE_ROOT}/asan.c" -o "${PROBE_ROOT}/asan_probe"
  ASAN_COMPILE="$?"
  ASAN_OPTIONS="detect_leaks=0:halt_on_error=1" run_probe asan_execute "${PROBE_ROOT}/asan_probe"
  ASAN_RUN="$?"
  if [ "${ASAN_COMPILE}" -eq 0 ] && [ "${ASAN_RUN}" -ne 0 ] && grep -q "AddressSanitizer" "${PROBE_LOG}"; then
    echo "RESULT[asan]=PASS" | tee -a "${PROBE_LOG}"
  else
    echo "RESULT[asan]=FAIL" | tee -a "${PROBE_LOG}"
  fi

  cat >"${PROBE_ROOT}/ubsan.c" <<'EOF'
#include <limits.h>
int main(void) {
    volatile int maximum = INT_MAX;
    volatile int result = maximum + 1;
    return result == 0;
}
EOF
  run_probe ubsan_compile "${PREFIX}/bin/clang" -fsanitize=undefined \
    -fno-sanitize-recover=undefined "${PROBE_ROOT}/ubsan.c" -o "${PROBE_ROOT}/ubsan_probe"
  UBSAN_COMPILE="$?"
  run_probe ubsan_execute "${PROBE_ROOT}/ubsan_probe"
  UBSAN_RUN="$?"
  if [ "${UBSAN_COMPILE}" -eq 0 ] && [ "${UBSAN_RUN}" -ne 0 ] && grep -q "runtime error" "${PROBE_LOG}"; then
    echo "RESULT[ubsan]=PASS" | tee -a "${PROBE_LOG}"
  else
    echo "RESULT[ubsan]=FAIL" | tee -a "${PROBE_LOG}"
  fi

  mkdir -p "${PROBE_ROOT}/ctest"
  cat >"${PROBE_ROOT}/ctest/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.20)
project(laplace_ctest_probe C)
enable_testing()
add_executable(probe probe.c)
add_test(NAME deterministic_probe COMMAND probe)
EOF
  cat >"${PROBE_ROOT}/ctest/probe.c" <<'EOF'
#include <stdio.h>
int main(void) {
    puts("CTEST_PASS");
    return 0;
}
EOF
  run_probe cmake_configure "${PREFIX}/bin/cmake" -S "${PROBE_ROOT}/ctest" \
    -B "${PROBE_ROOT}/ctest/build" -DCMAKE_C_COMPILER="${PREFIX}/bin/clang"
  CMAKE_CONFIGURE="$?"
  run_probe cmake_build "${PREFIX}/bin/cmake" --build "${PROBE_ROOT}/ctest/build"
  CMAKE_BUILD="$?"
  run_probe ctest_execute "${PREFIX}/bin/ctest" --test-dir "${PROBE_ROOT}/ctest/build" \
    --output-on-failure
  CTEST_RUN="$?"
  if [ "${CMAKE_CONFIGURE}" -eq 0 ] && [ "${CMAKE_BUILD}" -eq 0 ] && [ "${CTEST_RUN}" -eq 0 ]; then
    echo "RESULT[cmake_ctest]=PASS" | tee -a "${PROBE_LOG}"
  else
    echo "RESULT[cmake_ctest]=FAIL" | tee -a "${PROBE_LOG}"
  fi

  cat >"${PROBE_ROOT}/timed_tb.sv" <<'EOF'
module timed_tb;
  initial begin
    #5;
    $display("TIMED_PASS");
    $finish;
  end
endmodule
EOF
  run_probe verilator_compile "${PREFIX}/bin/verilator" --binary --timing \
    --Mdir "${PROBE_ROOT}/verilator_obj" "${PROBE_ROOT}/timed_tb.sv"
  VERILATOR_COMPILE="$?"
  run_probe verilator_execute "${PROBE_ROOT}/verilator_obj/Vtimed_tb"
  VERILATOR_RUN="$?"
  if [ "${VERILATOR_COMPILE}" -eq 0 ] && [ "${VERILATOR_RUN}" -eq 0 ] && grep -q "TIMED_PASS" "${PROBE_LOG}"; then
    echo "RESULT[verilator_timed_binary]=PASS" | tee -a "${PROBE_LOG}"
  else
    echo "RESULT[verilator_timed_binary]=FAIL" | tee -a "${PROBE_LOG}"
  fi
  echo "probe_log=${PROBE_LOG}"
  return 0
}

install_tools() {
  if [ -z "${CONDA}" ] || [ ! -x "${CONDA}" ]; then
    echo "Conda is unavailable; no installation was attempted."
    echo "Install a user-local conda-compatible solver, then rerun this command."
    return 0
  fi
  echo "Installing the pinned tool profile under ${PREFIX}."
  echo "This may download substantial packages and is intended as an explicit user action."
  mkdir -p "${PREFIX}" "${PKGS_DIR}" 2>/dev/null
  CONDA_ACTION="create"
  if [ -f "${PREFIX}/conda-meta/history" ]; then
    CONDA_ACTION="install"
  fi
  CONDA_PKGS_DIRS="${PKGS_DIR}" "${CONDA}" "${CONDA_ACTION}" --yes --prefix "${PREFIX}" --channel conda-forge --strict-channel-priority \
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
  probe) probe_tools ;;
  command)
    echo "CONDA_PKGS_DIRS=${PKGS_DIR} ${CONDA:-conda} create --yes --prefix ${PREFIX} --channel conda-forge --strict-channel-priority clang=18.1.8 clangxx=18.1.8 compiler-rt=18.1.8 cmake=3.30.5 verilator=5.032 cppcheck=2.14.1 ninja=1.12.1"
    ;;
  *)
    echo "Usage: $0 {report|install|probe|command}"
    ;;
esac

# Always leave an interactive shell or tmux pane open after diagnostics.
true
