#!/usr/bin/env bash

# Reproducibly create the isolated CUDA-12-family vLLM environment used by
# Phase 2 and Phase 3. This never modifies the system driver or other venvs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
UV="${LAPLACE_UV:-$(command -v uv 2>/dev/null)}"
ENVIRONMENT="${LAPLACE_VLLM_ENVIRONMENT:-${ROOT}/.venv-vllm-cu129}"
CACHE="${LAPLACE_UV_CACHE_DIR:-${ROOT}/.runtime/uv-cache}"
PYTHON="${LAPLACE_VLLM_BOOTSTRAP_PYTHON:-${ROOT}/.venv/bin/python}"
FFMPEG_LIBRARY_PATH="${LAPLACE_FFMPEG_LIBRARY_PATH:-${ROOT}/.runtime/ffmpeg7/lib}"

BOOTSTRAP_STATUS=2
if [ ! -x "${PYTHON}" ]; then
  echo "Python 3.11 bootstrap interpreter is missing: ${PYTHON}"
else
  mkdir -p "${CACHE}" 2>/dev/null
  if [ -n "${UV}" ] && [ -x "${UV}" ]; then
    if [ ! -x "${ENVIRONMENT}/bin/python" ]; then
      env UV_CACHE_DIR="${CACHE}" "${UV}" venv --python "${PYTHON}" "${ENVIRONMENT}"
      BOOTSTRAP_STATUS="$?"
    else
      BOOTSTRAP_STATUS=0
    fi
    if [ "${BOOTSTRAP_STATUS}" -eq 0 ]; then
      env UV_CACHE_DIR="${CACHE}" "${UV}" pip install \
        --python "${ENVIRONMENT}/bin/python" \
        'vllm==0.25.0+cu129' \
        --torch-backend=cu129 \
        --extra-index-url https://wheels.vllm.ai/0.25.0/cu129 \
        --index-strategy unsafe-best-match
      BOOTSTRAP_STATUS="$?"
    fi
  else
    echo "uv is unavailable; using the slower pip fallback."
    if [ ! -x "${ENVIRONMENT}/bin/python" ]; then
      "${PYTHON}" -m venv "${ENVIRONMENT}"
      BOOTSTRAP_STATUS="$?"
    else
      BOOTSTRAP_STATUS=0
    fi
    if [ "${BOOTSTRAP_STATUS}" -eq 0 ]; then
      "${ENVIRONMENT}/bin/python" -m pip install \
        'vllm==0.25.0+cu129' \
        --extra-index-url https://wheels.vllm.ai/0.25.0/cu129 \
        --extra-index-url https://download.pytorch.org/whl/cu129
      BOOTSTRAP_STATUS="$?"
    fi
  fi
  if [ "${BOOTSTRAP_STATUS}" -eq 0 ]; then
    env PATH="${ENVIRONMENT}/bin:${PATH}" \
      LD_LIBRARY_PATH="${FFMPEG_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      "${ENVIRONMENT}/bin/python" -c \
      'import shutil, torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA_UNAVAILABLE"); print(shutil.which("ninja") or "NINJA_UNAVAILABLE")'
    BOOTSTRAP_STATUS="$?"
  fi
fi

finish_with_status() {
  return "$1"
}
finish_with_status "${BOOTSTRAP_STATUS}"
