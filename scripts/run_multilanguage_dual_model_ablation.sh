#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "${ROOT}/.tools/multilanguage/bin" ]]; then
  export PATH="${ROOT}/.tools/multilanguage/bin:${PATH}"
fi
ACTION="${1:-}"
PYTHON="${LAPLACE_PYTHON:-${ROOT}/.venv/bin/python}"
CONFIG="${LAPLACE_ABLATION_CONFIG:-${ROOT}/codex_a6000/experiments/multilanguage_dual_model_ablation_v1/experiment.json}"
OUTPUT_ROOT="${ROOT}/outputs/a6000_agent_team/experiments/multilanguage_dual_model_ablation_v1"
CORPUS_ROOT="${OUTPUT_ROOT}/corpus"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${OUTPUT_ROOT}/launcher_logs"
LOG_PATH="${LOG_DIR}/${STAMP}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

if [[ ! "${ACTION}" =~ ^(phase1|phase2|status|merge)$ ]]; then
  echo "Usage: $0 {phase1|phase2|status|merge}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Configured Laplace Python is not executable: ${PYTHON}" >&2
  exit 2
fi

cd "${ROOT}"
echo "Laplace dual-model ablation launcher: ${STAMP}"
echo "Configuration: ${CONFIG}"
echo "Model servers are expected to be running separately; this launcher never starts them."

if [[ -z "${LAPLACE_ABLATION_BASE_REVISION:-}" ]]; then
  echo "LAPLACE_ABLATION_BASE_REVISION must name the reviewed clean checkpoint commit." >&2
  exit 2
fi
if [[ "${ACTION}" == "status" ]]; then
  "${PYTHON}" -m research_workspace.multilanguage_ablation phase-status --config "${CONFIG}"
  exit
fi
if [[ "${ACTION}" == "merge" ]]; then
  "${PYTHON}" -m research_workspace.multilanguage_ablation merge-report --config "${CONFIG}"
  exit
fi

if [[ -z "${LAPLACE_ABLATION_HELD_OUT_ROOT:-}" ]]; then
  echo "LAPLACE_ABLATION_HELD_OUT_ROOT must name the evaluator-owned pack outside the repository." >&2
  exit 2
fi

"${PYTHON}" -m research_workspace.multilanguage_ablation "validate-${ACTION}" --config "${CONFIG}"
"${PYTHON}" -m research_workspace.multilanguage_ablation validate-manifest --config "${CONFIG}"
"${PYTHON}" -m research_workspace.multilanguage_ablation validate-corpus --config "${CONFIG}" --corpus-overlay "${CORPUS_ROOT}"
"${PYTHON}" -m research_workspace.multilanguage_ablation validate-heldout --config "${CONFIG}"
"${PYTHON}" -m research_workspace.multilanguage_ablation plan-only --phase "${ACTION}" --config "${CONFIG}"
"${PYTHON}" -m research_workspace.multilanguage_ablation validate-runtime --phase "${ACTION}" --config "${CONFIG}"
"${PYTHON}" -m research_workspace.multilanguage_ablation "run-${ACTION}" --config "${CONFIG}"

echo "${ACTION} results are preserved under ${OUTPUT_ROOT}; use '$0 status' or '$0 merge'."
