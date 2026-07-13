#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "ERROR: run from inside the cloned Laplace repository." >&2
  exit 2
fi
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv/bin/python is missing or not executable." >&2
  exit 3
fi
exec .venv/bin/python codex_a6000/scripts/preflight_laplace_a6000.py
