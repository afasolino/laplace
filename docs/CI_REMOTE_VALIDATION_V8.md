# v8 CI and remote validation

All workflows use read-only repository permissions, immutable 40-character
action pins, bounded timeouts, fixture-only test settings, sanitized artifacts,
and no model/GPU commands or application secrets.

Local equivalents:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_ci.py \
  --output outputs/release_candidate_v8_ci_<UTC>/ci_validation.json
PYTHONPATH=src .venv/bin/pytest -q --ignore=tests/test_operator_gui_e2e.py
PYTHONPATH=src .venv/bin/pytest -q tests/test_operator_gui_e2e.py
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/research_workspace
```

The remote matrix is Ubuntu and Windows on Python 3.11 and 3.12. Remote CI is
triggered only by pushing `feature/release-candidate-review-v8`; it does not
merge, tag, publish, or access a GPU. If authentication or remote execution is
unavailable, record `NOT_EXECUTED_ENVIRONMENT_LIMITATION` with the exact command:

```bash
git push --dry-run origin feature/release-candidate-review-v8
git push origin feature/release-candidate-review-v8
gh run list --branch feature/release-candidate-review-v8
```

Polling follows 5-minute, then 10-minute, then 15-minute intervals. Repeated
interactive status checks are prohibited.

