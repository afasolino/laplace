# v8 CI and remote validation

All workflows use read-only repository permissions, immutable 40-character
action pins, bounded timeouts, fixture-only test settings, sanitized artifacts,
and no model/GPU commands or application secrets. Checkout v5 and setup-python
v6 are pinned to the exact commits resolved from their official repositories;
the package builder is pinned to `uv==0.11.28`.

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

The first remote run demonstrated and then verified the package-builder repair.
It also exposed clean-clone and Windows import assumptions that local prepared
worktree tests could not reveal; these are recorded as RCV8-009 and RCV8-010.
Only the conclusions for the final exact commit are eligible for certification.
