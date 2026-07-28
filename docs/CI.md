# Continuous integration

Eight least-privilege workflows cover lint/types, unit/integration, browser fixtures,
package build, migrations, security, documentation and release candidates. Every
third-party action is pinned to a 40-character commit with its release tag in a
comment. Workflow permissions are `contents: read`; no AI-agent secret or
write-capable token is referenced.

The unit matrix covers Ubuntu and Windows with Python 3.11 and 3.12. Browser fixtures
run on Ubuntu with Chromium provisioned as a test dependency. Once dependencies are
installed, core tests use local fixtures, `NO_PROXY=*`, no provider endpoint and no
model acquisition.

The release-candidate workflow is manual and runs the same one-command CPU/fixture
certification as local review. Uploaded artifacts are limited to sanitized
`outputs/ci/` results with seven-day retention. They contain no runtime user state.

```bash
PYTHONPATH=src python scripts/run_architecture_release_v7_certification.py
```

Local validation:

```bash
PYTHONPATH=src python scripts/validate_ci.py \
  --output outputs/ci/ci-validation.json
```

This checks workflow set/names, YAML, immutable action pins, permissions, timeouts,
fixture mode, OS/Python matrix, script existence, artifact scope and absence of
GPU/model-server commands or secret references.

The CI does not perform live-model testing. The release record reports
`BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE` until a separately authorized live
certification is run.
