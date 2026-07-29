# v8 package audit

`scripts/package_release.py` builds the wheel and normalized sdist twice with the
commit timestamp as `SOURCE_DATE_EPOCH`. Matching filenames and SHA-256 values are
required. Members must be relative, bounded, regular, non-symlink files and must
exclude Git metadata, environment files, models, runtime state, logs, data, and
outputs.

The wheel is installed with `--no-deps --no-index` in a fresh virtual
environment. The audit imports package and metadata under isolated Python, then
runs installed `laplace --version` with the exact 40-character build revision.
All declared entry points are exercised with `--help` in the locked development
environment.

`requirements.lock` is an exact-version CPU/fixture development lock; it does not
contain wheel hashes because it is not an artifact-resolution lock. Artifact
SHA-256 values, dependency inventory, CycloneDX-style offline SBOM, installed
license metadata, and `pip check` are recorded. Offline inventory does not claim
current vulnerability-database coverage. A current online scanner remains a
release-operator action in an approved network environment.

Linux packaging is executed locally. Windows entry points and path behavior are
covered by the remote Python 3.11/3.12 matrix; when remote CI is unavailable this
is recorded as an environment limitation, not PASS.

