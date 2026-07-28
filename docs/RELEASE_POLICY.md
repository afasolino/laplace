# Release policy

Laplace uses Semantic Versioning. Before 1.0, a minor release may add or deliberately
change a documented API or state contract; patch releases preserve every supported
schema and command. Breaking changes require a changelog entry, compatibility
adapter or ordered migration, old-state fixtures and an explicit rollback boundary.

Every build records the semantic version from package metadata and a 40-character
Git revision. `laplace --version` and `/api/v1/version` report both. A release
candidate must come from a clean commit, pass the CPU/fixture certification command,
and retain its certification archive hash.

Release artifacts are built twice with a fixed `SOURCE_DATE_EPOCH`. Both wheel and
normalized sdist must be byte-identical, contain only safe relative regular
files/directories, and install into an isolated environment without network access.
Models, outputs, state, credentials, databases, logs and environment files are
excluded.

The dependency lock pins the CPU/fixture development set. Dependency and license
inventories are regenerated from the certification interpreter. Missing or ambiguous
license metadata is a review item, never silently assigned by Laplace.

Branches are protected through required CI checks and review. Release automation has
read-only repository permissions unless a separately approved tag/release job needs
more. This v7 task does not push, tag or merge.

Live-model certification is a separate release annotation. CPU/fixture success must
not be represented as model quality. For v7 its exact status is
`BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE`.
