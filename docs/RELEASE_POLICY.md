# Release policy

Laplace v2.0.0 uses Semantic Versioning. The v2 release surface preserves the
certified v1 control-plane contracts and adds explicitly versioned Core, Zetsu,
repository-agent, worktree, result-delivery, ACI, memory, and lifecycle behavior.
Breaking changes require a changelog entry, compatibility adapter or ordered
migration, old-state fixtures and an explicit rollback boundary.

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
more. The corrective branch does not merge, tag, or modify the immutable
production checkout. Promotion remains a separate human decision.

Live-model certification is a separate release annotation. CPU/fixture success must
not be represented as model quality. The mandatory C8 SiliconMind-vs-CodeV RTL
experiment is recorded as `BLOCKED` when its local candidate artifact or certified
full-topology toolchain is absent; no model route is silently promoted.

## v8 decision boundary

The non-GPU candidate must close all P0/P1 findings and produce a verified
sanitized archive before live eligibility. The only pre-live approval is
`GO_FOR_CONTROLLED_LIVE_GPU_CERTIFICATION`. Final
`GO_FOR_RELEASE_REVIEW_AFTER_LIVE_CERTIFICATION` requires a live PASS; blocked or
yielded GPU coordination is never promoted to a live PASS. The review does not
merge, tag, publish, or release.
