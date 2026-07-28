# Threat model v7

Laplace processes authorized but potentially hostile documents, archives, Markdown,
repositories, patches and model text. It also separates mutually untrusted users in
server mode. The trusted computing base is the local Python application, configured
storage root, browser origin, operating-system permissions, selected local provider
and operator-controlled backup provider. Models and document text are never trusted
as command sources.

Primary protected assets are credentials, sessions, private corpora and attachments,
repository grants/worktrees, artifacts, provenance/audit records, configuration and
backups. Boundaries are browser-to-API, API-to-domain service, domain-to-storage,
provider adapter-to-local endpoint, desktop sync-to-logical repository and
backup-provider-to-external key custody.

Controls include:

- bounded archive members, expansion size and compression ratio; no links/traversal;
- strict PDF/DOCX admission before parser handoff;
- normalized identifiers with collision rejection;
- raw HTML and unsafe Markdown-link rejection;
- explicit untrusted-source envelopes for prompt-injection content;
- Git top-level, relative-path, symlink/hardlink, nested-repository, submodule and
  inode-race checks;
- exact Host/Origin checks and double-submit CSRF checks;
- opaque rotated sessions, expiry, revision-bound authorization and capability checks;
- owner checks before corpus, artifact, repository or worktree lookup;
- credential-free loopback provider endpoints, redirects disabled and bounded bodies;
- structured log/config validation and redaction;
- migration/backup manifest hashes, safe restore paths and transactional rollback;
- resumable sync replay accepted only when operation, sequence and payload hash match.

The deterministic seed is `7003`; Hypothesis uses a derandomized 50-example profile
and reports minimal failing examples. CI runs the adversarial suite, Bandit high
severity scan, high-confidence secret scan and offline dependency consistency check.
It does not query a network vulnerability database.

