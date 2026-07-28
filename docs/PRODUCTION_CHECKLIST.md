# Laplace production checklist

Use this as a fail-closed release gate.

## Identity and state

- [ ] External state root is outside Git and mode `0700`.
- [ ] Registered-user parent is `0700`; registry is `0600`.
- [ ] `afasolino@unisa.it` is enabled, Operator/admin, default Quality.
- [ ] Activation was completed with a one-time local code; no password is hardcoded.
- [ ] Live password hashes, codes, cookies, CSRF values, bearer tokens, and keys are absent from Git and bundles.
- [ ] Invalid registry reload retains the last valid snapshot.
- [ ] Role/tier/disable/password/repository changes revoke affected sessions.

## Network and authentication

- [ ] Uvicorn binds only `127.0.0.1:8765`.
- [ ] vLLM binds only loopback ports 8102/8103.
- [ ] HTTPS external URL, Host, Origin, and proxy IPs are explicit.
- [ ] TLS hostname matches and certificate has at least 14 days remaining.
- [ ] Remote cookie is HttpOnly, Secure, SameSite Strict, Path `/`.
- [ ] Login has generic failures, per-IP/email backoff, body/time bounds, and no secret logs.
- [ ] CSRF, Host, Origin, forwarded-header, CORS, CSP, clickjacking, MIME, and referrer tests pass.
- [ ] Remote PWA is disabled.

## Authorization and isolation

- [ ] Basic receives chat only and no repository IDs/tool schemas/agent controls.
- [ ] Plus sees only server-authorized logical repositories.
- [ ] Agent worktree and path-escape tests pass.
- [ ] Conversation, research, and artifact cross-user tests pass.
- [ ] Repository revocation blocks subsequent/resumed actions.
- [ ] Artifact read/export verifies SHA-256.
- [ ] Normal exports contain no internal provenance.

## Models, queues, and robustness

- [ ] Selected profile and routes resolve against the installed local vLLM.
- [ ] No automatic model download is possible.
- [ ] Quality reservation and global/per-user limits match configuration.
- [ ] Deep Research limit is two globally and one per user.
- [ ] Health and readiness pass; unavailable endpoints show degraded state.
- [ ] SQLite stores use WAL, busy timeout, private modes, and recover after restart.
- [ ] Cancellation releases the request/admission resource.
- [ ] Logs have bounded rotation/retention.
- [ ] Backup and rollback were rehearsed without silently copying secrets.

## UI and documentation

- [ ] Desktop, tablet, mobile, keyboard, focus, labels, names, contrast, and reduced-motion checks pass.
- [ ] Malicious Markdown, link, filename, source-title, repository-name, and error text tests pass.
- [ ] Chat, response details, diff/tests, research evidence, and Operator status are readable.
- [ ] Browser storage contains no authentication credential.
- [ ] All six guides and all 12 screenshots exist.
- [ ] Screenshot manifest hashes match and secret scan passes.

## Live and shutdown

- [ ] Unrelated GPU PIDs were recorded before startup and remain untouched.
- [ ] Existing local main and CodeV artifacts were used.
- [ ] Real Quality chat and selected repository-bound Agent smoke pass.
- [ ] Reverse-proxy/SSH fixture passes; model ports remain private.
- [ ] Only Laplace-owned processes are stopped.
- [ ] Final endpoint/GPU observations and preserved unrelated PID evidence are recorded.
- [ ] Certification tarball manifest and SHA-256 verify.
