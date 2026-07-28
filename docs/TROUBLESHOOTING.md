# Laplace troubleshooting

## The server refuses to start

- `registry_path_required` or missing registry: run the bootstrap command in [QUICKSTART.md](QUICKSTART.md).
- `registry_parent_permissions`: set the registry directory to mode `0700`.
- `registry_file_permissions`: set the YAML to mode `0600`.
- `unknown_repository_id`: register the logical repository server-side before referencing it in user YAML.
- Reverse-proxy configuration failure: supply an HTTPS `--external-url` plus matching explicit `--allowed-origin` and `--allowed-host`.
- Non-loopback HTTP refusal: use loopback/SSH/HTTPS. The insecure override is development-only.

## Sign-in or activation fails

Authentication responses intentionally do not reveal whether an email exists. Check the private authentication audit by trace ID, not by submitted secret. A bootstrapped account requires **First activation**, not normal sign-in. Re-running bootstrap/reset invalidates the prior code and all sessions.

After repeated failures, wait for the `Retry-After` interval. Do not weaken rate limiting.

## The session expired or was revoked

Sign in again. The visible composer text remains in the page, and an active conversation draft is saved server-side when possible. Role, tier, enabled state, default lane, password, registry access, and repository changes revoke affected sessions.

## Readiness is degraded

Inspect:

```bash
curl -sS http://127.0.0.1:8765/api/v1/health
curl -sS http://127.0.0.1:8765/api/v1/readiness
```

Health only proves process liveness. Readiness checks authentication state, session DB, state directories, and lane routing. The Operator dashboard shows sanitized model endpoint state.

## Chat is queued or unavailable

Keep the draft. Check the selected lane, queue position, model display name, and readiness. Quality has reserved capacity. Validation may escalate once to Quality; it never retries indefinitely or silently downgrades.

## Agent repository is absent or rejected

The client list is server-generated. Confirm both the repository authorization DB grant and the user's external `authorized_repo_ids`. Revoke/re-grant only by logical ID. Never work around a rejection by accepting a client filesystem path.

Path failures for traversal, symlink, hard link, mount, nested Git, worktree, submodule, environment, or subprocess CWD are security decisions. Inspect the audit record without exposing protected paths to the user.

## Artifact download reports an integrity failure

Stop using the artifact. Compare its current SHA-256 to the registry and lifecycle event log. Do not update the registry merely to match unexplained content. Restore from a verified source or regenerate through the governed workflow.

## Remote access fails

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_remote_access.py \
  --external-url https://laplace.example.org
```

Check DNS/certificate hostname/expiry, explicit Host/Origin, proxy IP trust, and that the proxy passes `X-Forwarded-Proto: https`. Ports 8102/8103 must remain externally unreachable. Do not expose Uvicorn directly as a shortcut.

## Browser tests cannot find Chromium

```bash
.venv/bin/playwright install chromium
```

Then rerun with `PYTHONPATH=src`. The screenshot tool fails if any required image is missing/empty or the browser logs an error.

## SQLite is locked

Laplace uses WAL and a 15-second busy timeout. Check for a long-lived external SQLite writer, confirm disk space/inodes, and retry. Do not delete WAL/SHM files while a process is active. For recovery, stop the owning service gracefully and take a consistent backup before repair.

## Disk is nearly full

Stop admitting expensive work, preserve source files, and rotate/archive derived logs, old worktrees, and certification outputs according to retention policy. Never delete source documents or an active SQLite file. Artifact creation is atomic and should fail without publishing incomplete metadata.

## Safe shutdown

Stop the Operator service, then only model PIDs whose ownership record and live command identity match. The lifecycle manager waits for graceful exit and does not force-kill after timeout. Confirm unrelated GPU processes remain unchanged.

If a trace is needed for support, share the trace ID and sanitized failure category—not a password, activation code, cookie, raw registry, private path, or document content.
