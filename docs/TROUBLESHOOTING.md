# Laplace troubleshooting

## v2 control-plane preflight

Start the normal development topology explicitly:

```bash
laplace zetsu start --nocodev --repo /home/giando/work/laplace-v2
laplace zetsu status --repo /home/giando/work/laplace-v2 --json
laplace zetsu sessions --json
```

`--nocodev` means CodeV is intentionally disabled. It is not a readiness failure.
Use the full topology only for an explicitly authorized RTL experiment, then stop
it and return to `laplace zetsu start --nocodev`.

The repository section of Zetsu status separately reports the canonical root,
logical `repo_id`, registration, owner grant, granted revision, current HEAD,
clean/dirty state, and `agent_task_ready`. A degraded model/MCP status must not be
interpreted as repository readiness, and a registered repository without a grant
must remain `repository_not_authorized`.

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

Independent capability changes also revoke sessions. Sign in again before deciding that a newly granted navigation item or repository is missing.

## Readiness is degraded

Inspect:

```bash
curl -sS http://127.0.0.1:8765/api/v1/health
curl -sS http://127.0.0.1:8765/api/v1/readiness
```

Health only proves process liveness. Readiness checks authentication state, session DB, state directories, and lane routing. The Operator dashboard shows sanitized model endpoint state.

## Chat is queued or unavailable

Keep the draft. Check the selected lane, queue position, model display name, and readiness. Quality has reserved capacity. Validation may escalate once to Quality; it never retries indefinitely or silently downgrades.

If a selector is empty, inspect the authenticated `/api/v1/providers` response. It
should contain enabled logical routes and sanitized capabilities but no endpoints.
Validate configuration offline with `--validate-config` before doing any
deployment-specific provider diagnosis.

## Agent repository is absent or rejected

The client list is server-generated. Confirm both the repository authorization DB grant and the user's external `authorized_repo_ids`. Revoke/re-grant only by logical ID. Never work around a rejection by accepting a client filesystem path.

Path failures for traversal, symlink, hard link, mount, nested Git, worktree, submodule, environment, or subprocess CWD are security decisions. Inspect the audit record without exposing protected paths to the user.

For `per_user_worktree_quota`, inspect `laplace zetsu sessions --json`. Clean
failed, expired, or abandoned sessions are reconciled before a new admission;
clean worktrees are released through lifecycle machinery. A dirty worktree is
intentionally retained and is never silently deleted. `STALE_GRANT` or
`repository_grant_changed` requires a fresh worktree after authorization remains
valid.

For `repository_state_not_materialized`, the target exists only in the caller's
uncommitted checkout or otherwise is absent from the exact granted commit. Do not
ask Zetsu to copy dirty files or auto-commit them. Create a normal checkpoint
commit and delegate a new session.

## Personal corpus upload or indexing fails

- Read the exact manifest reason; do not rename an executable merely to match an allowed extension.
- Parser timeout, encrypted PDF, macro content, external DOCX relationship, MIME/magic mismatch, nested archive, traversal, and Unicode collision are security rejections.
- For interrupted staging, reopen Knowledge, reselect the same bytes, and preview again. Identical staged content is idempotent.
- For disk pressure or quota, cancel staging. Do not delete another user's source or active SQLite/WAL files.
- Only accepted files are indexable, and retrieval remains disabled until explicit index confirmation.
- A deleted source disappears from retrieval immediately even though bytes may remain during soft-delete retention.

See [PERSONAL_CORPUS.md](PERSONAL_CORPUS.md) for formats, defaults, backup, and recovery.

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

CPU/fixture tests do not claim live-model quality. The C8 candidate comparison is
explicitly `BLOCKED` when the SiliconMind artifact or certified full-topology
vLLM executable is absent; see the C8 certification rather than inventing a
metric or changing the CodeV route.

## v8 coordination statuses

- `BLOCKED_BY_SPECDEC_ACTIVE`: leave SpecDec untouched and defer the live gate.
- `BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP`: resolve the observation or process
  identity failure; do not start a model.
- `YIELDED_TO_SPECDEC`: Laplace-owned groups were released and this run must not
  reacquire the GPU.
- `sync_target_not_clean`, `sync_branch_conflict`, or
  `sync_patch_plan_mismatch`: inspect local changes and create a new plan; never
  force apply.

## Zetsu agent or Qwen3.8 migration fails

For a Zetsu agent failure, keep the persistent checkpoint and inspect the exact
failure category. `zetsu_agent_checkpoint_*` indicates a resume binding/schema
problem; do not restart the same session with different owner/repository/base
revision/objective. Cancellation and command/step/wall limits are enforced during
the loop. A mutating task cannot finish until deterministic verification passes
after its latest mutation. `zetsu_agent_compaction_*` is fail-closed: diagnose
serving/context behavior before resuming rather than discarding exact task state.

For Qwen3.8 preparation, do not infer missing MTP from shard filenames. Validate
`config.json` plus safetensor index/header tensor keys. Resolve the selected
checkpoint to an immutable Hugging Face revision and retain the recorded artifact
hashes. Do not work around a P6/P7 serving incompatibility by broadly upgrading
Laplace; isolate the smallest justified serving-environment change and rerun the
affected gate. A failed P7 disables MTP; it does not require abandoning a
successfully certified P6.
