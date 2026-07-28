# Laplace administrator guide

## Runtime registry and first account

The registered-user YAML is external runtime state. `LAPLACE_USER_REGISTRY` defaults to `<state_root>/auth/registered_users.yaml`. Its parent must be mode `0700` and the file mode `0600`. The committed [example](../configs/registered_users.example.yaml) contains an intentionally invalid demonstration hash.

Bootstrap the required initial account:

```bash
PYTHONPATH=src .venv/bin/python -m research_workspace.user_admin bootstrap \
  --registry /var/lib/laplace/auth/registered_users.yaml \
  --session-store /var/lib/laplace/auth/sessions.sqlite3 \
  --email afasolino@unisa.it \
  --user-id usr_afasolino \
  --display-name "Alfonso Fasolino" \
  --capability-tier operator \
  --capability chat \
  --capability agent \
  --capability research \
  --capability operator \
  --capability admin \
  --capability personal_corpus \
  --capability shared_corpus_ingest \
  --capability repository_admin \
  --capability model_admin \
  --role admin \
  --default-lane quality
```

Bootstrap creates or replaces the account atomically, hashes a random one-time code with Argon2id, revokes prior sessions, and prints the code once. It never sets a hardcoded password.

Available commands are:

```text
bootstrap  add  list  validate  disable  enable
set-role  set-tier  set-capabilities  set-default-lane
authorize-repo  revoke-repo
reset-password  revoke-sessions  reload
```

Every command takes `--registry`. `reset-password`, `revoke-sessions`, and bootstrap/add accept `--session-store` where needed. No command accepts a plaintext password argument. Reset prints a new one-time activation code and revokes all sessions.

## Roles, capabilities, and model lanes

Roles, legacy profiles, named capabilities, and quality lanes are separate:

- Basic: chat only, with no repository IDs, tools, agent endpoints, or background work.
- Plus default: Chat, Agent, and personal corpus, with Agent bound to an explicitly authorized logical repository ID.
- Operator legacy default: Chat, Research, Operator, Admin, shared ingestion, repository admin, and model admin. Agent and personal corpus are deliberately independent.

The example `afasolino@unisa.it` combines all nine capabilities. `frcapone@unisa.it` is a normal user with `chat`, `agent`, and `personal_corpus`. To change an existing assignment locally:

```bash
PYTHONPATH=src .venv/bin/python -m research_workspace.user_admin set-capabilities \
  --registry /var/lib/laplace/auth/registered_users.yaml \
  --email frcapone@unisa.it \
  --capability chat \
  --capability agent \
  --capability personal_corpus
```

The Operator **Users** table provides the same independent checkboxes. Both paths revoke the affected user's sessions and record an audit event. Hidden navigation is not authorization; every endpoint checks its named capability server-side.

Quality, Standard, and Economy are model-quality lanes, not privileges. Changing a role, capability, enabled state, default lane, password, or repository authorization invalidates the affected user's active sessions.

## Repository registration and authorization

Repositories are registered server-side by an admin. The canonical root is resolved, required to equal the Git top-level directory, and pinned by device/inode identity. A user registry entry may reference only a registered repository ID at server startup.

Use **Operations → Repository onboarding** with `admin` role plus the independent `repository_admin` capability:

1. Register a logical ID and canonical server-side Git root.
2. Grant the logical ID to the user's stable user ID with `HEAD`, an exact commit, or another server-resolved ref.
3. The server pins the resolved commit, updates both authorization stores, revokes the user's sessions, and audits the action.
4. The user signs in again and confirms the logical ID appears in the Agent repository selector.

The equivalent registry-only synchronization command is:

```bash
PYTHONPATH=src .venv/bin/python -m research_workspace.user_admin authorize-repo \
  --registry /var/lib/laplace/auth/registered_users.yaml \
  --email researcher@example.org \
  --repo-id project-alpha
```

Revocation prevents all new/resumed actions. Existing audit evidence is retained.

Never add a repository path to a normal-user form. Canonical paths are accepted only by the administrator registration endpoint and shown only to Operator/admin inspection.

## Session and registry operations

- GUI: **Users → Revoke** revokes all sessions for one user.
- CLI: `revoke-sessions --session-store ...`.
- SIGHUP: atomically reloads the registry, preserves the last valid snapshot on failure, and revokes changed users.
- GUI admin action: **Operations → Reload registry** does the same.
- `reload --pid <operator-pid>` validates locally, then signals the service.

The SQLite session store contains hashes of opaque session and CSRF values, uses WAL/busy timeout, and enforces idle and absolute expiration.

## Models, profiles, queues, and GPU

Use the measured profile selection in `configs/selected_serving_profiles.json`. The selected default profile is P1 FP8 KV; P4 is the widest measured high-context profile. Do not repeat a full sweep unless the model/vLLM/profile code changed.

Only one main generative model is resident by default. Optional vision models must be loaded on demand and unloaded before restoring the text model. GPU VRAM, system RAM, iGPU memory, and NPU memory are not a single shared model-memory pool.

The lifecycle manager stores PID ownership and checks executable/model/host/port identity before stopping. Never use broad `pkill` patterns. Quality capacity is reserved; Basic/Plus work has explicit global and per-user limits; Deep Research admits two globally and one per user.

## Logs, audit, and provenance

HTTP logs are structured JSON with trace ID, method, path, status, and duration. Authentication audit contains a hash of normalized email—not the submitted password, activation code, session, or CSRF secret. Configure logrotate or the Caddy rolling JSON log example for bounded retention.

Artifact lifecycle events contain action, decision, reason, capability, artifact ID, repository/session binding, content hash, and trace ID. Normal downloads omit provenance; the Operator provenance endpoint is explicit.

## Backup and restore

Stop the Operator service or take SQLite-consistent backups before copying:

1. Copy the external state root to encrypted, access-controlled storage.
2. Exclude model files (reprovision from already authorized local artifacts) and temporary worktrees unless required.
3. By default exclude `auth/registered_users.yaml`, session DB, pseudonym key, bearer-token file, and any TLS private key.
4. If secrets are explicitly selected, encrypt them separately and preserve modes `0700`/`0600`.
5. Record application Git revision, selected profiles, registry schema version, and database hashes.

Restore into a new mode-`0700` state root, restore selected secrets only under explicit authorization, validate the registry, start on loopback, inspect readiness, and only then restore the reverse proxy.

## Reverse proxy and TLS

Use [REMOTE_ACCESS.md](REMOTE_ACCESS.md) and the Caddy/Nginx examples. Uvicorn and vLLM remain loopback-only. The proxy must pass an explicit Host, `X-Forwarded-*` only from a configured trusted proxy, stream without buffering, limit bodies, and terminate TLS. Remote mode disables PWA by default.

## Upgrades and rollback

1. Back up non-secret external state plus explicitly selected encrypted secrets.
2. Create a new implementation worktree/revision.
3. Install it into a dedicated environment and run all static/API/browser/security checks.
4. Stop the Operator service gracefully; do not stop unrelated GPU work.
5. Deploy the new revision and run health/readiness/auth/chat smokes.
6. On failure, stop only the new service, restore the prior code revision and compatible state backup, then verify hashes and readiness.

Database schema changes must be versioned, backed up before migration, applied transactionally, and rolled back from the pre-migration copy on failure. Registered-user and capability stores migrate legacy profile records to capability schema v2. Corpus and worktree stores initialize their versioned schemas idempotently.

Personal-corpus backup/restore, quotas, purge, and HMAC-key handling are documented in [PERSONAL_CORPUS.md](PERSONAL_CORPUS.md) and [STORAGE_AND_RETENTION.md](STORAGE_AND_RETENTION.md). Worktree cleanup and recovery are in [AGENT_WORKTREES.md](AGENT_WORKTREES.md).

## Emergency shutdown

```bash
sudo systemctl stop laplace-operator
sudo systemctl stop laplace-model-servers
```

Confirm ports 8765, 8102, and 8103 no longer listen, then inspect GPU processes. Do not signal a PID unless the Laplace owner record and live command identity both match. Do not close the user's shell or tmux session.
