# Laplace Chat

`laplace chat` is a terminal client for the resident Laplace Operator.

It intentionally does **not** instantiate `LaplaceCore`,
`PersonalCorpusStore`, `TieredServingService`, a scheduler, a sandbox manager,
or another repository agent. The resident Operator remains the single
composition root.

Current v2 bindings:

- ordinary conversation: `POST /api/v1/chat`
- create engineering session: `POST /api/v1/agent/sessions`
- engineering turn:
  `POST /api/v1/agent/sessions/{session_id}/messages` (the current v2 route)
- deterministic status: `GET /api/v1/agent/sessions/{session_id}/status`
- durable agent transcript:
  `GET /api/v1/agent/sessions/{session_id}/messages`
- deterministic cancellation: `POST /api/v1/agent/sessions/{session_id}/cancel`
- bounded result page:
  `GET /api/v1/agent/sessions/{session_id}/results/{result_id}`

The client reads `/api/v1/openapi.json` and builds only fields actually declared by the
current Operator request model. An unknown required field is a contract error,
not an invitation to guess.

## Usage

Start the Laplace runtime, normally without CodeV:

```bash
laplace zetsu start --nocodev
```

Then:

```bash
cd /path/to/registered/repository
laplace chat
```

Default mode is `agent`. The first engineering instruction creates one
owner-bound Operator session. Every later natural-language instruction uses the
same session and its same isolated worktree through `/messages`; it never creates
one worktree per prompt. Use `/new` only when you deliberately need a separate
task/worktree.

Without `--verification`, agent turns are inspection-only: the repository agent
refuses edits without a deterministic verifier. To permit a bounded edit turn,
provide a server-validated verifier and confirm that turn:

```bash
laplace chat --verification pytest -q tests/test_component.py
```

`--access read` refuses a verifier-backed (therefore potentially mutating) agent
turn. `--access confirm` asks before it sends one; `--access write` is an
explicit non-interactive choice for an already-authorized local terminal.

Useful options:

```bash
laplace chat --mode chat --access read
laplace chat --resume last
laplace chat --repo-id laplace-v2
```

Inside the terminal:

```text
/mode agent|chat
/access read|confirm|write
/status
/diff
/tests
/result <artifact> [offset]
/cancel
/contract
/context
/history
/compact
/model [quality|standard|economy]
/new
/resume <session-id|last>
/exit
```

`/status`, `/diff`, `/tests`, `/result`, `/cancel`, `/contract`, `/context`,
and `/history` do not call the model merely to monitor an existing task.
`/diff` obtains one owner-bound durable `handoff.patch` page when a result is
available; it never asks a model to reconstruct a diff. Large result artifacts
remain paged rather than being dumped into the terminal.

## Authentication

The client is loopback-only and never follows HTTP redirects.

Token resolution order:

1. `LAPLACE_CHAT_TOKEN`
2. `LAPLACE_ZETSU_TOKEN`
3. file named by `LAPLACE_CHAT_TOKEN_FILE` (must not be group/world accessible)

The actual authentication header is derived from the Operator OpenAPI security
scheme when available. If the current deployment uses a custom dependency not
represented in OpenAPI, an available token is sent as a Bearer token.

Before every authenticated mutation request, the client performs
`POST /api/v1/session`, reads the returned `csrf_token`, and sends that nonce as
`X-CSRF-Token`. The same loopback HTTP opener is reused, so an Operator
same-origin session cookie, if issued, is preserved automatically. Neither the
authentication token nor the CSRF token is persisted in chat-session state or
printed. Read-only status/OpenAPI operations do not fetch or send a CSRF token.

No authentication or CSRF token is printed by the CLI.
