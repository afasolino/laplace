# Agent, personal corpus, and GUI v6 audit

Date: 2026-07-28
Certified base: `0d9a1d54f445ad25a8bb84d3133c3377f446c476`
Implementation branch: `feature/agent-personal-corpus-gui-v6`

## Audit scope and input evidence

The implementation worktree started clean at the certified base. The stable checkout
`/home/giando/work/laplace` was also clean and remains a read-only reference for this
work. No Laplace, vLLM, uvicorn, Playwright, Chrome, or Chromium processes were running
at the start of the audit. Python 3.11.15 and `uv` 0.11.28 were available. The expected
test, lint, type-check, security, and browser tools were installed; the optional
`python-docx` package was not installed.

`laplace_agent_gui_followup_issues.md` is not present in either checkout, any visible
branch history, or `/home/giando/work` at the start of this work. The complete
`prompts/laplace_codex_agent_personal_corpus_gui_v6.md` is therefore the authoritative
issue inventory. This is an input limitation, not a reason to weaken any v6 acceptance
criterion.

The baseline `PYTHONPATH=src pytest -q` run completed all collection and test execution.
All non-network tests passed. Two existing real-HTTP/browser tests could not create an
AF_INET socket because the managed execution sandbox returned `PermissionError:
[Errno 1] Operation not permitted`; they are rerun outside that restriction during the
browser/live phase.

## Existing control-to-enforcement map

| Current GUI control | Frontend handler | API endpoint | Service method | Authorization before v6 | Storage | Audit/provenance | Known limitation at base |
|---|---|---|---|---|---|---|---|
| Chat composer | `submitChat` | `POST /api/v1/chat` | `TieredServingService.chat` | Any enabled Basic/Plus/Operator tier | conversation SQLite plus tier audit JSONL | tier audit and conversation messages | Text clears only after completion; static domains; synchronous coarse progress |
| Chat domain | inline static `<select>` | `POST /api/v1/chat` | `_route` | Same as Chat | none | domain in tier audit | Only General and SystemVerilog shown; unknown IDs not registry-validated |
| Conversation rail | conversation handlers | `/api/v1/conversations*` | `ConversationStore` | authenticated owner | owner-keyed SQLite | HTTP log; message metadata | No explicit corpus retrieval state |
| Deep Research | research handlers | `/api/v1/research/jobs*` | `DeepResearchService`, `ResearchAdmissionStore` | Operator tier only | external research job layout and SQLite admission | Operator action events and research records | Research is coupled to Operator tier |
| Repository Agent | `runAgent`, `cancelAgent` | `/api/v1/agent/sessions*` | `TieredServingService`, `AgentSandboxManager` | Plus tier exactly; registry list plus repository-grant store on create | external worktree directories; bindings only in memory | tier audit JSONL | Operator cannot also be Agent; no restart recovery, list/history, resume, close/discard/export, or quotas |
| Agent repository selector | `buildAgentView` | capability response plus Agent create | `authorized_for_user`, `require_grant` | Plus tier and both authorization stores | registry YAML plus repository SQLite | grant changes audited by Operator API | Empty state is only an empty selector; stale and quota states absent |
| Agent domain | static JavaScript options | Agent run endpoint | `_route`, validator/backend | Plus tier | none | domain indirectly in validation | Only Python/SystemVerilog; not backend-registry driven |
| Operator navigation | `navDefinition` | Operator endpoints | multiple | Operator tier exactly | external Operator state | Operator events | Hidden controls are derived from a mutually exclusive tier |
| User access table | `loadUsers` | `/api/v1/admin/users`, tier update endpoint | registry and `UserCapabilityStore` | Operator tier | private registry YAML and capability SQLite | auth and Operator audit | One tier string cannot express Agent+Operator |
| Model/GPU controls | `loadModels` | model/profile endpoints | lifecycle/profile operators | Operator tier | external state | Operator events and owned-process records | Must remain independently model-admin protected |
| Governed corpus inventory | no end-user corpus view | `GET /api/v1/corpora` | filesystem inventory | Operator tier | governed corpus under external state | research provenance | No owner-private personal corpus |
| Markdown response | `renderMarkdown` | n/a | n/a | browser rendering only | none | none | Safe text construction exists, but GFM tables and table copy are absent |

## Security and storage findings

- Registered-email authentication, Argon2id password handling, CSRF validation,
  Host/Origin checks, secure response headers, and session revision checks already
  provide a sound base and must be preserved.
- Registry schema version 1 permits only a legacy tier. Capability SQLite also stores
  only that tier. Both need a backward-compatible, versioned migration.
- Repository paths are server registered and path validation rejects traversal,
  symlinks, hardlinks, mounts, nested repositories, and submodules. Worktree APIs must
  continue to accept only logical repository IDs.
- Worktree bindings contain canonical paths and are currently returned by the API.
  Non-Operator responses need logical, sanitized records.
- Worktree session state is process memory and cannot recover after restart.
- No `<state_root>/personal_corpora/` store, upload quarantine, corpus metadata,
  extraction/chunk hashes, personal retrieval snapshot, purge state, or corresponding
  provenance exists.
- Existing request-size middleware is useful but multipart entry, ZIP central
  directory, extracted-text, owner quota, and disk-pressure limits do not exist.

## v6 implementation decisions

1. Add named independent capabilities while retaining a legacy tier only as a
   migration/default profile. Endpoint checks use named capabilities.
2. Migrate both the YAML registry and SQLite capability store without invalidating
   schema-v1 input. Any effective access change changes the user revision and revokes
   or invalidates sessions.
3. Add a versioned domain registry used by both API validation and all selectors.
4. Persist worktree lifecycle records under the external Agent state root, enforce
   per-user/global quotas, recover inspectable sessions after restart, and redact
   canonical paths from owner responses.
5. Add an owner-keyed, HMAC-pseudonymized personal corpus store with quarantine,
   strict manifest validation, bounded extraction, deterministic chunks, immediate
   retrieval deletion, and audited lifecycle operations.
6. Keep personal corpus content read-only to Agent tools and require explicit
   retrieval selection on each eligible request.
7. Use bounded polling for truthful request states. No progress percentages, private
   reasoning, or fabricated ETA are exposed.
8. Render Markdown using DOM text nodes only, adding semantic GFM-style tables and
   explicit table-copy actions without accepting raw HTML.

## Acceptance evidence locations

Implementation and fixture evidence is recorded by tests and the certification bundle
under `outputs/agent_personal_corpus_gui_v6_certification_<UTC>/`. Live evidence is
kept separate from committed source and documentation. Sanitized fixture screenshots
and their manifest are stored in `docs/user_guide/assets/`.
