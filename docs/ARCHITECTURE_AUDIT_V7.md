# Architecture audit v7

Audit date: 2026-07-28

Verified starting revision:
`c0b71b22a1d44b3fb7dcf7071d085fb1af987b63`, the committed v6
certification revision. The production checkout was read-only, clean, and at
`0d9a1d54f445ad25a8bb84d3133c3377f446c476`. GPU and model-server state was
not probed because v7 explicitly forbids it.

## Product surfaces and entry points

| Surface | Entry point | GUI/API | Primary service/store | Identity and authorization | Configuration | Current duplication or constraint |
|---|---|---|---|---|---|---|
| Desktop/local project | `laplace`, `research-workspace`, `laplace_server:create_project_app` | Project HTML from `ui.py`; routes in `api.py` | `ChatEngine`, project `ConversationStore`, `Library`, project SQLite/files | OS user and selected project; no multi-user session | `PROJECT_CONFIG.yaml`, project YAML, legacy `RW_*` overrides | Project conversations and provider settings differ from Operator contracts |
| Server/multi-user | `laplace-operator-server`, `operator_api:create_operator_app` | Static Operator PWA and `/api/v1/*` | `OperatorService`, `TieredServingService`, conversations, personal corpus, artifacts | Registered email, session cookie, independent capabilities | CLI arguments, registered-user YAML, serving profiles | UI contains mode-specific panels and its own Markdown/progress components |
| Research plane | `research` CLI and `/api/v1/research/jobs*` | Operator Research panel | `DeepResearchService`, `ResearchAdmissionStore`, `ResearchStoreLayout` | `research` capability and admission controls | external state root and adapter settings | Job/source/claim records are versioned separately |
| Agent/worktree plane | Agent routes and Operator Agent panel | `/api/v1/agent/sessions*`, `/api/v1/worktrees*` | `AgentSandboxManager`, repository authorization, tiered serving | `agent` capability plus both repository grants | repository registry, grants and worktree quotas | Team-runner worktrees and user worktrees use related but distinct records |
| Local project corpus | `library-ingest`, `/projects/{name}/library/ingest` | Project Library | `Library`, `documents`, project SQLite and derived files | owning desktop user | project configuration | Uses “library/collection” while server uses “corpus” |
| Personal corpus | Knowledge GUI and `/api/v1/personal-corpora*` | Operator PWA | `PersonalCorpusStore` | owner plus `personal_corpus` capability | external private state and policy | Correctly isolated but not expressed through a common `CorpusStore` protocol |
| Shared governed corpus | reference CLI and `/api/v1/corpora` | Operator inventory | `governed_corpus`, `ReferenceLibrary`, `GovernedCorpusPromoter` | governed ingestion and Operator capabilities | manifests and registered source bundles | Retrieval support is route-specific |
| FormalScience library | project commands and project GUI | project API | `library.py`, `documents.py`, `retrieval.py` | project owner | `project.yaml` and project paths | Overlaps with local corpus terminology |
| Attachments | `/api/chat/.../attachments` | project chat | project `ConversationStore` | conversation owner | project storage | Not interchangeable with indexed corpora |
| Provider/runtime | `llm.py`, `model_routing.py` | project settings and Operator diagnostics | Ollama/OpenAI-compatible/vLLM classes, route caller | loopback endpoint policy | project settings, serving manifests/profiles | Provider payloads and capabilities lack one typed neutral contract |
| Model lifecycle | Operator model routes and deployment scripts | Operator-only controls | `ModelServerAdmission`, `ModelServerController` | `model_admin` and owned-process checks | serving profiles and process records | Must remain separate from provider invocation |
| Identity | activation/login/admin CLI and GUI | auth/admin routes | registered-user YAML, `SessionStore`, `UserCapabilityStore` | Argon2id, revision-bound sessions | registry plus external SQLite | YAML schema and capability SQLite have independent migrations |
| Repository grants | admin CLI/GUI and Agent | repository routes | `RepositoryAuthorizationStore` | `repository_admin`; user requires active dual grant | repository registry and SQLite | Canonical paths are correctly Operator-only |
| Audit/provenance | diagnostics, events, artifact views | multiple read APIs | JSONL audit, `ArtifactRegistry`, execution records | capability-scoped views | external state | Event envelopes and schema versions differ by subsystem |

## Persistent-store inventory

| Store | Current owner | Location class | Current version signal | Migration concern |
|---|---|---|---|---|
| Project metadata/conversations | desktop project | project-local `Data/` or configured SQLite | table existence; legacy records | must remain usable without server identity |
| Global project registry | desktop CLI | user-local registry | implicit | path portability and permission checks |
| Registered users | server auth | external YAML | schema 1/2 | retain v1 parser and atomic v2 output |
| Sessions | server auth | external SQLite | table shape | session revision and expiry must survive |
| Server conversations | Operator API | external SQLite | table shape | owner isolation and message compatibility |
| Repository authorization | server admin | external SQLite | table shape | canonical paths never enter ordinary exports |
| Agent worktrees | Agent plane | external SQLite plus Git worktrees | schema table | retained worktrees require restart recovery |
| Personal corpus | Knowledge plane | external SQLite plus owner-pseudonymous files | schema/chunking tables | sources, chunks and tombstones must move atomically |
| Artifact registry | shared planes | external SQLite plus content | table shape | immutable hashes and lineage continuity |
| Research/admission/jobs | Research plane | external SQLite plus job directories | record schema fields | interrupted stages must remain resumable |
| Audit/event logs | all planes | append-only external JSONL | event-local schema | migration itself needs an audit envelope |

No migration in v7 may target these production locations. Tests and certification use
fresh synthetic copies below temporary directories or the ignored certification
output root.

## Contract and lifecycle findings

- `llm.py` has useful local adapters but exposes backend-specific methods and does
  not provide one capability record for GUI routing. Model lifecycle is correctly
  separate and ownership-checked.
- `chat.py` and `conversations.py` both define conversation records for their
  respective operating modes. They require compatibility adapters, not immediate
  table unification.
- project `Library`, personal corpora, and the governed corpus share retrieval
  concepts but have different authorization and retention semantics.
- artifact content, execution records, research provenance, corpus events and
  authentication audit are trustworthy subsystem implementations with different
  envelopes.
- local configuration accepts a fixed legacy YAML shape, while server configuration
  is assembled from CLI/environment/state. Neither produces setting-level
  provenance.
- the Operator PWA already centralizes its own Markdown tables, citations, progress
  and capability navigation. The project GUI uses server-rendered components.
  Consolidation should begin with shared response contracts and behavior tests,
  retaining mode-specific pages.
- existing services support Linux and Windows paths, but state transfer,
  repository sync and diagnostics need explicit cross-platform path handling.

## Required compatibility boundaries

1. New protocols depend only on provider-neutral versioned models.
2. Existing project and server services remain callable through adapters.
3. Frontends receive provider, route and model IDs, never transport ports or
   canonical server paths.
4. Configuration is merged before service construction; no client reads deployment
   secrets.
5. Store migration uses preflight, backup, lock, transactional application and
   integrity verification against fixture state.
6. Desktop sync transfers a confirmed Git change set through a logical repository
   ID and a transport interface; it never stores credentials.
7. Governance schedules actions but all destructive purge remains explicit,
   scoped, dry-runnable and audited.

## Operating-system and release constraints

- Python 3.11 and 3.12 are supported on Ubuntu and Windows.
- POSIX permission enforcement is reported as unsupported where Windows cannot
  express it; path and ownership validation remains mandatory.
- subprocesses use argv arrays, explicit working directories and bounded timeouts.
- the release package excludes models, output archives, runtime state and secrets.
- core CPU/fixture tests require no external listener, provider, model download or
  GPU.
- every GPU/live-model gate is recorded exactly as
  `BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE`.

