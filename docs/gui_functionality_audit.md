# GUI and API functionality audit

Audit date: 2026-07-27
Audited revision: `6fe6c18621cc20aa044d2690e965378ef143f3f1`
Audited implementation branch: `trial/pre-v4-20260727_115915`
Superseding specification: `prompts/laplace_codex_production_gui_auth_remote_provenance_v5.md`

## Executive finding

The starting interface is an authenticated diagnostic Operator Plane, not a
production multi-user workspace. Its strongest existing properties are semantic
HTML, a restrictive content-security policy, `textContent`-based rendering,
localhost defaults, backend role checks, CSRF checks on mutations, responsive CSS,
and fixture-backed API and Playwright tests.

The release-blocking finding is browser authentication: the GUI asks for a bearer
token, stores it in `sessionStorage`, and sends it in every request. Chat and agent
results are rendered as complete JSON envelopes. There is no registered-email
registry, opaque cookie session, activation flow, account screen, persistent
conversation store, per-user research ownership, artifact provenance registry,
production proxy configuration, or documented remote-access self-check.

## Existing action map

`Operator` in this table means the current `CapabilityTier.OPERATOR` check. The
current API roles are `read`, `operate`, `approve`, and `admin`.

| GUI action | Frontend handler | API operation | Current role/capability | CSRF | Backend service | Current success path | Current failure path | Existing coverage |
|---|---|---|---|---|---|---|---|---|
| Authenticate | `signIn`, `establishSession` | `POST /api/v1/session` | Any configured bearer principal | No | `OperatorAuth` | Token maps to an in-memory principal; a CSRF nonce is returned | Generic HTTP error shown in login dialog | API auth test; Chromium GUI login |
| Sign out | `signOut` | None | Any | No | Browser only | Deletes the token from `sessionStorage` | No server-side session is revoked | No logout API test |
| Refresh overview | `loadDashboard` | `GET /api/v1/dashboard` | Operator | No | `OperatorService` | Run, approval, research, and sanitized server summaries render | Warning banner | API and Chromium dashboard smoke |
| Send chat | `sendTierChat` | `POST /api/v1/chat` | Basic or Plus; Operator is currently not chat-enabled | Yes | `TieredServingService` | Tool-free model call is routed and audited | Error banner; composer state is not managed | Capability/API fixture tests |
| Select chat lane/domain | Form state in `sendTierChat` | Included in `POST /api/v1/chat` | Basic or Plus | Yes on send | `TieredServingService` | Requested/effective lane and route are in the JSON result | Domain/lane validation error | Routing and escalation unit tests |
| Bind agent session | `bindAgentSession` | `POST /api/v1/agent/sessions` | Plus | Yes | `AgentSandboxManager`, `RepositoryAuthorizationStore` | Repository ID resolves to a grant and isolated worktree | JSON error banner | Repository-isolation API/unit tests |
| Run agent request | `runAgent` | `POST /api/v1/agent/sessions/{id}/run` | Plus and owning user | Yes | `TieredServingService` | Bounded validated patch backend runs inside the binding | JSON error banner | API/unit isolation tests |
| Cancel/release agent | `cancelAgent` | `POST /api/v1/agent/sessions/{id}/cancel` | Plus and owning user | Yes | `AgentSandboxManager` | Clean worktree is released | JSON error banner | API/unit tests |
| Prepare run | `prepareRun` | `POST /api/v1/runs` | Operator plus current `operate`/`admin` role rules | Yes | `OperatorService` | Immutable request is hashed and displayed | Warning banner | API, service, and GUI tests |
| Inspect run | `inspectRun` | `GET /api/v1/runs/{id}` | Operator | No | `OperatorService` | Raw run JSON plus a readable gate matrix | Warning banner | Service/API tests; partial GUI smoke |
| Inspect artifact path | `inspectArtifact` | `GET /api/v1/artifacts?path=…` | Operator | No | API path guard over state/output roots | Preview and digest displayed | Warning banner | Traversal API test |
| Download artifact | inline click handler | `GET /api/v1/artifacts/download?path=…` | Operator | No | API path guard, `FileResponse` | Browser blob download | Warning banner | No browser download test |
| Create research job | `createResearch` | `POST /api/v1/research/jobs` | Operator | Yes | `DeepResearchService` | A local job is created | Warning banner | API and GUI fixture tests |
| Run research stages | `runResearch` | `POST /api/v1/research/jobs/{id}/run` then report GET | Operator | Yes on run | `DeepResearchService` | Completed stages, claims, and plain report text render | Warning banner | API and Chromium fixture tests |
| View corpora | `loadCorpora` | `GET /api/v1/corpora` | Operator | No | Governed corpus files | Snapshot cards render | Unhandled view-level rejection | No dedicated GUI test |
| Probe hardware/models | `loadHardware` | `GET /api/v1/model-servers/status` | Operator | No | `ModelServerManager` | Sanitized GPU and endpoint summaries render | In-panel safe failure | Partial API/GUI fixture coverage |
| Start/stop serving profile | `servingProfileAction` | `POST /api/v1/serving-profiles/action` | Operator and `admin` role | Yes | `ServingProfileOperator` | Exact owned profile lifecycle operation | Warning banner | Profile runtime unit tests, no GUI lifecycle test |
| Compare runs | `compareRuns` | `GET /api/v1/runs/compare/{left}/{right}` | Operator | No | `OperatorService` | A compact field comparison table renders | Warning banner | No dedicated GUI test |
| Refresh approvals | `loadApprovals` | `GET /api/v1/approvals` | Operator | No | `OperatorService` | Pending/current approvals render | Unhandled view-level rejection | API and GUI fixture tests |
| Approve/reject | dynamic button handler | `POST /api/v1/approvals/{id}/decision` | Operator with `approve` or `admin` role | Yes | `OperatorService` | Approval state refreshes | Promise rejection is not mapped locally | API and GUI fixture tests |
| View diagnostics | `loadDiagnostics` | `GET /api/v1/diagnostics` | Operator | No | API settings and adapters | Sanitized cards render | Unhandled view-level rejection | Static/API assertions only |
| Live event polling | `startEventStream` | `GET /api/v1/events?once=true` | Operator | No | `OperatorService.events` | Append-only events enter timeline | Warning and retry | API SSE test |

## Visible-control findings

- `Refresh`, `Sign out`, every navigation link, all forms, profile actions,
  approval decisions, and agent cancellation have handlers. There are no obvious
  dead buttons in the starting markup.
- Loading states are limited to a hardware placeholder. Other forms remain
  enabled during requests and have no duplicate-submit guard.
- Cancellation exists only for a bound agent session. Chat and research requests
  have no browser cancellation path.
- Retry and regenerate are absent. Errors are mostly flattened into a single
  banner with no category-specific recovery.
- Empty states exist for recent runs, research summary, approvals, corpora, and
  some evidence panels, but not consistently for every failed request.
- Chat, agent, run, and profile responses place full API objects into `<pre>`
  elements with `JSON.stringify`. This exposes implementation envelopes as the
  primary experience.
- Research reports render as escaped plain text, so they are safe but not
  readable Markdown. Citations, sources, conflicts, and uncertainties do not have
  dedicated presentation.
- The Plus repository control is a free-text repository-ID field. It does not
  accept a filesystem path, but it also does not list only the authenticated
  user's authorized repositories or explain revocation.
- Capability information is minimal. There is no role-aware Help/Capabilities or
  About/System Information page.
- The current CSS has desktop, tablet, mobile, reduced-motion, skip-link, visible
  focus, labels, and live-region support. Existing browser coverage checks only a
  small subset of accessibility requirements and does not establish WCAG 2.2 AA
  conformance.

## Security and privacy findings

- `sessionStorage["laplace_operator_token"]` is a browser-visible bearer
  credential and violates the production requirement.
- Authentication CSRF values are keyed to bearer-token digests in process memory.
  There is no opaque server-side browser session, expiry, revocation, rotation,
  password change, or logout-all operation.
- Origin validation is applied to state-changing authenticated operations, but
  there is no explicit allowed-Host middleware. CORS, proxy trust, and external
  URL validation are not modeled.
- The current CSP forbids third-party scripts, inline scripts, objects, framing,
  and foreign connections. `textContent` and DOM construction keep current output
  inert. There is no Markdown sanitizer because there is no Markdown renderer.
- The service worker caches only the shell and skips `/api/`, but production PWA
  mode is enabled by default and authenticated HTML can fall back to a cached
  shell. Remote production must default PWA off and explicitly clear old caches.
- API exceptions return structured categories, but several handlers include
  arbitrary `evidence` dictionaries. These require a centralized redaction and
  user/operator response split.
- Artifact authorization is path/role based rather than owner/repository/provenance
  based. A user-specific artifact registry does not exist.
- Research job GET/report routes are Operator-only but not owner-bound. There is
  no Basic/Plus research workspace isolation.
- The tier capability response returns user IDs and model IDs. It does not yet
  return authorized repositories, account state, or per-role help. Basic users do
  not receive agent controls after browser-side configuration, and agent endpoints
  are independently rejected by the backend.

## Persistence, queue, and model findings

- Conversation persistence, rename/archive/delete/reopen, server-side drafts, and
  retry lineage are absent.
- Tier request audits already capture capability, lane, route, limits, queue wait,
  trace ID, escalation, validators, and tool/network policy identifiers without
  prompt or response bodies.
- The admission scheduler already reserves quality capacity and exposes a queue
  snapshot. The GUI reduces this to a waiting count and does not show queue
  position, estimated wait, stage, quota reason, or cancellation.
- The selected profile configuration points quality and standard to the measured
  P1 route on loopback port 8201 and restricts CodeV to SystemVerilog. Historical
  compatibility routes remain 8102/8103. All identifiers need to come from
  reviewed configuration rather than frontend literals.

## API surface not clearly represented in the GUI

- `POST /api/v1/runs/{id}/start`
- `POST /api/v1/approvals`
- `POST /api/v1/model-servers/action`
- `GET /api/v1/research/jobs/{id}`
- `GET /api/v1/agent/sessions/{id}/status`
- `POST /api/v1/admin/tier/users`
- `POST /api/v1/admin/repositories`
- `POST /api/v1/admin/repository-grants`
- `POST /api/v1/admin/repository-grants/revoke`
- Complete streaming semantics for events and model output

The API also lacks the v5 account, activation, login/logout, session management,
conversation, authorized-repository listing, provenance, readiness, version,
research cancellation/ownership, registry reload, and user/session administration
operations.

## Required implementation direction

1. Keep explicit bearer authentication only as an opt-in non-browser API adapter.
   Add a strict external registered-user registry, Argon2id activation/password
   workflow, hashed opaque session store, session-bound CSRF, expiry, rotation,
   rate limits, revocation, and generic login failures.
2. Add owner-bound conversations, research records, and provenance records in
   SQLite WAL stores under the external state root.
3. Make the product shell authenticated by cookie, capability-driven by the
   server, and readable by default. Keep raw JSON in an Operator-only disclosure.
4. Render a strict Markdown subset through one audited DOM builder, never through
   `innerHTML`; reject unsafe schemes and mark external links.
5. Add explicit deployment modes, Host/Origin/proxy validation, secure cookies,
   readiness/version endpoints, reverse-proxy fixtures, and remote-access checks.
6. Replace free-text repository selection with server-authorized choices and
   present bindings, diffs, verification, and denials without exposing host paths.
7. Disable PWA by default for production and make the service worker a
   network-only, cache-clearing implementation if explicitly enabled.
8. Expand self-checking API, security, isolation, browser, responsive,
   keyboard-only, accessibility, failure, remote-proxy, load, and shutdown tests.
