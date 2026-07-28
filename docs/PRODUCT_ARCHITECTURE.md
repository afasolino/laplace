# Product architecture

Laplace is one local-first product with two operating modes. Both modes use the same
versioned service contracts and provider vocabulary; they do not share authorization
or silently share state.

## Standard terminology

| Term | Meaning |
|---|---|
| project | A desktop user's FormalScience working root and its configuration |
| workspace | The logical UI/session context containing a project or server account |
| repository | A registered Git top-level identified to clients by logical ID |
| worktree | One isolated Git checkout owned by an Agent task |
| collection | A named grouping inside a desktop FormalScience library |
| corpus | An indexed retrieval boundary: project, personal or shared |
| library | A curated set of collections, currently the desktop FormalScience library |
| attachment | Content scoped to a conversation, not automatically a corpus source |
| artifact | Immutable generated/exported content with owner, hash and provenance |
| source | Original authorized content from which text/evidence was derived |
| run | A reproducible execution envelope containing stages and artifacts |
| job | A persisted unit of queued or resumable work |
| session | An authenticated browser session or an explicitly named Agent session |
| conversation | An owner-scoped ordered sequence of messages |
| provider | A local inference transport adapter, never a frontend port |
| model | A provider-advertised model ID |
| route | A provider/model selection with limits and policy |
| lane | A scheduling class such as quality, standard, economy or CodeV |

## Dependency direction

```text
mode-specific GUI/CLI
        |
mode-specific HTTP adapter
        |
provider-neutral contracts + service protocols
        |
compatibility adapters / domain services
        |
SQLite, files, Git and configured local providers
```

The GUI never imports a runtime client. Provider payloads are contained in
`providers.py`; both APIs expose only safe route/provider summaries. Canonical paths
are held by Operator-only service records and represented by logical IDs elsewhere.

Provider operations are asynchronous. SQLite/file/Git store methods are synchronous
transaction boundaries and HTTP adapters dispatch them through bounded worker
execution where required.

## Versioned contracts

`contracts.py` defines strict schema-v1 records for conversations, messages,
attachments, corpora, corpus sources, retrieval snapshots, artifacts, provenance,
repository grants, worktrees, jobs, providers, routes, capabilities and audit events.
Unknown fields fail validation. Persisted legacy records continue through explicit
adapters and migrations rather than being rewritten on read.

`architecture.py` defines `ModelProvider`, `EmbeddingProvider`,
`ConversationStore`, `CorpusStore`, `RetrievalService`, `ArtifactStore`,
`ProvenanceStore`, `RepositoryService`, `WorktreeService`, `IdentityService`,
`CapabilityService`, `JobService`, `AuditService` and `ConfigurationService`.
`fixture_services.py` supplies deterministic implementations of every service
boundary for tests, evaluation and degraded-mode certification.

## State ownership

Desktop projects own their project configuration, sources, derived indexes,
conversations and outputs. Server users own conversations, personal sources and
artifacts; shared corpora and registered repositories remain governed resources.
Agent worktrees are retained external execution state. Attachments, corpora and
repositories never become interchangeable merely because they contain similar files.

Lifecycle ownership is separate from invocation. An Ollama or vLLM provider invokes a
configured endpoint but cannot download, start or stop a model. Only the existing
model lifecycle service may control a process after matching its ownership record.
