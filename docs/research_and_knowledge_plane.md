# Research and Knowledge Plane

`DeepResearchService` runs the explicit 13-stage workflow from normalized
question through report generation. Each stage is resumable and represented
in `events.jsonl`.

Every job contains:

```text
job.json                  events.jsonl
search_queries.jsonl      sources/
source_manifest.json      claims.jsonl
contradictions.json       evidence_ledger.json
report.md                 report.html
research.lock.json
```

Adapters are pluggable for governed local documents, uploaded local documents,
self-hosted SearXNG, arXiv, Crossref, Semantic Scholar, explicit URLs, and
exact-revision GitHub inspection. Web adapters enforce bounded HTTPS fetches,
robots policy, rate limits, timeouts, and size limits. SearXNG is optional and
must be loopback-hosted.

Sources are deduplicated by canonical URL and content hash. Claims name
supporting and contradicting source IDs. Every report citation must resolve to
the evidence ledger. Reports distinguish grounded fact, disagreement,
inference, open question, and recommendation.

Stores are physically separated as `governed_corpus`,
`exploratory_research_store`, and `personal_workspace_store`. Research never
promotes itself. `research promote` requires an approved operator action,
licence/permitted-use metadata, retrieval snapshot and hash, topic metadata, a
human-curated or independently authored summary, and passing retrieval
relevance tests. It creates a new immutable governed snapshot and leaves the
old snapshot readable.

