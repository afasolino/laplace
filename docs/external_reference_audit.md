# External reference audit

Audit date: 2026-07-27. Revisions were resolved with `git ls-remote`. No
reference framework was installed, vendored, imported at runtime, or copied
wholesale.

| Repository | Audited revision | Licence finding | Use |
|---|---|---|---|
| anthropics/skills | `b29e7cf65e5cb78a5ac33d582270551bc74a14eb` | mixed: many Apache-2.0 examples; document skills are source-available | Agent Skills directory/front-matter concept only; native skills authored |
| obra/superpowers | `3dcbd5c4b48e02263fbf4a3c01e3fe4f81d584d9` | MIT | reviewed; not adopted |
| gsd-build/get-shit-done | `bdcaab2c752d9a33a1a1ca9acf3a3c81fb991815` | MIT; upstream archived | reviewed; not adopted |
| yamadashy/repomix | `b921db74351fd9185897a641a8f39c0b82de7fcb` | MIT | optional packet-backend concept only; no dependency |
| open-telemetry/opentelemetry-python | `4ea521a73c0f7ad67b7c04dff1e1dc6d00324a69` | Apache-2.0 | JSONL field compatibility and no-op concept; native recorder |
| langchain-ai/langgraph | `30c4d58db86455128e42ddec96b1ba53c553ba22` | MIT | reviewed; runner not replaced |
| temporalio/sdk-python | `60e3b73474eea0e7d69c96a68f041a77bcc6f14d` | MIT | reviewed; runner not replaced |
| pewdiepie-archdaemon/odysseus | `d8a2059df8e53bc7275c45339849d14c8651e73c` | AGPL-3.0 | ideas only; no code copied or dependency |
| openclaw/openclaw | `079ac9390d842bd49e0b59cd75fa9707b4e1a893` | MIT | reviewed; no personal assistant/memory runtime |
| NousResearch/hermes-agent | `d71033a4077a6dfdcdb42c9e9eeab4c41e4a7012` | MIT | selective provider/safety/research patterns; native implementation |
| tinyhumansai/openhuman | `fd04d37d34aa0da1a6cbb82bbb6d256dcf003227` | GPL-3.0 | reviewed; no memory/orchestrator adoption |
| thedotmack/claude-mem | `132b46343e60ecf4057c427736c57b08f7615dfe` | Apache-2.0 | reviewed; mutable memory explicitly excluded |

The Odysseus sparse inspection was limited to `services/hwfit`,
`routes/hwfit_routes.py`, diagnostic/model-discovery routes,
`src/task_scheduler.py`, `src/event_bus.py`, and `src/mcp_manager.py`. Native
Laplace code uses only the general separation of hardware facts, fit decision,
endpoint diagnostics, and ownership-aware lifecycle.

The Hermes sparse inspection was limited to delegation lifecycle, web provider
registry, browser safety, and the arXiv research skill. Native Laplace code
retains provider isolation, bounded fetches, citation verification, and exact
repository revisions without importing Hermes.

No source from Odysseus, Hermes, or another reference is present in the
certification implementation. The licence column is an audit finding, not a
claim that every subdirectory has identical terms; Anthropic explicitly uses
mixed terms.

