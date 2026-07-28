# Frontend consolidation

Laplace retains the project GUI and the Operator PWA because they serve distinct
operating modes. Consolidation occurs at the presentation-contract and behavior
level, not by introducing another frontend.

`presentation.py` defines shared schema-v1 records for:

- safe Markdown with raw HTML disabled;
- file/page/section/chunk citations;
- truthful request states without percentage or private reasoning;
- stop/retry/edit/export controls;
- storage explanations;
- capability-aware navigation.

The Operator PWA remains the reference implementation for safe Markdown tables,
horizontal scrolling, TSV/Markdown copy, citations and bounded polling. Project
server-rendered views consume the same terms and response semantics incrementally.
Mode-specific panels—server user administration and desktop project settings—remain
separate.

Frontend code receives provider and route IDs plus public capability records. It
must not contain Ollama/vLLM payloads, endpoints, canonical paths, secrets or process
controls. A degraded provider disables only routes that depend on that provider and
shows a sanitized reason.
