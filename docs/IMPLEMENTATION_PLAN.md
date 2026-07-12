# Executable implementation plan

Each phase is gated by its listed automated checks and status artifact. Recoverable failures are fixed before advancing; unavailable external hardware, runtimes, models, or licensed software are recorded with the exact probe evidence.

1. **Foundation:** scaffold the typed Python package, validated configuration, JSON logs, CLI, deterministic manifests, bootstrap scripts, and architecture/security documentation. Verify a clean virtual-environment install, tests, CLI help, lint, and types.
2. **Discovery:** implement tolerant OS/CPU/RAM/disk/GPU/runtime/NPU probes with raw command evidence. Test absent and malformed tools, then write `outputs/system_probe.json` and its report on this host.
3. **Model runtime:** implement localhost-only Ollama and OpenAI-compatible providers plus a deterministic mock; benchmark installed, explicitly configured models at concurrency one and 8k/4k contexts. Record measured selection or `MODEL_REQUIRED` without downloading licensed weights.
4. **Documents:** implement immutable, hash-addressed ingestion, parser fallbacks, page-preserving chunks, quarantine, metadata updates, deletion, and rebuild. Validate normal/duplicate/malformed/scanned-style fixtures and provenance coverage.
5. **Retrieval:** implement SQLite persistence, local deterministic embeddings, lexical/semantic/hybrid search, filters, evidence packets, grounded-answer rules, and citation validation. Benchmark a labelled mini-corpus.
6. **Extraction:** implement strict provenance-bearing scientific schemas, deterministic unit validation, JSON/CSV batch exports, comparison tables, and a labelled-fixture benchmark.
7. **Analysis:** implement deterministic parsers for Python/build/Git/Vivado/Cadence/CUDA and CSV/JSON results, earliest-error selection, metrics, and structured run comparisons.
8. **Drafting:** implement claim maps, citation checking, unsupported-claim marking, notation/number preservation, consistency checks, and local evidence exports.
9. **API/UI:** implement localhost FastAPI endpoints and a compact offline HTML interface with validated uploads; run API, binding, path, MIME, and smoke tests.
10. **Optional NPU:** probe official/local provider availability after baseline completion; implement a bounded optional embedding provider with correctness fallback and record measurements or a precise blocker.
11. **Optional vision:** implement on-demand PDF page rendering/inspection hooks with explicit uncertainty and model-residency safeguards; keep native extraction primary.
12. **Packaging:** run the full offline test/lint/type/security and benchmark suite, write user/install/status/benchmark/limitations guides, lifecycle/backup scripts, and the final manifest with exact reproduction commands.

