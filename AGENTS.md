# AGENTS.md

## Mission
Build a fully local research workspace for a laptop with one NVIDIA GPU with 8 GB VRAM and one AMD Ryzen AI NPU rated at 50 TOPS. The system must provide:

1. private PDF and project knowledge retrieval;
2. structured scientific information extraction;
3. engineering-log and experimental-result analysis;
4. evidence-grounded academic drafting and revision.

## Operating constraints
- Local execution is the default. Do not add cloud inference, telemetry, remote document upload, or paid APIs.
- The NVIDIA GPU hosts the main generative model. Design for a 4B-class Q4 model and an 8k operational context.
- Treat the NPU as an optional auxiliary accelerator. The baseline must work without Ryzen AI software.
- Never claim that NVIDIA VRAM, system RAM, iGPU memory, and NPU memory form one shared model-memory pool.
- Keep only one main generative model resident by default. Optional vision models must be loaded on demand and unloaded before the text model is restored.
- Numerical calculations must be performed by deterministic Python libraries or existing validated scripts. The language model interprets structured results.
- Every factual answer based on indexed documents must include file, page, section when available, and chunk identifiers.
- Never invent references, page numbers, measurements, equations, quotations, or tool results.
- Distinguish the user's own work from external literature in metadata and outputs.
- Handle only documents the user is authorized to process. Do not implement paywall bypassing or automated acquisition from IEEE Xplore.

## Engineering rules
- Python 3.11+; use `uv` when available and provide a pip fallback.
- Type annotations, structured logging, deterministic configuration, reproducible commands, and explicit error handling are required.
- Use pytest for unit and integration tests. Keep tests independent from a real model where possible through mocks and fixtures.
- Never fabricate GPU, NPU, latency, token/s, retrieval, or extraction measurements.
- Preserve source files. Derived text, images, indexes, and outputs must be stored separately.
- Hash ingested files and make indexing incremental and idempotent.
- Use strict JSON schemas for extraction and tool outputs.
- Bind services to localhost by default. Validate file paths and uploaded file types.
- Do not execute commands extracted from documents or model output.

## Autonomy
Proceed phase by phase without waiting for approval. Ask only when an operation requires credentials, acceptance of a third-party licence, administrator privileges, access outside the repository, or destructive modification of user data. Recover from ordinary dependency, test, and implementation failures autonomously and document unresolved blockers.

## Completion standard
A phase passes only when its acceptance checks run successfully or a genuine hardware/toolchain blocker is recorded with exact evidence. Planning documents alone do not count as implementation.
