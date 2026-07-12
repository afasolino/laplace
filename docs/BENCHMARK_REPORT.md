# Benchmark report

The measured backend is Ollama on `http://127.0.0.1:11434`, confirmed loopback-only by the service probe and `netstat`. Installed models are `qwen3:4b` (Q4_K_M, 2,497,293,931 bytes, 4.0B) and `qwen3-embedding:0.6b` (Q8_0, 639,150,858 bytes, 1024 dimensions).

The benchmark used one generation at a time, an 8192-token operational context, and a 4096-token safe context. Ollama streamed responses exposed prompt/evaluation counts, TTFT, generation duration, load duration, and total latency. `nvidia-smi` samples identified the GPU as `NVIDIA GeForce RTX 5060 Laptop GPU` (8151 MiB) with observed GPU utilization up to 100% and peak sampled VRAM of 3832 MiB (the exact per-run samples are in `outputs/model_benchmark.json`). Ollama/llama-server process working-set sampling recorded CPU RAM peaks where available.

Measured non-thinking generation runs:

| Workload | Prompt tokens | Generated tokens | TTFT (s) | Generation (s) | Output tok/s | Total (s) | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short writing | 31 | 160 | 0.945 | 2.343 | 68.30 | 3.272 | 3832 |
| Grounded writing, 4k context | 3300 | 160 | 6.252 | 2.707 | 59.11 | 8.958 | 3162 |
| Grounded writing, 8k context | 7049 | 160 | 7.766 | 3.166 | 50.53 | 10.918 | 3832 |
| Structured JSON extraction | 54 | 48 | 0.620 | 0.678 | 70.80 | 1.296 | 3832 |

The embedding run returned eight 1024-dimensional vectors in 1.911 s (4.186 texts/s). Thinking mode was also exercised; it produced 160 thinking tokens with 790 thinking-output characters and no visible answer under the bounded output budget, so non-thinking remains the default for routine rewriting.

The end-to-end run generated a one-page local PDF fixture, preserved SHA-256 and page/chunk provenance, embedded/retrieved one relevant chunk, and generated a grounded JSON answer. Citation validation passed for `grounded_fixture.pdf`, page 1, chunk `f512c9cfd29fe1f6de7b:p1:c0`. Retrieval latency was 1.555 ms in the final run.

No cloud endpoint, document upload, NPU driver, or vision model was used.

## Latest continuation rerun

The real benchmark was rerun after Ollama installation with the same exact models. It again verified `NVIDIA GeForce RTX 5060 Laptop GPU` through `nvidia-smi`, but an already-active Python GPU process was visible during the run. The observed values below are preserved exactly in `outputs/model_benchmark.json`; they are not normalized or substituted for the prior uncontended baseline.

| Workload | Prompt tokens | Generated tokens | TTFT (s) | Generation (s) | Output tok/s | Total (s) | Model load (s) | Peak VRAM (MiB) | CPU RAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short writing | 31 | 160 | 0.978 | 421.943 | 0.379 | 422.919 | 0.404 | 7795 | 1.09 GB |
| Grounded writing, 4k context | 3300 | 160 | 8.096 | 4.872 | 32.838 | 12.951 | 5.971 | 7747 | 0.81 GB |
| Grounded writing, 8k context | 7049 | 160 | 12.018 | 7.693 | 20.798 | 19.710 | 4.328 | 7745 | 2.55 GB |
| Structured JSON extraction | 54 | 48 | 0.542 | 2.225 | 21.575 | 2.765 | 0.234 | 7739 | 3.86 GB |

The continuation embedding run returned eight vectors in 3.823 s (2.093 texts/s). Thinking mode measured 29.315 output tokens/s with a 5.458 s generation duration; its visible output was bounded by the 160-token limit, so routine rewriting remains non-thinking. The latest run stayed below the physical 8 GiB device capacity but exceeded the configured 90% safety fraction while the concurrent process was present; do not treat its latency as an uncontended baseline. The previously verified uncontended peak was 3832 MiB.

The new user-facing `laplace` smoke is recorded in `outputs/laplace_smoke.json`: one local PDF, one page/chunk retained, grounded retrieval, valid exact citation after labeled extractive fallback, localhost health, start/stop, backup, cache confirmation, and an unapproved candidate queue.
