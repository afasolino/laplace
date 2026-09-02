# Qwen3.8 P8 production configuration

Laplace has one active Qwen3.8 quality/standard production profile.

- profile: `P8_qwen38_w4a16_mtp`
- served model: `laplace-quality-qwen38-mtp8`
- MTP speculative depth: 8
- context limit: 131072
- active profile: `configs/serving_profiles/P8_qwen38_w4a16_mtp.json`
- artifact manifest: `configs/model_manifests/qwen38_27b_a6000.json`
- frozen Step-B commit:
  `54fa762ff9bf7273c320e04183fdd69db391688b`
- frozen Step-B P8 blob:
  `cffae38fab1196d339fe9a7dd23a04616219e6d2`

Operational model paths are repository-relative. Moving the checkout does not
require editing the production configuration.

The P8 production gate owns the quality and standard lanes only. The
economy/RTL specialist route is independently versioned and may evolve without
changing P8 identity.

There is no active Qwen3.6 rollback, P6/P7 promotion path, floating
prequantized-candidate selection, or local-AWQ fallback.

`configs/model_manifests/qwen38_prequantized_source.json` is retained because
its SHA-256 is part of the immutable packed-artifact provenance. It is not an
operational model-selection policy.

If the packed artifact is missing or does not match its recorded bytes,
Laplace fails closed. A replacement model must never be selected or quantized
automatically.
