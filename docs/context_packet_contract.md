# Context packet contract

Every model call receives a native role-specific packet:

```text
context/<role>/<attempt>/context.md
context/<role>/<attempt>/context_manifest.json
context/<role>/<attempt>/context.sha256
```

`ContextPacketBuilder` normalizes UTF-8 and line endings, sorts inputs, records
source hashes, and binds the skill and corpus snapshot hashes. The packet is
bounded to 256 KiB. Held-out paths, secret-like paths, and secret-like content
are rejected. Prompt compaction happens before the context packet is attached.

The manifest records paths and hashes, not hidden evaluator content. A source
change produces a different packet hash. Rebuilding unchanged inputs produces
the same bytes and hash. Repomix is not a runtime dependency; native packet
generation is mandatory and default.

