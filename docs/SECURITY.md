# Security model

The service binds to loopback, rejects unsupported extensions, normalizes upload names, caps uploads at 50 MiB, and never executes document or model text. Source hashes identify immutable copies. Exports must redact local paths when shared. No cloud inference or automatic acquisition is implemented.

