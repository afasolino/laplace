# Architecture

Sources remain immutable under `data/documents/<sha256>/`; parsed page records, SQLite metadata, and derived outputs are separate. The deterministic path is ingest → provenance-bearing chunks → local hashed embeddings/lexical search → evidence packet → model interpretation. Numerical analysis happens in Python. The model provider is localhost-only and concurrency is one.

