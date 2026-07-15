# Curated reference catalogs

The catalogs in this directory are candidate sources for Laplace's read-only reference subsystem.

Laplace must create two governed logical libraries:

```text
<FORMALSCIENCE_ROOT>/Library/Python/
  00_policies/
  10_language_stdlib/
  20_architecture_patterns/
  30_web_api/
  40_testing/
  50_typing_validation/
  60_tooling_packaging/
  70_agents_mcp/
  80_security_performance/
  90_manifests/

<FORMALSCIENCE_ROOT>/Library/SystemVerilog/
  00_policies/
  10_rtl_patterns/
  20_interfaces/
  30_verification/
  40_tooling/
  50_vendor_flows/
  90_manifests/
```

Before synchronizing any source, Codex must request approval for network access, resolve an exact commit SHA, inspect the licence and notices at that commit, hash the selected files, and record permitted use. Selected reference files remain immutable and read-only. Entire monorepositories must not be indexed by default.

The experiment's external C/Verilog acquisition is described by
`codex_a6000/governed_corpus/external_acquisition_plan.json`. With approved
network access, acquire only those pinned files with:

```bash
.venv/bin/python scripts/acquire_multilanguage_governed_references.py --repository-root .
```

The six selected snapshots are installed under
`codex_a6000/governed_corpus/external/`; every selected file is mode `0444`
and `installed_external_references.json` records its exact resolved commit,
licence, source origin and SHA-256. Re-running acquisition verifies and reuses
identical content. Production validation still fails closed if any snapshot is
missing or changed. Candidate catalogue entries are never treated as installed
content.

Precedence is always:

```text
target repository + task specification
→ target-project conventions
→ private curated references
→ curated open-source references
→ model prior knowledge
```
