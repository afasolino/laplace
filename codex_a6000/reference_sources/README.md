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

Precedence is always:

```text
target repository + task specification
→ target-project conventions
→ private curated references
→ curated open-source references
→ model prior knowledge
```
