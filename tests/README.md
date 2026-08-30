# Test ownership index

Tests retain their historical campaign names where that provenance matters. Use
this index to locate the current behavioral owners without treating a campaign
suffix as the component boundary.

| Source component | Primary deterministic tests | Deferred or environment-specific coverage |
| --- | --- | --- |
| `laplace_core.py` | `test_laplace_core_g1.py`, `test_laplace.py` | service integration markers in the same modules |
| `operator/`, `operator_api.py`, `operator_server.py` | `test_operator_api.py`, `test_operator_agent_conversation.py`, `test_auth_provenance_security.py`, `test_tiered_serving.py` | browser/interactive operator tests |
| `agent_sandbox.py`, `agent_infrastructure/` | `test_candidate_assurance_worktree.py`, `test_mutation_authority_v32.py` | POSIX lifecycle/security cases |
| `zetsu_agent.py`, `zetsu_checkpoint.py`, `zetsu_handoff.py`, `zetsu_state.py` | `test_zetsu_agent_checkpoint.py`, `test_zetsu_agent_objective_isolation_v31.py`, `test_zetsu_hotfix_lifecycle.py` | POSIX process and live-runtime cases |
| document/retrieval/engineering contracts | `test_engineering.py`, `test_research_web_adapters.py`, `test_laplace.py` | optional dependency cases remain marked |
| routing/serving/experiment seams | `test_rtl_worker_routing_patch.py`, `test_paired_benchmark.py`, `test_dual_model_ablation.py` | GPU and A6000 cases are never selected by the deterministic marker |
| CI taxonomy and architecture rules | `test_non_a6000_certification.py`, `test_import_boundaries.py` | remote CI validates the full OS matrix and coverage gate |

Run the repository certification runner for the authoritative local subset:

```powershell
$env:PYTHONPATH = 'src'
python scripts/run_non_a6000_certification.py --output-root .runtime/v3-non-a6000/<run-name>
```

Do not infer that deselected tests passed: their marker category is recorded as
`DEFERRED` in the resulting classification.
