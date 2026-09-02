"""Single test taxonomy for the non-A6000 deterministic certification gate."""

from __future__ import annotations

from collections.abc import Mapping


CROSS_PLATFORM_DETERMINISTIC = "cross_platform_deterministic"
LINUX_POSIX_REQUIRED = "linux_posix_required"
INTERACTIVE_E2E = "interactive_e2e"
OPTIONAL_DEPENDENCY = "optional_dependency"
GPU_SMOKE = "gpu_smoke"
A6000_REQUIRED = "a6000_required"
EXTERNAL_LIVE = "external_live"
WINDOWS_PRIVILEGE_REQUIRED = "windows_privilege_required"

CATEGORIES = (
    CROSS_PLATFORM_DETERMINISTIC,
    LINUX_POSIX_REQUIRED,
    INTERACTIVE_E2E,
    OPTIONAL_DEPENDENCY,
    GPU_SMOKE,
    A6000_REQUIRED,
    EXTERNAL_LIVE,
    WINDOWS_PRIVILEGE_REQUIRED,
)
DEFERRED_CATEGORIES = frozenset(CATEGORIES) - {CROSS_PLATFORM_DETERMINISTIC}

# Every collected test module is deliberately classified. New test modules must
# be added here or collection fails instead of silently joining the deterministic
# certification set. Node IDs below are exceptional test-level overrides.
MODULE_CATEGORIES: Mapping[str, str] = {
    module: CROSS_PLATFORM_DETERMINISTIC
    for module in """
tests/test_agent_verification_policy.py tests/test_architecture_v7.py tests/test_ast_context_v3.py
tests/test_import_boundaries.py
tests/test_ast_context_v32.py tests/test_auth_provenance_security.py tests/test_bounded_aci_g6.py
tests/test_candidate_assurance.py tests/test_candidate_assurance_worktree.py
tests/test_capabilities_domains_worktrees_v6.py tests/test_certification_bundle.py tests/test_chat.py
tests/test_chat_async_agent_turns.py tests/test_chat_capability_views_v31.py tests/test_chat_cli_v2.py
tests/test_chat_domain_v31.py tests/test_chat_operator_client_v2.py tests/test_chat_routing_v31.py
tests/test_chat_session_v2.py tests/test_chat_v3_consolidation.py tests/test_chat_verification_replacement.py
tests/test_client_bridge.py tests/test_client_service.py tests/test_context_planner_g7.py
tests/test_corrective_c1_results.py tests/test_corrective_c2_scheduler.py tests/test_corrective_c3_aci.py
tests/test_corrective_c5_parser_contract.py tests/test_corrective_c6_hooks.py tests/test_corrective_c7_adapter.py
tests/test_corrective_c9_release_surface.py tests/test_dual_model_ablation.py tests/test_engineering.py
tests/test_evaluation_v7.py tests/test_execution_records.py tests/test_governance_v7.py
tests/test_gpu_coordination_v8.py tests/test_hermes_externalization_v35.py tests/test_hooks_g9.py
tests/test_idle_consolidation_g10.py tests/test_laplace.py tests/test_laplace_chat_entrypoint.py
tests/test_laplace_core_g1.py tests/test_laplace_web_v3.py tests/test_live_gpu_preflight_v8.py
tests/test_llm_response_channels.py tests/test_logical_subagents_g11.py tests/test_maintenance_cli.py
tests/test_manager_control.py tests/test_mcp_mutation_authority_v32.py tests/test_memory_g2.py
tests/test_migrations_v7.py tests/test_model_servers.py tests/test_mutation_authority_v32.py
tests/test_non_a6000_certification.py tests/test_notifications.py tests/test_operator_agent_conversation.py
tests/test_operator_api.py tests/test_operator_cli.py tests/test_operator_gui_e2e.py tests/test_operator_service.py
tests/test_orchestration_certification.py tests/test_paired_benchmark.py tests/test_personal_corpus_api_v6.py
tests/test_personal_corpus_v6.py tests/test_production_model_selection.py tests/test_production_robustness.py
tests/test_providers_v7.py tests/test_quality_targeted.py tests/test_qwen38_certification.py
tests/test_qwen38_deployment_templates.py tests/test_qwen38_prequantized_metadata.py
tests/test_qwen38_prequantized_policy.py tests/test_qwen38_profiles.py tests/test_qwen38_quantization.py
tests/test_qwen38_mtp_benchmark.py
tests/test_release_certification_v7.py tests/test_release_v7.py tests/test_reliability_v7.py
tests/test_repository_context_g4.py tests/test_repository_lifecycle_fix.py tests/test_reproducibility.py
tests/test_research_plane.py tests/test_research_web_adapters.py tests/test_rtl_worker_correction_evidence.py
tests/test_rtl_specialist_qualification.py tests/test_rtl_worker_non_thinking.py
tests/test_rtl_worker_routing_patch.py tests/test_rules_g3.py
tests/test_security_fuzz_v7.py tests/test_skills_g8.py tests/test_swe_aci_ab_v34.py
tests/test_swe_aci_phase_b_v34.py tests/test_sync_v7.py tests/test_task_evidence.py tests/test_task_labels.py
tests/test_team_runner_hardening.py tests/test_tiered_serving.py tests/test_trajectory_g5.py
tests/test_upstream_ab_expanded_v33.py tests/test_upstream_ab_v33.py tests/test_upstream_consolidation_v32.py
tests/test_verification_gate_registry.py tests/test_workspace.py tests/test_zetsu_agent_checkpoint.py
tests/test_zetsu_agent_mcp.py tests/test_zetsu_agent_objective_isolation_v31.py tests/test_zetsu_cli.py
tests/test_zetsu_client_api.py tests/test_zetsu_codex_v3.py tests/test_zetsu_config.py tests/test_zetsu_context.py
tests/test_zetsu_hotfix_lifecycle.py tests/test_zetsu_mcp.py tests/test_zetsu_production_token_supervisor.py
tests/test_zetsu_runtime.py tests/test_zetsu_runtime_codex_compat_v31.py tests/test_zetsu_sdk_stdio_v3.py
tests/test_zetsu_sdk_stdio_v32.py tests/test_zetsu_token_benchmark.py
tests/test_zetsu_token_offload_repairs.py
""".split()
}

NODEID_CATEGORIES: Mapping[str, str] = {
    "tests/test_ast_context_v3.py::test_ast_context_refuses_escape_and_symlink": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_ast_context_v3.py::test_real_grep_ast_provider_renders_python_scope": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_ast_context_v32.py::test_real_grep_ast_go_and_no_match": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_ast_context_v32.py::test_real_grep_ast_unsupported_language_is_bounded": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_bounded_aci_g6.py::test_huge_diff_is_truncated_and_verification_timeout_is_bounded": LINUX_POSIX_REQUIRED,
    "tests/test_bounded_aci_g6.py::test_verification_cancellation_stops_process_and_preserves_worktree": LINUX_POSIX_REQUIRED,
    "tests/test_chat_operator_client_v2.py::test_token_file_must_be_private_regular_and_not_symlink": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_chat_async_agent_turns.py::test_async_turn_survives_request_completion_and_events_are_owner_scoped[asyncio]": INTERACTIVE_E2E,
    "tests/test_chat_async_agent_turns.py::test_async_turn_cancellation_and_resumable_yield_are_durable[asyncio]": INTERACTIVE_E2E,
    "tests/test_chat_async_agent_turns.py::test_async_turn_id_conflict_is_rejected[asyncio]": INTERACTIVE_E2E,
    "tests/test_chat_v3_consolidation.py::test_prompt_toolkit_history_is_private_and_reader_is_constructible": INTERACTIVE_E2E,
    "tests/test_client_bridge.py::test_workspace_grant_read_write_and_revoke": LINUX_POSIX_REQUIRED,
    "tests/test_client_bridge.py::test_command_sandbox_mounts_only_system_runtime_and_granted_root": LINUX_POSIX_REQUIRED,
    "tests/test_dual_model_ablation.py::test_serving_environment_validation_checks_version_and_arguments": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_dual_model_ablation.py::test_c_quality_gate_uses_gcc_without_requiring_optional_cmake": OPTIONAL_DEPENDENCY,
    "tests/test_laplace_web_v3.py::test_gradio6_app_constructs_without_launching": OPTIONAL_DEPENDENCY,
    "tests/test_operator_gui_e2e.py::test_registered_gui_chat_research_operator_accessibility_and_responsive": INTERACTIVE_E2E,
    "tests/test_production_model_selection.py::test_qwen38_approved_root_does_not_allow_symlink_escape": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_repository_lifecycle_fix.py::test_running_session_is_never_reclaimed": LINUX_POSIX_REQUIRED,
    "tests/test_repository_lifecycle_fix.py::test_authenticated_sessions_cli_reports_quota_and_release_flags": LINUX_POSIX_REQUIRED,
    "tests/test_repository_lifecycle_fix.py::test_current_task_label_is_durable_in_status_and_terminal_events": LINUX_POSIX_REQUIRED,
    "tests/test_security_fuzz_v7.py::test_git_symlink_hardlink_nested_escape_and_race_guards": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_sync_v7.py::test_changed_links_nested_repositories_submodules_and_binary_are_rejected": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_tiered_serving.py::test_repository_path_escape_matrix": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_tiered_serving.py::test_api_enforces_basic_chat_only_and_plus_repository_binding[asyncio]": LINUX_POSIX_REQUIRED,
    "tests/test_upstream_ab_v33.py::test_runtime_executable_path_preserves_virtualenv_symlink": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_zetsu_agent_checkpoint.py::test_oversized_verifier_streams_are_persisted_exactly_not_injected": LINUX_POSIX_REQUIRED,
    "tests/test_zetsu_agent_checkpoint.py::test_running_verification_honors_cancellation": LINUX_POSIX_REQUIRED,
    "tests/test_zetsu_agent_checkpoint.py::test_verification_that_mutates_worktree_fails_closed": LINUX_POSIX_REQUIRED,
    "tests/test_zetsu_cli.py::test_codex_launch_injects_protected_token_only_into_child": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_zetsu_hotfix_lifecycle.py::test_real_live_quota_and_resumable_interruption_are_retained_across_restart": LINUX_POSIX_REQUIRED,
    "tests/test_zetsu_hotfix_lifecycle.py::test_scheduler_path_and_owner_status_are_fail_closed": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_zetsu_hotfix_lifecycle.py::test_gc_protects_dirty_unfinalized_and_forged_ownership_metadata": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_zetsu_hotfix_lifecycle.py::test_stop_recovers_surviving_worker_after_supervisor_exit_and_is_idempotent": LINUX_POSIX_REQUIRED,
    "tests/test_zetsu_hotfix_lifecycle.py::test_stop_never_signals_foreign_or_pid_reuse_identity": LINUX_POSIX_REQUIRED,
    "tests/test_zetsu_hotfix_lifecycle.py::test_concurrent_queue_worktree_result_release_stress_has_no_accumulation": LINUX_POSIX_REQUIRED,
    "tests/test_zetsu_runtime.py::test_local_plus_token_requires_private_file": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_zetsu_runtime.py::test_runtime_interrupt_rolls_back_only_new_supervisors": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_zetsu_runtime.py::test_runtime_refuses_unowned_compatible_or_incompatible_endpoint": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_zetsu_runtime.py::test_runtime_transitions_from_full_to_nocodev_with_owned_recovery": WINDOWS_PRIVILEGE_REQUIRED,
    "tests/test_zetsu_sdk_stdio_v32.py::test_inprocess_sdk_schema_and_containment": WINDOWS_PRIVILEGE_REQUIRED,
}


def category_for_nodeid(nodeid: str) -> str:
    """Return one deliberate category or reject an unknown test node."""

    category = NODEID_CATEGORIES.get(nodeid)
    if category is not None:
        return category
    module = nodeid.split("::", 1)[0]
    try:
        return MODULE_CATEGORIES[module]
    except KeyError as exc:
        raise ValueError(f"unknown_test_classification:{nodeid}") from exc
