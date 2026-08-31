"""Guard the intended transport and orchestration import direction."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "research_workspace"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    from_imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    direct_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    return from_imports | direct_imports


def _private_from_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name.startswith("_") and not alias.name.startswith("__")
    }


def test_core_does_not_depend_on_operator_transport_or_zetsu_adapter() -> None:
    imports = _imports(ROOT / "laplace_core.py")

    assert not {"operator_api", "operator", "zetsu_agent", "zetsu_mcp"} & imports


def test_operator_settings_is_transport_only() -> None:
    imports = _imports(ROOT / "operator" / "settings.py")

    assert not any(name.startswith(("research_workspace", "operator_api", "zetsu")) for name in imports)


def test_operator_request_schemas_depend_only_on_manager_policy() -> None:
    imports = _imports(ROOT / "operator" / "agent_requests.py")

    assert imports == {"__future__", "typing", "pydantic", "manager_control"}


def test_operator_non_agent_request_models_depend_only_on_route_contracts() -> None:
    imports = _imports(ROOT / "operator" / "request_models.py")

    assert imports == {"__future__", "pydantic", "research_models", "typing"}


def test_operator_transport_helpers_remain_leaf_modules() -> None:
    assert _imports(ROOT / "operator" / "responses.py") == {
        "__future__", "collections.abc"
    }
    assert _imports(ROOT / "operator" / "json_utils.py") == {"__future__", "json"}
    assert _imports(ROOT / "operator" / "research_payloads.py") == {
        "__future__", "json", "pathlib"
    }
    assert _imports(ROOT / "operator" / "artifacts.py") == {
        "__future__", "pathlib", "fastapi"
    }
    assert _imports(ROOT / "operator" / "static_routes.py") == {
        "__future__", "fastapi", "fastapi.responses", "pathlib"
    }
    assert _imports(ROOT / "operator" / "client_routes.py") == {
        "__future__",
        "auth",
        "client_service",
        "collections.abc",
        "fastapi",
        "request_models",
    }
    assert _imports(ROOT / "operator" / "auth_routes.py") == {
        "__future__",
        "auth",
        "auth_registry",
        "auth_sessions",
        "collections.abc",
        "fastapi",
        "fastapi.responses",
        "request_models",
        "settings",
        "time",
    }
    assert _imports(ROOT / "operator" / "run_routes.py") == {
        "__future__",
        "auth",
        "collections.abc",
        "fastapi",
        "operator_service",
        "request_models",
    }


def test_operator_worktree_routes_are_a_separate_transport_boundary() -> None:
    assert _imports(ROOT / "operator" / "worktree_routes.py") == {
        "__future__",
        "auth",
        "collections.abc",
        "fastapi",
        "hashlib",
        "operator_service",
        "personal_corpus",
        "request_models",
        "service_tiers",
    }
    facade = ROOT / "operator_api.py"
    assert "operator.worktree_routes" in _imports(facade)
    assert "register_worktree_routes(" in facade.read_text(encoding="utf-8")


def test_shared_agent_git_primitive_has_no_policy_dependencies() -> None:
    imports = _imports(ROOT / "agent_infrastructure" / "git.py")

    assert imports == {
        "__future__",
        "collections.abc",
        "os",
        "pathlib",
        "shutil",
        "signal",
        "subprocess",
        "time",
    }


def test_shared_agent_process_primitive_has_no_policy_dependencies() -> None:
    assert _imports(ROOT / "agent_infrastructure" / "process.py") == {
        "__future__",
        "os",
        "signal",
        "subprocess",
        "typing",
    }


def test_authoritative_sandbox_uses_shared_git_primitive() -> None:
    imports = _imports(ROOT / "agent_sandbox.py")

    assert "agent_infrastructure.git" in imports
    assert "self._runner(" not in (ROOT / "agent_sandbox.py").read_text(encoding="utf-8")


def test_zetsu_handoff_artifacts_do_not_depend_on_the_coordinator() -> None:
    imports = _imports(ROOT / "zetsu_handoff.py")

    assert "zetsu_agent" not in imports
    assert {"agent_infrastructure.git", "zetsu_checkpoint"} <= imports


def test_zetsu_tools_do_not_depend_on_the_coordinator() -> None:
    imports = _imports(ROOT / "zetsu_tools.py")

    assert "zetsu_agent" not in imports
    assert {"personal_corpus", "zetsu_state"} <= imports


def test_current_cross_module_consumers_use_named_internal_interfaces() -> None:
    consumers = (
        "acquisition.py",
        "library.py",
        "model_artifacts.py",
        "model_routing.py",
        "multilanguage_ablation.py",
        "orchestration_certification.py",
        "quality_improvement.py",
        "repair_protocol.py",
        "rtl_contract.py",
        "team_runner.py",
        "governed_corpus.py",
        "research_web_adapters.py",
    )

    assert all(not _private_from_imports(ROOT / consumer) for consumer in consumers)
