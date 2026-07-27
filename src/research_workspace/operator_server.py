"""Start the authenticated local Research and Operator Plane."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import Sequence

import uvicorn

from .operator_api import (
    AuthCredential,
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from .agent_sandbox import AgentSandboxManager
from .operator_service import OperatorService
from .repository_authorization import RepositoryAuthorizationStore
from .research_models import LocalDirectoryResearchAdapter, ResearchAdapter
from .research_plane import DeepResearchService
from .service_tiers import (
    LanePolicy,
    LocalOpenAIChatBackend,
    ModelLane,
    ModelRoute,
    TierAuditLog,
    TieredServingService,
    ValidatedPatchAgentBackend,
)
from .serving_profile_runtime import ServingProfileOperator
from .user_capabilities import CapabilityTier, UserCapabilityStore


def _atomic_private_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _selected_lane_policy(repository_root: Path) -> LanePolicy:
    path = repository_root.resolve() / "configs/selected_serving_profiles.json"
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RuntimeError("Selected serving profile configuration is malformed")
    routes = raw.get("routes")
    if not isinstance(routes, dict) or set(routes) != {lane.value for lane in ModelLane}:
        raise RuntimeError("Selected serving routes are incomplete")
    parsed: dict[ModelLane, ModelRoute] = {}
    for lane in ModelLane:
        route = routes.get(lane.value)
        if not isinstance(route, dict) or set(route) != {
            "model_id",
            "endpoint",
            "priority",
            "context_limit",
            "output_limit",
        }:
            raise RuntimeError(f"Selected {lane.value} route is malformed")
        parsed[lane] = ModelRoute(
            lane=lane,
            model_id=str(route["model_id"]),
            endpoint=str(route["endpoint"]),
            priority=int(route["priority"]),
            context_limit=int(route["context_limit"]),
            output_limit=int(route["output_limit"]),
        )
    return LanePolicy(
        routes=parsed,
        quality_reserved_slots=int(raw["quality_reserved_slots"]),
        standard_capacity=int(raw["standard_capacity"]),
        economy_capacity=int(raw["economy_capacity"]),
    )


def load_or_create_tokens(
    path: Path,
) -> tuple[dict[str, str | AuthCredential], dict[str, str] | None]:
    """Load mode-0600 credentials or create operator, Basic, and Plus tokens."""

    if path.is_file():
        if path.stat().st_mode & 0o077:
            raise RuntimeError("Operator authentication file must have mode 0600")
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        tokens = raw.get("tokens") if isinstance(raw, dict) else None
        if not isinstance(tokens, dict):
            raise RuntimeError("Operator authentication file is malformed")
        parsed: dict[str, str | AuthCredential] = {}
        for token, value in tokens.items():
            if not isinstance(token, str):
                raise RuntimeError("Operator authentication file is malformed")
            if isinstance(value, str):
                parsed[token] = value
                continue
            if not isinstance(value, dict):
                raise RuntimeError("Operator authentication file is malformed")
            try:
                parsed[token] = AuthCredential(
                    role=str(value["role"]),
                    user_id=str(value["user_id"]),
                    capability_tier=CapabilityTier(str(value["capability_tier"])),
                )
            except (KeyError, ValueError) as exc:
                raise RuntimeError("Operator authentication file is malformed") from exc
        return parsed, None
    by_role = {
        role: f"laplace-{role}-{secrets.token_urlsafe(32)}"
        for role in ("read", "operate", "approve", "admin")
    }
    by_tier = {
        "basic": f"laplace-basic-{secrets.token_urlsafe(32)}",
        "plus": f"laplace-plus-{secrets.token_urlsafe(32)}",
    }
    token_roles: dict[str, str | AuthCredential] = {
        token: AuthCredential(
            role=role,
            user_id=f"operator-{role}",
            capability_tier=CapabilityTier.OPERATOR,
        )
        for role, token in by_role.items()
    }
    token_roles[by_tier["basic"]] = AuthCredential(
        role="read",
        user_id="basic-local",
        capability_tier=CapabilityTier.BASIC,
    )
    token_roles[by_tier["plus"]] = AuthCredential(
        role="read",
        user_id="plus-local",
        capability_tier=CapabilityTier.PLUS,
    )
    _atomic_private_json(
        path,
        {
            "schema_version": 2,
            "tokens": {
                token: (
                    {
                        "role": value.role,
                        "user_id": value.user_id,
                        "capability_tier": value.capability_tier.value,
                    }
                    if isinstance(value, AuthCredential)
                    else value
                )
                for token, value in token_roles.items()
            },
            "warning": "Secret role tokens; never include this file in bundles.",
        },
    )
    return token_roles, {**by_role, **by_tier}


def _parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(prog="laplace-operator-server")
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=repository_root / "outputs/operator_plane",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allowed-origin", action="append")
    parser.add_argument("--no-pwa", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    state_root = arguments.state_root.resolve()
    token_path = state_root / ".operator_auth.json"
    token_roles, initial_tokens = load_or_create_tokens(token_path)
    if initial_tokens is not None:
        print("Local authentication initialized. Tokens (shown once):")
        for name, token in initial_tokens.items():
            print(f"{name}: {token}")
        print(f"Secret token file: {token_path}")
    origins = tuple(
        arguments.allowed_origin
        or (
            f"http://127.0.0.1:{arguments.port}",
            f"http://localhost:{arguments.port}",
        )
    )
    operator = OperatorService(arguments.repository_root, state_root)
    adapters: dict[str, ResearchAdapter] = {
        "local_governed_corpus": LocalDirectoryResearchAdapter(
            state_root / "stores/governed_corpus",
            name="local_governed_corpus",
            source_type="official_documentation",
        ),
        "local_uploaded_documents": LocalDirectoryResearchAdapter(
            state_root / "stores/personal_workspace_store",
            name="local_uploaded_documents",
        ),
    }
    research = DeepResearchService(state_root, adapters)
    users = UserCapabilityStore(state_root / "tiered_serving/users.sqlite3")
    for value in token_roles.values():
        credential = (
            AuthCredential(
                role=value,
                user_id=f"operator-{value}",
                capability_tier=CapabilityTier.OPERATOR,
            )
            if isinstance(value, str)
            else value
        )
        users.set_user(
            credential.user_id,
            credential.capability_tier,
            enabled=True,
        )
    authorizations = RepositoryAuthorizationStore(
        state_root / "tiered_serving/repository_authorizations.sqlite3"
    )
    sandboxes = AgentSandboxManager(
        state_root / "tiered_serving/worktrees",
        authorizations,
    )
    local_chat = LocalOpenAIChatBackend()
    lane_policy = _selected_lane_policy(arguments.repository_root)
    tiered = TieredServingService(
        users=users,
        sandboxes=sandboxes,
        lane_policy=lane_policy,
        chat_backend=local_chat,
        agent_backend=ValidatedPatchAgentBackend(local_chat),
        audit_log=TierAuditLog(state_root / "tiered_serving/audit.jsonl"),
    )
    profile_operator = ServingProfileOperator(
        arguments.repository_root,
        state_root / "tiered_serving/profile_runtime",
        Path("/home/giando/work/laplace/.venv-vllm-cu129/bin/vllm"),
        Path("/home/giando/work/laplace/.runtime/ffmpeg7/lib"),
    )
    app = create_operator_app(
        operator,
        OperatorAuth(token_roles),
        settings=OperatorApiSettings(
            bind_host=arguments.host,
            port=arguments.port,
            allowed_origins=origins,
            pwa_enabled=not arguments.no_pwa,
        ),
        research=research,
        tiered=tiered,
        serving_profile_operator=profile_operator,
    )
    uvicorn.run(
        app,
        host=arguments.host,
        port=arguments.port,
        log_level="info",
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
