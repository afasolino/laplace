"""Start the authenticated local Research and Operator Plane."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
from pathlib import Path
from typing import Sequence

import uvicorn

from .artifact_registry import ArtifactRegistry
from .auth_registry import AuthRegistryError, RegisteredUserRegistry
from .auth_sessions import AuthAuditLog, RegisteredEmailAuth, SessionStore
from .conversations import ConversationStore
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
from .research_admission import ResearchAdmissionStore
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


def _selected_lane_policy(repository_root: Path, *, codev_enabled: bool = True) -> LanePolicy:
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
        codev_enabled=codev_enabled,
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
    xdg_state = os.environ.get("XDG_STATE_HOME")
    state_default = (
        Path(xdg_state).expanduser() / "laplace"
        if xdg_state
        else Path.home() / ".local/state/laplace"
    )
    parser = argparse.ArgumentParser(prog="laplace-operator-server")
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=state_default,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--deployment-mode",
        choices=["local", "ssh-tunnel", "reverse-proxy"],
        default="local",
    )
    parser.add_argument("--allowed-origin", action="append")
    parser.add_argument("--allowed-host", action="append")
    parser.add_argument("--trusted-proxy", action="append")
    parser.add_argument("--external-url")
    parser.add_argument("--user-registry", type=Path)
    parser.add_argument("--session-store", type=Path)
    parser.add_argument("--session-idle-timeout", type=int, default=30 * 60)
    parser.add_argument("--session-absolute-timeout", type=int, default=12 * 60 * 60)
    parser.add_argument("--allow-insecure-lan-http", action="store_true")
    parser.add_argument("--enable-bearer-api", action="store_true")
    parser.add_argument("--bearer-token-file", type=Path)
    parser.add_argument("--no-pwa", action="store_true")
    parser.add_argument("--codev-disabled", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    state_root = arguments.state_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_root, 0o700)
    registry_path = (
        arguments.user_registry
        or (
            Path(os.environ["LAPLACE_USER_REGISTRY"])
            if "LAPLACE_USER_REGISTRY" in os.environ
            else state_root / "auth/registered_users.yaml"
        )
    ).expanduser().resolve()
    session_path = (
        arguments.session_store or state_root / "auth/sessions.sqlite3"
    ).expanduser().resolve()
    if not registry_path.exists():
        raise RuntimeError(
            "Registered-user registry is missing. Run "
            "`python -m research_workspace.user_admin bootstrap` first."
        )
    token_roles: dict[str, str | AuthCredential] = {}
    if arguments.enable_bearer_api:
        if arguments.bearer_token_file is None:
            raise RuntimeError(
                "--enable-bearer-api requires an explicit --bearer-token-file"
            )
        token_roles, initial_tokens = load_or_create_tokens(
            arguments.bearer_token_file.expanduser().resolve()
        )
        if initial_tokens is not None:
            print("Non-browser API tokens initialized (shown once):")
            for name, token in initial_tokens.items():
                print(f"{name}: {token}")
            print("Store these values securely; the GUI never reads this file.")
    origins = tuple(
        arguments.allowed_origin
        or (
            f"http://127.0.0.1:{arguments.port}",
            f"http://localhost:{arguments.port}",
        )
    )
    external_host: str | None = None
    if arguments.external_url:
        from urllib.parse import urlsplit

        external_host = urlsplit(arguments.external_url).hostname
    hosts = tuple(
        arguments.allowed_host
        or tuple(
            value
            for value in ("127.0.0.1", "localhost", external_host)
            if value is not None
        )
    )
    if arguments.deployment_mode == "reverse-proxy":
        if not arguments.allowed_origin or not arguments.allowed_host:
            raise RuntimeError(
                "reverse-proxy mode requires explicit --allowed-origin and --allowed-host"
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
    authorizations = RepositoryAuthorizationStore(
        state_root / "tiered_serving/repository_authorizations.sqlite3"
    )
    registry = RegisteredUserRegistry(registry_path)
    for registered_user in registry.snapshot.users_by_id.values():
        for repo_id in registered_user.authorized_repo_ids:
            try:
                authorizations.repository(repo_id)
            except Exception as exc:
                raise AuthRegistryError(
                    "unknown_repository_id",
                    {"repo_id": repo_id, "user_id": registered_user.user_id},
                ) from exc
        users.set_user(
            registered_user.user_id,
            registered_user.capability_tier,
            enabled=registered_user.enabled,
            capabilities=registered_user.effective_capabilities,
        )
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
    sandboxes = AgentSandboxManager(
        state_root / "tiered_serving/worktrees",
        authorizations,
    )
    local_chat = LocalOpenAIChatBackend()
    lane_policy = _selected_lane_policy(
        arguments.repository_root,
        codev_enabled=not arguments.codev_disabled,
    )
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
    sessions = SessionStore(
        session_path,
        idle_timeout_seconds=arguments.session_idle_timeout,
        absolute_timeout_seconds=arguments.session_absolute_timeout,
    )
    registered_auth = RegisteredEmailAuth(
        registry,
        sessions,
        AuthAuditLog(state_root / "auth/authentication_audit.jsonl"),
    )
    conversations = ConversationStore(state_root / "conversations/conversations.sqlite3")
    artifacts = ArtifactRegistry(
        state_root / "artifacts/artifact_registry.sqlite3",
        state_root / "artifacts/content",
        state_root / "artifacts/lifecycle.jsonl",
        state_root / "auth/artifact_pseudonym.key",
    )
    research_admission = ResearchAdmissionStore(
        state_root / "research/research_admission.sqlite3"
    )
    app = create_operator_app(
        operator,
        OperatorAuth(token_roles),
        settings=OperatorApiSettings(
            bind_host=arguments.host,
            port=arguments.port,
            deployment_mode=arguments.deployment_mode,
            allowed_origins=origins,
            allowed_hosts=hosts,
            trusted_proxies=tuple(
                arguments.trusted_proxy or ("127.0.0.1", "::1")
            ),
            external_url=arguments.external_url,
            allow_insecure_lan_http=arguments.allow_insecure_lan_http,
            bearer_api_enabled=arguments.enable_bearer_api,
            codev_enabled=not arguments.codev_disabled,
            pwa_enabled=(
                not arguments.no_pwa
                and arguments.deployment_mode != "reverse-proxy"
            ),
        ),
        research=research,
        tiered=tiered,
        serving_profile_operator=profile_operator,
        registered_auth=registered_auth,
        conversation_store=conversations,
        artifact_registry=artifacts,
        research_admission=research_admission,
    )

    def reload_registry(_signum: int, _frame: object) -> None:
        before = registry.snapshot
        accepted, category, after = registry.try_reload()
        if not accepted:
            registered_auth.audit.append(
                "REGISTRY_RELOAD",
                outcome="DENIED",
                reason=category,
            )
            return
        changed = {
            user_id
            for user_id in set(before.users_by_id) | set(after.users_by_id)
            if before.users_by_id.get(user_id) != after.users_by_id.get(user_id)
        }
        for user_id in changed:
            sessions.revoke_user(user_id)
            current = after.users_by_id.get(user_id)
            if current is not None:
                users.set_user(
                    user_id,
                    current.capability_tier,
                    enabled=current.enabled,
                    capabilities=current.effective_capabilities,
                )
        registered_auth.audit.append(
            "REGISTRY_RELOAD",
            outcome="SUCCESS",
            reason=f"changed_users:{len(changed)}",
        )

    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, reload_registry)
    uvicorn.run(
        app,
        host=arguments.host,
        port=arguments.port,
        log_level="info",
        access_log=True,
        proxy_headers=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
