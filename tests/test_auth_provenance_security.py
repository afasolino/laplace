from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

from research_workspace.agent_sandbox import AgentSandboxManager
from research_workspace.artifact_registry import ArtifactRegistry, ArtifactRegistryError
from research_workspace.auth_registry import (
    AuthRegistryError,
    RegisteredUser,
    RegisteredUserRegistry,
    hash_secret,
    write_registry,
)
from research_workspace.auth_sessions import (
    AuthAuditLog,
    AuthSessionError,
    RegisteredEmailAuth,
    SessionStore,
)
from research_workspace.conversations import ConversationError, ConversationStore
from research_workspace.operator_api import OperatorApiSettings, OperatorAuth, create_operator_app
from research_workspace.operator_service import OperatorService
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.service_tiers import (
    LanePolicy,
    ModelLane,
    ModelRoute,
    TierAuditLog,
    TieredServingService,
)
from research_workspace.user_capabilities import CapabilityTier, UserCapabilityStore


ROOT = Path(__file__).resolve().parents[1]


class _Chat:
    def complete(
        self,
        *,
        messages: list[dict[str, str]] | tuple[dict[str, str], ...],
        route: ModelRoute,
        tools: tuple[object, ...],
        request_id: str,
    ) -> dict[str, object]:
        assert tools == ()
        del messages, route, request_id
        return {
            "content": "Safe **local** answer.\n```python\nprint('ok')\n```",
            "finish_reason": "stop",
            "verification_status": "PASSED",
        }


class _Agent:
    def run(self, **_kwargs: object) -> dict[str, object]:
        return {
            "content": "Validated patch applied.",
            "finish_reason": "stop",
            "verification_status": "PASSED",
            "modified_paths": ["src/example.py"],
        }


def _user(
    email: str,
    user_id: str,
    tier: CapabilityTier,
    *,
    role: str = "user",
    lane: str = "standard",
    enabled: bool = True,
    must_change: bool = False,
    secret: str = "correct horse battery staple",
    repositories: tuple[str, ...] = (),
) -> RegisteredUser:
    return RegisteredUser(
        email=email,
        user_id=user_id,
        display_name=email.split("@", 1)[0].replace(".", " ").title(),
        enabled=enabled,
        capability_tier=tier,
        role=role,
        default_lane=lane,
        authorized_repo_ids=repositories,
        password_hash=hash_secret(secret),
        must_change_password=must_change,
    )


def _registered_auth(
    root: Path,
    users: list[RegisteredUser],
    *,
    clock: object | None = None,
) -> RegisteredEmailAuth:
    registry_path = root / "auth/registered_users.yaml"
    write_registry(registry_path, users)
    sessions = SessionStore(
        root / "auth/sessions.sqlite3",
        idle_timeout_seconds=60,
        absolute_timeout_seconds=120,
        **({"clock": clock} if clock is not None else {}),
    )
    return RegisteredEmailAuth(
        RegisteredUserRegistry(registry_path),
        sessions,
        AuthAuditLog(root / "auth/authentication_audit.jsonl"),
    )


def _tiered(root: Path, users: list[RegisteredUser]) -> TieredServingService:
    capabilities = UserCapabilityStore(root / "tier/users.sqlite3")
    for user in users:
        capabilities.set_user(user.user_id, user.capability_tier, enabled=user.enabled)
    authorizations = RepositoryAuthorizationStore(root / "tier/repos.sqlite3")
    policy = LanePolicy(
        routes={
            lane: ModelRoute(
                lane=lane,
                model_id=f"fixture-{lane.value}",
                endpoint=f"http://127.0.0.1:{8200 + index}/v1",
                priority=3 - index,
                context_limit=8192,
                output_limit=1024,
            )
            for index, lane in enumerate(ModelLane)
        },
        quality_reserved_slots=1,
        standard_capacity=2,
        economy_capacity=4,
    )
    return TieredServingService(
        users=capabilities,
        sandboxes=AgentSandboxManager(root / "tier/worktrees", authorizations),
        lane_policy=policy,
        chat_backend=_Chat(),
        agent_backend=_Agent(),
        audit_log=TierAuditLog(root / "tier/audit.jsonl"),
    )


@pytest.mark.anyio
async def test_activation_login_cookie_csrf_and_generic_failure(tmp_path: Path) -> None:
    activation_code = "one-time-code-with-enough-entropy"
    account = _user(
        "afasolino@unisa.it",
        "usr_afasolino",
        CapabilityTier.OPERATOR,
        role="admin",
        lane="quality",
        must_change=True,
        secret=activation_code,
    )
    auth = _registered_auth(tmp_path, [account])
    app = create_operator_app(
        OperatorService(ROOT, tmp_path / "operator"),
        OperatorAuth({}),
        settings=OperatorApiSettings(
            deployment_mode="reverse-proxy",
            allowed_origins=("https://laplace.example",),
            allowed_hosts=("laplace.example",),
            external_url="https://laplace.example",
            fixture_mode=True,
        ),
        tiered=_tiered(tmp_path, [account]),
        registered_auth=auth,
        conversation_store=ConversationStore(tmp_path / "conversations.sqlite3"),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://laplace.example",
    ) as client:
        proxy_headers = {
            "X-Forwarded-For": "192.0.2.10",
            "X-Forwarded-Host": "laplace.example",
            "X-Forwarded-Proto": "https",
        }
        failed = await client.post(
            "/api/v1/auth/login",
            headers=proxy_headers,
            json={"email": "not-registered@example.test", "password": "wrong"},
        )
        assert failed.status_code == 401
        assert failed.json()["detail"] == "authentication_failed"
        assert "not-registered@example.test" not in auth.audit.path.read_text()

        activation = await client.post(
            "/api/v1/auth/activate",
            headers={"Origin": "https://laplace.example", **proxy_headers},
            json={
                "email": " AFASOLINO@UNISA.IT ",
                "activation_code": activation_code,
                "new_password": "a new password manager value of sufficient length",
            },
        )
        assert activation.status_code == 200
        cookie = activation.headers["set-cookie"]
        assert "laplace_session=" in cookie
        assert "HttpOnly" in cookie
        assert "Secure" in cookie
        assert "SameSite=strict" in cookie
        assert "Path=/" in cookie
        csrf = activation.json()["csrf_token"]
        assert auth.registry.require_user("usr_afasolino").must_change_password is False

        missing_csrf = await client.post(
            "/api/v1/conversations",
            json={"title": "Denied"},
        )
        assert missing_csrf.status_code == 403
        created = await client.post(
            "/api/v1/conversations",
            headers={"Origin": "https://laplace.example", "X-CSRF-Token": csrf},
            json={"title": "Private conversation"},
        )
        assert created.status_code == 200

        logout = await client.post(
            "/api/v1/auth/logout",
            headers={"Origin": "https://laplace.example", "X-CSRF-Token": csrf},
        )
        assert logout.status_code == 200
        signed_out = await client.get("/api/v1/auth/session")
        assert signed_out.status_code == 200
        assert signed_out.json()["status"] == "SIGNED_OUT"


@pytest.mark.anyio
async def test_production_readiness_probes_exact_local_model_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = _user(
        "operator@example.test",
        "operator",
        CapabilityTier.OPERATOR,
        role="admin",
    )
    auth = _registered_auth(tmp_path, [account])
    app = create_operator_app(
        OperatorService(ROOT, tmp_path / "operator"),
        OperatorAuth({}),
        settings=OperatorApiSettings(fixture_mode=False),
        tiered=_tiered(tmp_path, [account]),
        registered_auth=auth,
        conversation_store=ConversationStore(tmp_path / "conversations.sqlite3"),
    )

    class ModelWriter:
        def write(self, _value: bytes) -> None:
            return

        async def drain(self) -> None:
            return

        def close(self) -> None:
            return

        async def wait_closed(self) -> None:
            return

    serve_wrong_model = False

    async def open_model(
        _host: str,
        port: int,
    ) -> tuple[asyncio.StreamReader, ModelWriter]:
        lane = list(ModelLane)[port - 8200]
        model_id = "wrong-model" if serve_wrong_model else f"fixture-{lane.value}"
        body = json.dumps({"data": [{"id": model_id}]}).encode()
        reader = asyncio.StreamReader()
        reader.feed_data(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        reader.feed_eof()
        return reader, ModelWriter()

    monkeypatch.setattr(
        "research_workspace.operator_api.asyncio.open_connection",
        open_model,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8765",
    ) as client:
        ready = await client.get("/api/v1/readiness")
        assert ready.status_code == 200
        assert ready.json()["status"] == "READY"

        serve_wrong_model = True
        degraded = await client.get("/api/v1/readiness")
        assert degraded.status_code == 503
        assert set(degraded.json()["reasons"]) == {
            "model_identity_mismatch:quality",
            "model_identity_mismatch:standard",
            "model_identity_mismatch:economy",
        }


def test_session_idle_absolute_and_account_revision_expiry(tmp_path: Path) -> None:
    now = [1_000.0]
    user = _user("basic@example.test", "usr_basic", CapabilityTier.BASIC)
    auth = _registered_auth(tmp_path, [user], clock=lambda: now[0])
    logged_in = auth.login(
        user.email,
        "correct horse battery staple",
        client_ip="127.0.0.1",
        trace_id="trace-test",
    )
    assert len(logged_in.identifier) >= 22
    assert logged_in.identifier not in auth.sessions.path.read_bytes().decode(
        "utf-8", errors="ignore"
    )
    now[0] = 1_061.0
    with pytest.raises(AuthSessionError, match="session_expired"):
        auth.sessions.resolve(logged_in.identifier, auth.registry)

    second = auth.login(
        user.email,
        "correct horse battery staple",
        client_ip="127.0.0.1",
        trace_id="trace-test",
    )
    auth.registry.update_user(user.user_id, default_lane="quality")
    with pytest.raises(AuthSessionError, match="session_revoked"):
        auth.sessions.resolve(second.identifier, auth.registry)


def test_login_rate_limit_is_per_email_and_ip(tmp_path: Path) -> None:
    user = _user("basic@example.test", "usr_basic", CapabilityTier.BASIC)
    auth = _registered_auth(tmp_path, [user])
    for _ in range(5):
        with pytest.raises(AuthSessionError, match="authentication_failed"):
            auth.login(
                user.email,
                "wrong password",
                client_ip="192.0.2.20",
                trace_id="trace-rate",
            )
    with pytest.raises(AuthSessionError, match="authentication_rate_limited") as blocked:
        auth.login(
            user.email,
            "wrong password",
            client_ip="192.0.2.20",
            trace_id="trace-rate",
        )
    assert blocked.value.retry_after_seconds is not None


def test_registry_strict_permissions_and_last_valid_reload(tmp_path: Path) -> None:
    user = _user("basic@example.test", "usr_basic", CapabilityTier.BASIC)
    path = tmp_path / "auth/registered_users.yaml"
    write_registry(path, [user])
    registry = RegisteredUserRegistry(path)
    revision = registry.snapshot.revision
    path.write_text("schema_version: 1\nusers:\n  - broken: true\n", encoding="utf-8")
    os.chmod(path, 0o600)
    accepted, category, snapshot = registry.try_reload()
    assert accepted is False
    assert category == "invalid_user_schema"
    assert snapshot.revision == revision
    if os.name == "nt":
        write_registry(path, [user])
        assert RegisteredUserRegistry(path).snapshot.users_by_id[user.user_id] == user
        return
    os.chmod(path, 0o644)
    with pytest.raises(AuthRegistryError, match="registry_file_permissions"):
        RegisteredUserRegistry(path)


def test_registry_repository_tuple_update_is_serialized_as_a_strict_list(
    tmp_path: Path,
) -> None:
    user = _user("owner@example.test", "owner", CapabilityTier.PLUS)
    path = tmp_path / "auth/registered_users.yaml"
    write_registry(path, [user])
    registry = RegisteredUserRegistry(path)
    registry.update_user("owner", authorized_repo_ids=("repo-one",))
    assert (
        RegisteredUserRegistry(path).require_user("owner").authorized_repo_ids
        == ("repo-one",)
    )


def test_conversation_owner_isolation(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create("usr_one", title="One")
    store.append_message(
        "usr_one",
        conversation.conversation_id,
        role="user",
        content="private",
    )
    with pytest.raises(ConversationError, match="conversation_not_found"):
        store.get_with_messages("usr_two", conversation.conversation_id)
    assert store.list("usr_two") == []


def test_artifact_identity_hash_owner_repo_and_clean_export(tmp_path: Path) -> None:
    registry = ArtifactRegistry(
        tmp_path / "registry.sqlite3",
        tmp_path / "content",
        tmp_path / "events.jsonl",
        tmp_path / "owner.key",
    )
    first = registry.create(
        owner_user_id="usr_one",
        content=b"clean content\n",
        relative_path="reports/result.md",
        source_state_fingerprint="a" * 64,
        generator_model_route="quality",
        capability_tier="plus",
        trace_id="trace-create",
        repo_id="repo-one",
    )
    assert len(first.artifact_id) == 26
    assert first.owner_user_id not in first.pseudonymous_owner_id
    renamed = registry.rename(
        first.artifact_id,
        owner_user_id="usr_one",
        repo_id="repo-one",
        new_relative_path="reports/final.md",
        capability_tier="plus",
        trace_id="trace-rename",
    )
    assert renamed.artifact_id == first.artifact_id
    updated = registry.update_content(
        first.artifact_id,
        owner_user_id="usr_one",
        repo_id="repo-one",
        content=b"updated clean content\n",
        capability_tier="plus",
        trace_id="trace-update",
    )
    assert updated.content_sha256 == hashlib.sha256(b"updated clean content\n").hexdigest()
    exported = registry.export_normal(
        first.artifact_id,
        owner_user_id="usr_one",
        repo_id="repo-one",
        destination=tmp_path / "export",
        capability_tier="plus",
        trace_id="trace-export",
    )
    assert exported.name == "final.md"
    assert exported.read_text() == "updated clean content\n"
    assert "artifact_id" not in exported.read_text()
    with pytest.raises(ArtifactRegistryError, match="artifact_not_found"):
        registry.read(
            first.artifact_id,
            owner_user_id="usr_two",
            repo_id="repo-one",
            capability_tier="plus",
            trace_id="trace-denied",
        )
    with pytest.raises(ArtifactRegistryError, match="artifact_not_found"):
        registry.read(
            first.artifact_id,
            owner_user_id="usr_one",
            repo_id="repo-two",
            capability_tier="plus",
            trace_id="trace-denied",
        )
    content_path = next((tmp_path / "content").rglob("final.md"))
    content_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ArtifactRegistryError, match="artifact_integrity_failure"):
        registry.read(
            first.artifact_id,
            owner_user_id="usr_one",
            repo_id="repo-one",
            capability_tier="plus",
            trace_id="trace-tamper",
        )
    compact = json.dumps(registry.compact_operator_export())
    assert first.artifact_id in compact
    tombstone = registry.delete(
        first.artifact_id,
        owner_user_id="usr_one",
        repo_id="repo-one",
        capability_tier="plus",
        trace_id="trace-delete",
    )
    assert tombstone.deleted_at_utc is not None


def test_frontend_has_no_browser_credential_storage_or_unsafe_html_sink() -> None:
    javascript = (
        ROOT / "src/research_workspace/operator_web/app.js"
    ).read_text(encoding="utf-8")
    html = (ROOT / "src/research_workspace/operator_web/index.html").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "localStorage",
        "sessionStorage",
        ".innerHTML",
        "insertAdjacentHTML",
        "Access token",
        "Bearer ",
    ):
        assert forbidden not in javascript
    assert "Access token" not in html
    assert 'type="email"' in html
    assert "password_hash" not in html
    worker = (ROOT / "src/research_workspace/operator_web/sw.js").read_text(
        encoding="utf-8"
    )
    assert "cache.add" not in worker
    assert "caches.match" not in worker
