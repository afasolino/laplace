"""Local no-network administration for registered Laplace users."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
from pathlib import Path
from typing import Sequence

from .auth_registry import (
    AuthRegistryError,
    RegisteredUser,
    RegisteredUserRegistry,
    hash_secret,
    normalize_email,
    parse_registry,
    write_registry,
)
from .auth_sessions import AuthAuditLog, SessionStore
from .user_capabilities import Capability, CapabilityTier, default_capabilities


def _registry_path(value: str | None) -> Path:
    configured = value or os.environ.get("LAPLACE_USER_REGISTRY")
    if not configured:
        raise AuthRegistryError("registry_path_required")
    return Path(configured).expanduser().resolve()


def _activation_code() -> str:
    return secrets.token_urlsafe(32)


def _default_display_name(email: str) -> str:
    if normalize_email(email) == "afasolino@unisa.it":
        return "Alfonso Fasolino"
    local = email.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
    return local.title() or "Laplace User"


def _user_from_arguments(arguments: argparse.Namespace, code: str) -> RegisteredUser:
    email = str(arguments.email).strip()
    tier = CapabilityTier(str(arguments.capability_tier))
    selected = getattr(arguments, "capabilities", None)
    capabilities = tuple(
        sorted(
            (
                Capability(str(item))
                for item in selected
            )
            if selected
            else default_capabilities(tier),
            key=str,
        )
    )
    return RegisteredUser(
        email=email,
        user_id=str(arguments.user_id or f"usr_{normalize_email(email).split('@', 1)[0]}"),
        display_name=str(arguments.display_name or _default_display_name(email)),
        enabled=True,
        capability_tier=tier,
        role=str(arguments.role),
        default_lane=str(arguments.default_lane),
        authorized_repo_ids=(),
        password_hash=hash_secret(code),
        must_change_password=True,
        capabilities=capabilities,
    )


def _load_or_create(path: Path) -> RegisteredUserRegistry:
    if not path.exists():
        write_registry(path, [])
    return RegisteredUserRegistry(path)


def _print_once(code: str) -> None:
    print("One-time activation code (shown once):")
    print(code)
    print("The code is not stored in plaintext. Complete activation before closing this terminal.")


def _add_common_user_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--email", required=True)
    parser.add_argument("--user-id")
    parser.add_argument("--display-name")
    parser.add_argument(
        "--capability-tier",
        choices=[item.value for item in CapabilityTier],
        required=True,
    )
    parser.add_argument(
        "--capability",
        dest="capabilities",
        action="append",
        choices=[item.value for item in Capability],
        help=(
            "Independent capability; repeat as needed. If omitted, secure legacy "
            "defaults for the selected tier are used."
        ),
    )
    parser.add_argument("--role", choices=["user", "operator", "auditor", "admin"], required=True)
    parser.add_argument(
        "--default-lane",
        choices=["quality", "standard", "economy"],
        required=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research_workspace.user_admin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--registry", required=True, help="External registered_users.yaml")
    bootstrap.add_argument("--session-store", type=Path)
    _add_common_user_arguments(bootstrap)

    add = subparsers.add_parser("add")
    add.add_argument("--registry", required=True, help="External registered_users.yaml")
    add.add_argument("--session-store", type=Path)
    _add_common_user_arguments(add)

    list_command = subparsers.add_parser("list")
    list_command.add_argument("--registry", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--registry", required=True)

    for command in ("disable", "enable", "revoke-sessions"):
        current = subparsers.add_parser(command)
        current.add_argument("--registry", required=True)
        current.add_argument("--email", required=True)
        if command == "revoke-sessions":
            current.add_argument("--session-store", type=Path, required=True)

    for command, flag, choices in (
        ("set-role", "--role", ["user", "operator", "auditor", "admin"]),
        ("set-tier", "--capability-tier", [item.value for item in CapabilityTier]),
        ("set-default-lane", "--default-lane", ["quality", "standard", "economy"]),
    ):
        current = subparsers.add_parser(command)
        current.add_argument("--registry", required=True)
        current.add_argument("--email", required=True)
        current.add_argument(flag, choices=choices, required=True)

    set_capabilities = subparsers.add_parser("set-capabilities")
    set_capabilities.add_argument("--registry", required=True)
    set_capabilities.add_argument("--email", required=True)
    set_capabilities.add_argument(
        "--capability",
        dest="capabilities",
        action="append",
        choices=[item.value for item in Capability],
        default=[],
        help="Repeat to grant multiple independent capabilities; omit all to deny all.",
    )

    for command in ("authorize-repo", "revoke-repo"):
        current = subparsers.add_parser(command)
        current.add_argument("--registry", required=True)
        current.add_argument("--email", required=True)
        current.add_argument("--repo-id", required=True)

    reset = subparsers.add_parser("reset-password")
    reset.add_argument("--registry", required=True)
    reset.add_argument("--email", required=True)
    reset.add_argument("--session-store", type=Path)

    reload_command = subparsers.add_parser("reload")
    reload_command.add_argument("--registry", required=True)
    reload_command.add_argument("--pid", type=int)
    return parser


def _user_by_email(registry: RegisteredUserRegistry, email: str) -> RegisteredUser:
    user = registry.by_email(email)
    if user is None:
        raise AuthRegistryError("unknown_user")
    return user


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        path = _registry_path(arguments.registry)
        audit = AuthAuditLog(path.parent / "authentication_audit.jsonl")
        command = str(arguments.command)
        if command in {"bootstrap", "add"}:
            registry = _load_or_create(path)
            code = _activation_code()
            user = _user_from_arguments(arguments, code)
            existing = registry.by_email(user.email)
            if command == "add" and existing is not None:
                raise AuthRegistryError("user_already_exists")
            registry.upsert(user)
            session_path = arguments.session_store or path.parent / "sessions.sqlite3"
            revoked = SessionStore(session_path).revoke_user(user.user_id)
            audit.append(
                "ACCOUNT_BOOTSTRAP" if command == "bootstrap" else "ACCOUNT_ADD",
                outcome="SUCCESS",
                user_id=user.user_id,
                normalized_email=user.normalized_email,
                reason=f"sessions_revoked:{revoked}",
            )
            _json(
                {
                    "status": "BOOTSTRAPPED" if command == "bootstrap" else "ADDED",
                    "email": user.email,
                    "user_id": user.user_id,
                    "capability_tier": user.capability_tier.value,
                    "role": user.role,
                    "default_lane": user.default_lane,
                    "must_change_password": True,
                    "registry": str(path),
                }
            )
            _print_once(code)
            return 0

        registry = RegisteredUserRegistry(path)
        if command == "validate":
            parsed = parse_registry(path.read_bytes())
            _json(
                {
                    "status": "VALID",
                    "user_count": len(parsed.users_by_id),
                    "revision": parsed.revision,
                }
            )
            return 0
        if command == "list":
            _json(
                {
                    "status": "OK",
                    "users": [
                        {"user_id": user.user_id, **user.public()}
                        for user in sorted(
                            registry.snapshot.users_by_id.values(),
                            key=lambda item: item.normalized_email,
                        )
                    ],
                }
            )
            return 0
        if command == "reload":
            snapshot = registry.reload()
            if arguments.pid is not None:
                os.kill(arguments.pid, signal.SIGHUP)
            _json({"status": "RELOADED", "revision": snapshot.revision})
            audit.append("REGISTRY_RELOAD", outcome="SUCCESS")
            return 0

        user = _user_by_email(registry, str(arguments.email))
        if command in {"disable", "enable"}:
            enabled = command == "enable"
            registry.update_user(user.user_id, enabled=enabled)
            revoked = SessionStore(path.parent / "sessions.sqlite3").revoke_user(
                user.user_id
            )
            audit.append(
                "ACCOUNT_ENABLE" if enabled else "ACCOUNT_DISABLE",
                outcome="SUCCESS",
                user_id=user.user_id,
                reason=f"sessions_revoked:{revoked}",
            )
            _json({"status": "UPDATED", "email": user.email, "enabled": enabled})
            return 0
        if command == "set-role":
            registry.update_user(user.user_id, role=str(arguments.role))
            revoked = SessionStore(path.parent / "sessions.sqlite3").revoke_user(
                user.user_id
            )
            audit.append(
                "ROLE_CHANGE",
                outcome="SUCCESS",
                user_id=user.user_id,
                reason=f"sessions_revoked:{revoked}",
            )
            _json({"status": "UPDATED", "email": user.email, "role": arguments.role})
            return 0
        if command == "set-tier":
            tier = CapabilityTier(str(arguments.capability_tier))
            registry.update_user(user.user_id, capability_tier=tier)
            revoked = SessionStore(path.parent / "sessions.sqlite3").revoke_user(
                user.user_id
            )
            audit.append(
                "CAPABILITY_CHANGE",
                outcome="SUCCESS",
                user_id=user.user_id,
                reason=f"sessions_revoked:{revoked}",
            )
            _json(
                {
                    "status": "UPDATED",
                    "email": user.email,
                    "capability_tier": tier.value,
                }
            )
            return 0
        if command == "set-capabilities":
            capabilities = tuple(
                sorted(
                    {Capability(str(item)) for item in arguments.capabilities},
                    key=str,
                )
            )
            registry.update_user(user.user_id, capabilities=capabilities)
            revoked = SessionStore(path.parent / "sessions.sqlite3").revoke_user(
                user.user_id
            )
            audit.append(
                "CAPABILITY_CHANGE",
                outcome="SUCCESS",
                user_id=user.user_id,
                reason=f"sessions_revoked:{revoked}",
            )
            _json(
                {
                    "status": "UPDATED",
                    "email": user.email,
                    "capabilities": [item.value for item in capabilities],
                }
            )
            return 0
        if command == "set-default-lane":
            registry.update_user(user.user_id, default_lane=str(arguments.default_lane))
            revoked = SessionStore(path.parent / "sessions.sqlite3").revoke_user(
                user.user_id
            )
            audit.append(
                "DEFAULT_LANE_CHANGE",
                outcome="SUCCESS",
                user_id=user.user_id,
                reason=f"sessions_revoked:{revoked}",
            )
            _json(
                {
                    "status": "UPDATED",
                    "email": user.email,
                    "default_lane": arguments.default_lane,
                }
            )
            return 0
        if command in {"authorize-repo", "revoke-repo"}:
            repositories = set(user.authorized_repo_ids)
            if command == "authorize-repo":
                repositories.add(str(arguments.repo_id))
            else:
                repositories.discard(str(arguments.repo_id))
            registry.update_user(
                user.user_id,
                authorized_repo_ids=tuple(sorted(repositories)),
            )
            revoked = SessionStore(path.parent / "sessions.sqlite3").revoke_user(
                user.user_id
            )
            audit.append(
                "REPOSITORY_AUTHORIZATION_CHANGE",
                outcome="SUCCESS",
                user_id=user.user_id,
                reason=f"sessions_revoked:{revoked}",
            )
            _json(
                {
                    "status": "UPDATED",
                    "email": user.email,
                    "authorized_repo_ids": sorted(repositories),
                }
            )
            return 0
        if command == "reset-password":
            code = _activation_code()
            registry.update_user(
                user.user_id,
                password_hash=hash_secret(code),
                must_change_password=True,
            )
            if arguments.session_store is not None:
                revoked = SessionStore(arguments.session_store).revoke_user(user.user_id)
            else:
                revoked = SessionStore(path.parent / "sessions.sqlite3").revoke_user(
                    user.user_id
                )
            audit.append(
                "PASSWORD_RESET",
                outcome="SUCCESS",
                user_id=user.user_id,
                reason=f"sessions_revoked:{revoked}",
            )
            _json(
                {
                    "status": "PASSWORD_RESET",
                    "email": user.email,
                    "must_change_password": True,
                }
            )
            _print_once(code)
            return 0
        if command == "revoke-sessions":
            count = SessionStore(arguments.session_store).revoke_user(user.user_id)
            audit.append(
                "SESSION_REVOCATION",
                outcome="SUCCESS",
                user_id=user.user_id,
                reason=f"sessions_revoked:{count}",
            )
            _json({"status": "SESSIONS_REVOKED", "email": user.email, "count": count})
            return 0
        raise AuthRegistryError("unsupported_admin_command")
    except (AuthRegistryError, ValueError, OSError) as exc:
        category = exc.category if isinstance(exc, AuthRegistryError) else type(exc).__name__
        _json({"status": "ERROR", "failure_category": category})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
