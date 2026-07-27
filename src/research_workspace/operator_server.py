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
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from .operator_service import OperatorService
from .research_models import LocalDirectoryResearchAdapter, ResearchAdapter
from .research_plane import DeepResearchService


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


def load_or_create_tokens(path: Path) -> tuple[dict[str, str], str | None]:
    """Load mode-0600 role tokens or create them and reveal admin once."""

    if path.is_file():
        if path.stat().st_mode & 0o077:
            raise RuntimeError("Operator authentication file must have mode 0600")
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        tokens = raw.get("tokens") if isinstance(raw, dict) else None
        if not isinstance(tokens, dict) or not all(
            isinstance(token, str) and isinstance(role, str)
            for token, role in tokens.items()
        ):
            raise RuntimeError("Operator authentication file is malformed")
        return dict(tokens), None
    by_role = {
        role: f"laplace-{role}-{secrets.token_urlsafe(32)}"
        for role in ("read", "operate", "approve", "admin")
    }
    token_roles = {token: role for role, token in by_role.items()}
    _atomic_private_json(
        path,
        {
            "schema_version": 1,
            "tokens": token_roles,
            "warning": "Secret role tokens; never include this file in bundles.",
        },
    )
    return token_roles, by_role["admin"]


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
    token_roles, initial_admin = load_or_create_tokens(token_path)
    if initial_admin is not None:
        print("Operator authentication initialized. Admin token (shown once):")
        print(initial_admin)
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

