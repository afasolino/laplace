"""Codex-side configuration management for the Zetsu pairing layer."""

from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .zetsu_mcp import ZETSU_SCHEMA_VERSION, ZETSU_SKILL_VERSION

ZETSU_CONFIG_VERSION = 4
DEFAULT_ENDPOINT = "http://127.0.0.1:8765/mcp"
DEFAULT_TOKEN_ENV = "LAPLACE_ZETSU_TOKEN"
DEFAULT_TOOLS = (
    "search",
    "get_evidence",
    "project_context",
    "experiment_context",
    "delegate",
    "agent_task",
    "agent_task_status",
    "cancel_agent_task",
    "rtl_task",
    "verify",
    "get_result",
)
_BEGIN = "# BEGIN LAPLACE ZETSU MANAGED v4"
_END = "# END LAPLACE ZETSU MANAGED v4"
_SKILL_MARKER = "<!-- managed-by: laplace-zetsu v4 -->"
_MANAGED_PATTERN = re.compile(
    r"(?ms)^\s*# BEGIN LAPLACE ZETSU MANAGED v[1234]\n"
    r".*?^\s*# END LAPLACE ZETSU MANAGED v[1234]\s*\n?"
)
_SKILL_PATTERN = re.compile(r"<!-- managed-by: laplace-zetsu v[1234] -->")


class ZetsuConfigError(RuntimeError):
    """Zetsu configuration is invalid or cannot be managed safely."""


@dataclass(frozen=True)
class ZetsuStatus:
    configured: bool
    repository: str
    config_path: str
    skill_path: str
    endpoint: str | None
    token_env_var: str | None
    token_available: bool
    skill_installed: bool
    codex_available: bool
    config_version: int | None
    expected_config_version: int
    skill_version: str | None
    expected_skill_version: str
    schema_version: str
    compatible: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _managed_block(endpoint: str, token_env_var: str) -> str:
    tools = ", ".join(_quote(item) for item in DEFAULT_TOOLS)
    return "\n".join(
        (
            _BEGIN,
            "[mcp_servers.zetsu]",
            f"url = {_quote(endpoint)}",
            f"bearer_token_env_var = {_quote(token_env_var)}",
            "enabled = true",
            "required = false",
            f"enabled_tools = [{tools}]",
            'default_tools_approval_mode = "writes"',
            "startup_timeout_sec = 15",
            "tool_timeout_sec = 3700",
            _END,
        )
    )


def _skill_text() -> str:
    return f"""---
name: zetsu
description: Use Laplace for compact knowledge, verified Qwen agent delegation, RTL specialization, and evidence.
---

{_SKILL_MARKER}
# Zetsu

Use Codex local filesystem, shell, Git, builds and tests directly for simple work in the current checkout.
Use `search`, `project_context` or `experiment_context` for indexed, historical, literature or distributed
project knowledge, then expand only needed IDs with `get_evidence`. Use `delegate` for bounded reasoning.
Use `agent_task` for a coherent self-contained repository task when local Qwen can inspect/edit/verify it
more cheaply than repeated Codex orchestration. Batch related reads/edits and avoid polling or narration.
The exposed MCP schema is authoritative. On the normal path, do not grep Zetsu source/configuration,
runtime audit logs, checkpoints, or result artifacts before calling `agent_task`; those are anomaly-only
evidence surfaces and broad inspection defeats delegation token savings.
For a mutating `agent_task`, Codex must choose `verification_argv` before delegation and use a direct
`pytest`, `ruff`, or `mypy` executable accepted by the Zetsu verifier policy; never wrap it with
`python -m`. If one pytest invocation must include unrelated test files with repeated basenames such as
`test_public.py`, add pytest's official `--import-mode=importlib` to prevent import-name collisions.
Keep that verifier bound on resume and set `apply_to_repository=true` for authorized edits.
When a returned promotion is applied, its bound verifier passed after the latest mutation, and there are
no unresolved failures, treat the compact handoff as authoritative: do not reread the edits, inspect
result artifacts, or rerun the same verifier absent an anomaly. Stop the delegated path immediately on
that successful receipt. Exact patches/checkpoints remain persistent and `verify` can expand them only
when anomaly evidence requires it.
Use `rtl_task` only for policy-eligible bounded RTL work handled by CodeV. Never request whole repositories,
papers or logs when compact evidence suffices.
"""


def _paths(repository: Path) -> tuple[Path, Path]:
    configured_home = os.environ.get("CODEX_HOME")
    codex_home = (
        Path(configured_home).expanduser() if configured_home else Path.home() / ".codex"
    ).resolve()
    return (
        codex_home / "config.toml",
        repository / ".agents" / "skills" / "zetsu" / "SKILL.md",
    )


def _replace_managed_block(original: str, block: str | None) -> str:
    matches = list(_MANAGED_PATTERN.finditer(original))
    if len(matches) > 1:
        raise ZetsuConfigError("multiple_zetsu_managed_blocks")
    remaining = _MANAGED_PATTERN.sub("", original).rstrip()
    if block is None:
        return remaining + ("\n" if remaining else "")
    if remaining:
        return remaining + "\n\n" + block + "\n"
    return block + "\n"


def _atomic_text(path: Path, value: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path != "/mcp"
    ):
        raise ZetsuConfigError("zetsu_endpoint_invalid")
    loopback = parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }
    if not loopback and (parsed.scheme != "https" or not parsed.hostname):
        raise ZetsuConfigError("zetsu_endpoint_must_use_https_or_loopback")


def configure(
    repository: Path,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    token_env_var: str = DEFAULT_TOKEN_ENV,
) -> ZetsuStatus:
    """Configure the managed user Codex MCP block plus this repository's local Skill."""

    repo = repository.resolve()
    if not repo.is_dir():
        raise ZetsuConfigError("repository_not_found")
    _validate_endpoint(endpoint)
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", token_env_var):
        raise ZetsuConfigError("token_env_var_invalid")

    config_path, skill_path = _paths(repo)
    original = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    if _managed_section(original) is None and re.search(
        r"(?m)^\s*\[mcp_servers\.zetsu\]\s*$", original
    ):
        raise ZetsuConfigError("zetsu_codex_config_owned_by_user")
    if (
        skill_path.exists()
        and _SKILL_PATTERN.search(skill_path.read_text(encoding="utf-8")) is None
    ):
        raise ZetsuConfigError("zetsu_skill_path_owned_by_user")

    updated = _replace_managed_block(original, _managed_block(endpoint, token_env_var))
    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise ZetsuConfigError("codex_config_toml_invalid") from exc
    _atomic_text(config_path, updated)
    _atomic_text(
        skill_path,
        _skill_text(),
        mode=0o644,
    )
    return status(repo)


def remove(repository: Path) -> ZetsuStatus:
    """Remove only Zetsu-owned Codex configuration and skill content."""

    repo = repository.resolve()
    config_path, skill_path = _paths(repo)
    if (
        skill_path.exists()
        and _SKILL_PATTERN.search(skill_path.read_text(encoding="utf-8")) is None
    ):
        raise ZetsuConfigError("zetsu_skill_path_owned_by_user")
    if config_path.exists():
        updated = _replace_managed_block(config_path.read_text(encoding="utf-8"), None)
        if updated:
            _atomic_text(config_path, updated)
        else:
            config_path.unlink()
            try:
                config_path.parent.rmdir()
            except OSError:
                pass
    if skill_path.exists():
        skill_path.unlink()
        for parent in (skill_path.parent, skill_path.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                break
    return status(repo)


def _managed_section(text: str) -> str | None:
    pattern = re.compile(
        r"(?ms)^\s*# BEGIN LAPLACE ZETSU MANAGED v[1234]\n"
        r"(.*?)^\s*# END LAPLACE ZETSU MANAGED v[1234]"
    )
    match = pattern.search(text)
    return match.group(1) if match else None


def _extract_value(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}\s*=\s*\"([^\"]+)\"\s*$", text)
    return match.group(1) if match else None


def status(repository: Path) -> ZetsuStatus:
    repo = repository.resolve()
    config_path, skill_path = _paths(repo)
    text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    managed = _managed_section(text)
    configured = managed is not None
    endpoint = _extract_value(managed or "", "url") if configured else None
    token_env_var = _extract_value(managed or "", "bearer_token_env_var") if configured else None
    skill_installed = (
        skill_path.exists()
        and _SKILL_PATTERN.search(skill_path.read_text(encoding="utf-8")) is not None
    )
    current_skill = (
        skill_installed
        and skill_path.exists()
        and _SKILL_MARKER in skill_path.read_text(encoding="utf-8")
    )
    config_match = re.search(r"BEGIN LAPLACE ZETSU MANAGED v([1234])", text)
    detected_config_version = int(config_match.group(1)) if config_match else None
    current_config = detected_config_version == ZETSU_CONFIG_VERSION
    skill_text = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
    if _SKILL_MARKER in skill_text:
        detected_skill_version: str | None = ZETSU_SKILL_VERSION
    elif "managed-by: laplace-zetsu v3" in skill_text:
        detected_skill_version = "legacy-v3"
    elif "managed-by: laplace-zetsu v2" in skill_text:
        detected_skill_version = "legacy-v2"
    elif "managed-by: laplace-zetsu v1" in skill_text:
        detected_skill_version = "legacy-v1"
    else:
        detected_skill_version = None
    return ZetsuStatus(
        configured=configured,
        repository=str(repo),
        config_path=str(config_path),
        skill_path=str(skill_path),
        endpoint=endpoint,
        token_env_var=token_env_var,
        token_available=bool(token_env_var and os.environ.get(token_env_var)),
        skill_installed=skill_installed,
        codex_available=shutil.which("codex") is not None,
        config_version=detected_config_version,
        expected_config_version=ZETSU_CONFIG_VERSION,
        skill_version=detected_skill_version,
        expected_skill_version=ZETSU_SKILL_VERSION,
        schema_version=ZETSU_SCHEMA_VERSION,
        compatible=bool(configured and current_config and current_skill),
    )
