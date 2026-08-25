"""Neutral, deterministic repository-agent verification policy."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .repository_authorization import RepositoryAuthorizationError, validate_workspace_path
from .service_tiers import ServiceTierError

_ALLOWED_VERIFY_EXECUTABLES = frozenset({"pytest", "ruff", "mypy"})


def _relative_target(worktree: Path, value: str) -> None:
    normalized = value.replace("\\", "/")
    if normalized == ".git" or normalized.startswith(".git/"):
        raise ServiceTierError("zetsu_agent_git_metadata_forbidden")
    try:
        validate_workspace_path(worktree, normalized)
    except RepositoryAuthorizationError as exc:
        raise ServiceTierError(f"zetsu_agent_{exc.category}", exc.evidence) from exc


def validate_verification_argv(worktree: Path, value: object) -> list[str]:
    """Validate an allowlisted deterministic verifier without executing it."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ServiceTierError("zetsu_agent_verify_argv_invalid")
    argv = [str(item) for item in value]
    if len(argv) > 64 or any(not item or len(item) > 1_000 or "\x00" in item for item in argv):
        raise ServiceTierError("zetsu_agent_verify_argv_invalid")
    executable = Path(argv[0]).name
    if (
        argv[0] != executable
        or "/" in argv[0]
        or "\\" in argv[0]
        or executable not in _ALLOWED_VERIFY_EXECUTABLES
    ):
        raise ServiceTierError("zetsu_agent_verify_command_forbidden")

    lowered = [item.casefold() for item in argv[1:]]
    forbidden_options = {
        "-c",
        "-p",
        "-o",
        "--override-ini",
        "--basetemp",
        "--confcutdir",
        "--config",
        "--config-file",
        "--cache-dir",
        "--custom-typeshed-dir",
        "--python-executable",
        "--junit-xml",
        "--junitxml",
        "--output-file",
        "--pyargs",
        "--rootdir",
    }
    if any(
        item in forbidden_options
        or any(item.startswith(prefix + "=") for prefix in forbidden_options)
        for item in lowered
    ):
        raise ServiceTierError("zetsu_agent_verify_command_forbidden")
    if executable == "ruff" and (
        "--fix" in lowered
        or "--unsafe-fixes" in lowered
        or "format" in lowered
        or (argv[1:] and not argv[1].startswith("-") and argv[1] != "check")
    ):
        raise ServiceTierError("zetsu_agent_verify_command_forbidden")

    skip_next_value = False
    for index, item in enumerate(argv[1:], start=1):
        if skip_next_value:
            skip_next_value = False
            continue
        lowered_item = item.casefold()
        if lowered_item in {"-k", "-m"} and executable == "pytest":
            skip_next_value = True
            continue
        if item.startswith("@"):
            raise ServiceTierError("zetsu_agent_verify_command_forbidden")
        candidate = item.split("::", 1)[0]
        if item.startswith("-"):
            if "=" in item:
                option_value = item.split("=", 1)[1]
                if Path(option_value).is_absolute() or ".." in Path(option_value).parts:
                    raise ServiceTierError("zetsu_agent_verify_path_forbidden")
            continue
        if executable == "ruff" and index == 1 and item == "check":
            continue
        path_value = Path(candidate)
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ServiceTierError("zetsu_agent_verify_path_forbidden")
        if candidate not in {"."} and not (
            "/" in candidate
            or "\\" in candidate
            or path_value.suffix in {".py", ".pyi"}
            or (worktree / candidate).exists()
        ):
            raise ServiceTierError("zetsu_agent_verify_target_invalid")
        if candidate != ".":
            _relative_target(worktree, candidate)
    if skip_next_value:
        raise ServiceTierError("zetsu_agent_verify_argv_invalid")
    return argv
