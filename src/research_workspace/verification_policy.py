"""Neutral, deterministic repository-agent verification policy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from .repository_authorization import RepositoryAuthorizationError, validate_workspace_path
from .service_tiers import ServiceTierError

_ALLOWED_VERIFY_EXECUTABLES = frozenset({"pytest", "ruff", "mypy"})
_MAX_VERIFICATION_STEPS = 16

VerificationStep: TypeAlias = tuple[str, tuple[str, ...]]
VerificationPlan: TypeAlias = tuple[VerificationStep, ...]


def _relative_target(worktree: Path, value: str) -> None:
    normalized = value.replace("\\", "/")
    if normalized == ".git" or normalized.startswith(".git/"):
        raise ServiceTierError("zetsu_agent_git_metadata_forbidden")
    try:
        validate_workspace_path(worktree, normalized)
    except RepositoryAuthorizationError as exc:
        raise ServiceTierError(f"zetsu_agent_{exc.category}", exc.evidence) from exc


def _verification_cwd(worktree: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or len(value) > 500 or "\x00" in value:
        raise ServiceTierError("zetsu_agent_verify_cwd_invalid")
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ServiceTierError("zetsu_agent_verify_cwd_forbidden")
    if normalized == ".":
        return ".", worktree.resolve(strict=True)
    if normalized == ".git" or normalized.startswith(".git/"):
        raise ServiceTierError("zetsu_agent_verify_cwd_forbidden")
    try:
        target = validate_workspace_path(worktree, normalized)
    except RepositoryAuthorizationError as exc:
        raise ServiceTierError(f"zetsu_agent_{exc.category}", exc.evidence) from exc
    if not target.is_dir():
        raise ServiceTierError("zetsu_agent_verify_cwd_invalid")
    return target.relative_to(worktree.resolve(strict=True)).as_posix(), target


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


def verification_qualifies_for_promotion(argv: Sequence[str]) -> bool:
    """Return whether an admitted verifier is an actual read-only final check.

    Admission validates path containment and executable safety. Qualification
    additionally rejects informational and collection-only invocations, which
    cannot establish a promotion binding.
    """

    if not argv:
        return False
    executable = Path(argv[0]).name
    lowered = [item.casefold() for item in argv[1:]]
    informational = {"--help", "-h", "--version", "-v", "--collect-only", "--co", "--setup-plan"}
    if any(item in informational for item in lowered):
        return False
    if executable == "pytest":
        return True
    if executable == "ruff":
        return bool(argv[1:]) and argv[1] == "check" and not any(
            item in {"--show-files", "--show-settings"} for item in lowered
        )
    if executable == "mypy":
        return "--install-types" not in lowered and any(not item.startswith("-") for item in argv[1:])
    return False


def normalize_verification_plan(
    worktree: Path,
    *,
    verification_argv: object = None,
    verification_plan: object = None,
) -> VerificationPlan | None:
    """Normalize one caller-owned, shell-free verification contract."""

    if verification_argv is not None and verification_plan is not None:
        raise ServiceTierError("zetsu_agent_verification_contract_conflict")
    if verification_argv is not None:
        return ((".", tuple(validate_verification_argv(worktree, verification_argv))),)
    if verification_plan is None:
        return None
    if (
        not isinstance(verification_plan, Sequence)
        or isinstance(verification_plan, (str, bytes))
        or not 1 <= len(verification_plan) <= _MAX_VERIFICATION_STEPS
    ):
        raise ServiceTierError("zetsu_agent_verify_plan_invalid")

    normalized: list[VerificationStep] = []
    for raw_step in verification_plan:
        if not isinstance(raw_step, Mapping):
            raise ServiceTierError("zetsu_agent_verify_plan_invalid")
        if set(raw_step) - {"cwd", "argv"} or "argv" not in raw_step:
            raise ServiceTierError("zetsu_agent_verify_plan_invalid")
        cwd, cwd_path = _verification_cwd(worktree, raw_step.get("cwd", "."))
        argv = tuple(validate_verification_argv(cwd_path, raw_step.get("argv")))
        normalized.append((cwd, argv))
    return tuple(normalized)


def verification_plan_to_json(plan: VerificationPlan | None) -> list[dict[str, object]] | None:
    if plan is None:
        return None
    return [{"cwd": cwd, "argv": list(argv)} for cwd, argv in plan]


def verification_plan_digest(plan: VerificationPlan) -> str:
    encoded = json.dumps(
        verification_plan_to_json(plan),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verification_plan_qualifies_for_promotion(plan: VerificationPlan) -> bool:
    return bool(plan) and all(
        verification_qualifies_for_promotion(argv) for _, argv in plan
    )
