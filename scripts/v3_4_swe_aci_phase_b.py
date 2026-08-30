#!/usr/bin/env python3
"""Run Step 3.4 Phase-B paired real-agent A/B against the resident quality model."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from research_workspace.agent_sandbox import (
    AgentSandboxManager,
    AgentToolPolicy,
)
from research_workspace.operator_server import selected_lane_policy
from research_workspace.personal_corpus import PersonalCorpusStore
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.service_tiers import (
    LanePolicy,
    LocalOpenAIChatBackend,
    ModelLane,
    ModelRoute,
    ServiceTierError,
    TierAuditLog,
    TieredServingService,
)
from research_workspace.swe_aci_ab import SWE_AGENT_REFERENCE_COMMIT
from research_workspace.swe_aci_phase_b import (
    LAPLACE_V34_PHASE_B_BASELINE,
    ACITrace,
    AgentTaskRecord,
    PhaseBTask,
    RecordingBaselineCoordinator,
    RecordingCandidateCoordinator,
    assess_phase_b,
    load_phase_b_tasks,
)
from research_workspace.upstream_ab import manifest_sha256
from research_workspace.user_capabilities import (
    Capability,
    CapabilityTier,
    UserCapabilityStore,
)
from research_workspace.zetsu_agent import ZetsuAgentCoordinator

JsonObject: TypeAlias = dict[str, object]
USER_ID = "v34-phase-b"
_FIXED_GIT_DATE = "2026-08-27T00:00:00+00:00"


class AgentBackendLike(Protocol):
    def run(self, **kwargs: object) -> JsonObject: ...


class _NeverAgentBackend:
    def run(self, **kwargs: object) -> JsonObject:
        del kwargs
        raise AssertionError("legacy agent backend must not execute Phase B")


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:600]
        raise RuntimeError(f"v34_phase_b_git_failed:{' '.join(args)}:{detail}")
    return result.stdout.strip()


def repository_tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(root).parts
        and "__pycache__" not in path.parts
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _long_module(
    *,
    prefix: str,
    marker_line: int,
    marker_name: str,
    marker_value: str,
    total: int,
) -> str:
    lines = [f'# controlled v34 Phase-B fixture: {prefix}']
    for line_number in range(2, total + 1):
        if line_number == marker_line:
            lines.append(f'{marker_name} = "{marker_value}"')
        else:
            lines.append(f"{prefix}_padding_{line_number:03d} = {line_number}")
    return "\n".join(lines) + "\n"


def _test_body(task: PhaseBTask) -> str:
    if task.task_id == "mutate-scale":
        return (
            "from src.edit_target import scale\n\n"
            "def test_target() -> None:\n"
            "    assert scale(4) == 12\n"
        )
    if task.task_id == "mutate-offset":
        return (
            "from src.edit_target import compute_offset\n\n"
            "def test_target() -> None:\n"
            "    assert compute_offset() == 7\n"
        )
    if task.task_id == "mutate-label":
        return (
            "from src.edit_target import format_label\n\n"
            "def test_target() -> None:\n"
            '    assert format_label("x") == "new:x"\n'
        )
    if task.task_id == "mutate-transform":
        return (
            "from src.edit_target import transform\n\n"
            "def test_target() -> None:\n"
            "    assert transform(3) == 10\n"
        )
    return (
        "from src.edit_target import compute_offset\n\n"
        "def test_fixture_imports() -> None:\n"
        "    assert compute_offset() == 4\n"
    )


def make_fixture(root: Path, task: PhaseBTask) -> tuple[str, str]:
    shutil.rmtree(root, ignore_errors=True)
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src/__init__.py").write_text("", encoding="utf-8")

    alpha = _long_module(
        prefix="alpha",
        marker_line=22,
        marker_name="HEAD_SECRET",
        marker_value="alpha-head-a119",
        total=225,
    )
    alpha_lines = alpha.splitlines()
    alpha_lines[114] = 'MIDPOINT_SECRET = "alpha-mid-7f3c"'
    alpha_lines[194] = 'TAIL_SECRET = "alpha-tail-91bd"'
    (root / "src/long_alpha.py").write_text(
        "\n".join(alpha_lines) + "\n",
        encoding="utf-8",
    )
    (root / "src/long_beta.py").write_text(
        _long_module(
            prefix="beta",
            marker_line=76,
            marker_name="BETA_SECRET",
            marker_value="beta-secret-c248",
            total=180,
        ),
        encoding="utf-8",
    )
    (root / "src/search_one.py").write_text(
        'shared_locator = "one"\n',
        encoding="utf-8",
    )
    (root / "src/search_two.py").write_text(
        'shared_locator = "two"\n\n'
        "def special_search_function() -> str:\n"
        '    return "function-result-84ce"\n',
        encoding="utf-8",
    )
    (root / "src/search_three.py").write_text(
        'shared_locator = "three"\n'
        'UNIQUE_SEARCH_TOKEN = "unique-search-5aa2"\n',
        encoding="utf-8",
    )
    for index in range(6):
        (root / f"src/decoy_{index}.py").write_text(
            f'DECOY_{index} = "not-the-target-{index}"\n',
            encoding="utf-8",
        )
    (root / "src/edit_target.py").write_text(
        "SCALE = 2\n\n"
        "def scale(value: int) -> int:\n"
        "    return value * SCALE\n\n"
        "def compute_offset() -> int:\n"
        "    return 4\n\n"
        "def format_label(value: str) -> str:\n"
        '    return f"old:{value}"\n\n'
        "def transform(value: int) -> int:\n"
        "    return (value + 1) * 2\n",
        encoding="utf-8",
    )
    (root / "tests/test_fixture.py").write_text(
        _test_body(task),
        encoding="utf-8",
    )

    git(root, "init", "-q")
    git(root, "config", "user.name", "Laplace v3.4 Phase B")
    git(root, "config", "user.email", "v34-phase-b@example.invalid")
    git(root, "add", "--all")
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = _FIXED_GIT_DATE
    env["GIT_COMMITTER_DATE"] = _FIXED_GIT_DATE
    committed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "commit",
            "-qm",
            "controlled v34 phase-b fixture",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if committed.returncode != 0:
        raise RuntimeError(
            "v34_phase_b_fixture_commit_failed:"
            + (committed.stderr or committed.stdout).strip()[:600]
        )
    revision = git(root, "rev-parse", "HEAD")
    return revision, repository_tree_sha(root)


def _validate_phase_a(repo: Path) -> str:
    path = repo / ".runtime/v34/phase-a/result.json"
    if not path.is_file():
        raise RuntimeError("v34_phase_b_phase_a_evidence_missing")
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("v34_phase_b_phase_a_evidence_invalid")
    data = cast(dict[str, object], raw)
    assessment = data.get("assessment")
    if not isinstance(assessment, dict):
        raise TypeError("v34_phase_b_phase_a_assessment_missing")
    expected = {
        "assessment": "PROMISING",
        "governance_safe": True,
        "promising_primitives": ["view", "search", "syntax_preflight"],
        "final_adoption_decision": None,
    }
    for key, value in expected.items():
        if assessment.get(key) != value:
            raise RuntimeError(
                f"v34_phase_b_phase_a_{key}_mismatch:"
                f"expected={value!r}:actual={assessment.get(key)!r}"
            )
    if data.get("laplace_revision") != LAPLACE_V34_PHASE_B_BASELINE:
        raise RuntimeError("v34_phase_b_phase_a_laplace_revision_mismatch")
    if data.get("swe_agent_revision") != SWE_AGENT_REFERENCE_COMMIT:
        raise RuntimeError("v34_phase_b_phase_a_swe_revision_mismatch")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality_route(repo: Path) -> tuple[LanePolicy, ModelRoute, str]:
    policy = selected_lane_policy(repo)
    route = policy.routes.get(ModelLane.QUALITY)
    if route is None:
        raise RuntimeError("v34_phase_b_quality_route_missing")
    if not route.endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise RuntimeError("v34_phase_b_quality_endpoint_not_loopback")
    selected = repo / "configs/selected_serving_profiles.json"
    selected_sha = (
        hashlib.sha256(selected.read_bytes()).hexdigest()
        if selected.is_file()
        else "not-present"
    )
    return policy, route, selected_sha


def _phase_b_repo_id(task: PhaseBTask, arm: str) -> str:
    return f"v34-{task.task_id}-{arm}"


def _service(
    *,
    repo: Path,
    task: PhaseBTask,
    arm: str,
    runtime: Path,
    source_repo: Path,
    base_revision: str,
    trace: ACITrace,
) -> tuple[TieredServingService, ZetsuAgentCoordinator]:
    state = runtime / "state" / task.task_id / arm
    state.mkdir(parents=True, exist_ok=True)
    authorizations = RepositoryAuthorizationStore(state / "repositories.sqlite3")
    repo_id = _phase_b_repo_id(task, arm)
    authorizations.register(repo_id, source_repo)
    authorizations.grant(
        USER_ID,
        repo_id,
        base_revision=base_revision,
    )

    users = UserCapabilityStore(state / "users.sqlite3")
    users.set_user(
        USER_ID,
        CapabilityTier.PLUS,
        capabilities=frozenset({Capability.CHAT, Capability.AGENT}),
    )
    sandboxes = AgentSandboxManager(state / "worktrees", authorizations)
    policy, _route, _selected_sha = _quality_route(repo)
    backend = LocalOpenAIChatBackend(timeout_seconds=300.0)
    tiered = TieredServingService(
        users=users,
        sandboxes=sandboxes,
        lane_policy=policy,
        chat_backend=backend,
        agent_backend=cast(AgentBackendLike, _NeverAgentBackend()),
        audit_log=TierAuditLog(state / "tier-audit.jsonl"),
    )
    corpus = PersonalCorpusStore(state / "corpus")
    coordinator_cls = (
        RecordingCandidateCoordinator
        if arm == "candidate"
        else RecordingBaselineCoordinator
    )
    coordinator = coordinator_cls(
        tiered,
        corpus,
        trace=trace,
    )
    return tiered, coordinator


def _expected_fraction(observed: Sequence[str], expected: Sequence[str]) -> float:
    if not expected:
        return 1.0
    observed_set = set(observed)
    return len(observed_set & set(expected)) / len(set(expected))


def _relevance(observed: Sequence[str], expected: Sequence[str]) -> float:
    if not expected:
        return 1.0 if not observed else 0.0
    observed_set = set(observed)
    if not observed_set:
        return 0.0
    return len(observed_set & set(expected)) / len(observed_set)


def _telemetry(result: Mapping[str, object]) -> tuple[int, int, int, bool]:
    raw = result.get("telemetry")
    if not isinstance(raw, Mapping):
        return 0, 0, 0, False

    def integer(key: str) -> int:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value

    input_tokens = integer("qwen_input_tokens")
    output_tokens = integer("qwen_output_tokens")
    tool_rounds = integer("tool_calls")
    calls = integer("qwen_calls")
    reported_calls = integer("qwen_usage_reported_calls")
    source = raw.get("qwen_token_usage_source")
    complete_flag = raw.get("qwen_token_usage_complete")
    complete = (
        calls > 0
        and reported_calls == calls
        and source == "model_reported_per_request"
        and complete_flag is True
    )
    return input_tokens, output_tokens, tool_rounds, complete



def _failure_checkpoint(
    coordinator: ZetsuAgentCoordinator,
) -> JsonObject:
    paths = sorted(coordinator.checkpoints.root.glob("*.json"))
    if len(paths) != 1:
        raise RuntimeError(
            f"v34_phase_b_failure_checkpoint_count_invalid:{len(paths)}"
        )
    try:
        raw: object = json.loads(paths[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("v34_phase_b_failure_checkpoint_invalid") from exc
    if not isinstance(raw, dict):
        raise TypeError("v34_phase_b_failure_checkpoint_not_object")
    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise TypeError("v34_phase_b_failure_session_id_missing")
    checkpoint = coordinator.checkpoints.read(session_id)
    if checkpoint is None:
        raise RuntimeError("v34_phase_b_failure_checkpoint_missing")
    return checkpoint


_RECORD_KEYS = {
    "task_id",
    "kind",
    "correct",
    "input_tokens",
    "output_tokens",
    "tool_rounds",
    "wall_seconds",
    "failed",
    "verifier_passed",
    "repository_coverage",
    "relevance",
    "usage_complete",
    "security_preserved",
    "model_id",
    "fixture_sha256",
}


def _record_from_mapping(
    value: object,
    *,
    task: PhaseBTask,
) -> AgentTaskRecord:
    if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
        raise TypeError("v34_phase_b_partial_record_schema_invalid")

    def boolean(key: str) -> bool:
        raw = value.get(key)
        if not isinstance(raw, bool):
            raise TypeError(f"v34_phase_b_partial_{key}_invalid")
        return raw

    def integer(key: str) -> int:
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise TypeError(f"v34_phase_b_partial_{key}_invalid")
        return raw

    def number(key: str, *, unit_interval: bool = False) -> float:
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"v34_phase_b_partial_{key}_invalid")
        result = float(raw)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"v34_phase_b_partial_{key}_invalid")
        if unit_interval and result > 1.0:
            raise ValueError(f"v34_phase_b_partial_{key}_invalid")
        return result

    task_id = value.get("task_id")
    kind = value.get("kind")
    model_id = value.get("model_id")
    fixture_sha256 = value.get("fixture_sha256")
    verifier_raw = value.get("verifier_passed")
    if task_id != task.task_id or kind != task.kind:
        raise ValueError("v34_phase_b_partial_task_prefix_mismatch")
    if not isinstance(model_id, str) or not model_id:
        raise TypeError("v34_phase_b_partial_model_id_invalid")
    if (
        not isinstance(fixture_sha256, str)
        or len(fixture_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in fixture_sha256)
    ):
        raise ValueError("v34_phase_b_partial_fixture_sha256_invalid")
    if verifier_raw is not None and not isinstance(verifier_raw, bool):
        raise TypeError("v34_phase_b_partial_verifier_passed_invalid")

    return AgentTaskRecord(
        task_id=task.task_id,
        kind=task.kind,
        correct=boolean("correct"),
        input_tokens=integer("input_tokens"),
        output_tokens=integer("output_tokens"),
        tool_rounds=integer("tool_rounds"),
        wall_seconds=number("wall_seconds"),
        failed=boolean("failed"),
        verifier_passed=verifier_raw,
        repository_coverage=number(
            "repository_coverage",
            unit_interval=True,
        ),
        relevance=number("relevance", unit_interval=True),
        usage_complete=boolean("usage_complete"),
        security_preserved=boolean("security_preserved"),
        model_id=model_id,
        fixture_sha256=fixture_sha256,
    )


def _load_partial(
    path: Path,
    *,
    tasks: Sequence[PhaseBTask],
    current_model_id: str,
    phase_a_sha: str,
    task_manifest_sha: str,
    selected_profile_sha: str,
) -> tuple[list[AgentTaskRecord], list[AgentTaskRecord], str]:
    if not path.is_file():
        return [], [], "NONE"

    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("v34_phase_b_partial_invalid") from exc
    if not isinstance(raw, dict):
        raise TypeError("v34_phase_b_partial_not_object")
    schema = raw.get("schema")
    if schema not in {
        "laplace.v34.swe_aci_phase_b_partial.v1",
        "laplace.v34.swe_aci_phase_b_partial.v2",
    }:
        raise ValueError("v34_phase_b_partial_schema_invalid")

    completed = raw.get("completed_pairs")
    before_raw = raw.get("baseline_records")
    after_raw = raw.get("candidate_records")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 <= completed <= len(tasks)
        or not isinstance(before_raw, list)
        or not isinstance(after_raw, list)
        or len(before_raw) != completed
        or len(after_raw) != completed
    ):
        raise ValueError("v34_phase_b_partial_shape_invalid")

    if schema == "laplace.v34.swe_aci_phase_b_partial.v2":
        expected_metadata = {
            "laplace_revision": LAPLACE_V34_PHASE_B_BASELINE,
            "swe_agent_revision": SWE_AGENT_REFERENCE_COMMIT,
            "phase_a_result_sha256": phase_a_sha,
            "task_manifest_sha256": task_manifest_sha,
            "selected_profile_sha256": selected_profile_sha,
            "quality_model_id": current_model_id,
        }
        for key, expected in expected_metadata.items():
            if raw.get(key) != expected:
                raise ValueError(
                    f"v34_phase_b_partial_metadata_mismatch:{key}"
                )

    baseline: list[AgentTaskRecord] = []
    candidate: list[AgentTaskRecord] = []
    for index in range(completed):
        task = tasks[index]
        before = _record_from_mapping(before_raw[index], task=task)
        after = _record_from_mapping(after_raw[index], task=task)
        if before.fixture_sha256 != after.fixture_sha256:
            raise ValueError(
                f"v34_phase_b_partial_fixture_mismatch:{task.task_id}"
            )
        if before.model_id != current_model_id or after.model_id != current_model_id:
            raise ValueError(
                f"v34_phase_b_partial_model_mismatch:{task.task_id}"
            )
        if not before.usage_complete or not after.usage_complete:
            raise ValueError(
                f"v34_phase_b_partial_usage_incomplete:{task.task_id}"
            )
        baseline.append(before)
        candidate.append(after)

    return baseline, candidate, str(schema)


def _write_partial(
    path: Path,
    *,
    baseline: Sequence[AgentTaskRecord],
    candidate: Sequence[AgentTaskRecord],
    phase_a_sha: str,
    task_manifest_sha: str,
    selected_profile_sha: str,
    quality_model_id: str,
) -> None:
    payload: JsonObject = {
        "schema": "laplace.v34.swe_aci_phase_b_partial.v2",
        "laplace_revision": LAPLACE_V34_PHASE_B_BASELINE,
        "swe_agent_revision": SWE_AGENT_REFERENCE_COMMIT,
        "phase_a_result_sha256": phase_a_sha,
        "task_manifest_sha256": task_manifest_sha,
        "selected_profile_sha256": selected_profile_sha,
        "quality_model_id": quality_model_id,
        "completed_pairs": len(baseline),
        "baseline_records": [item.as_json() for item in baseline],
        "candidate_records": [item.as_json() for item in candidate],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _oracle_ok(task: PhaseBTask, worktree: Path) -> bool:
    if task.oracle is None:
        return True
    path = task.oracle.get("path")
    required = task.oracle.get("required")
    forbidden = task.oracle.get("forbidden")
    if (
        not isinstance(path, str)
        or not isinstance(required, list)
        or not isinstance(forbidden, list)
    ):
        raise TypeError("v34_phase_b_oracle_schema_invalid")
    text = (worktree / path).read_text(encoding="utf-8")
    return all(
        isinstance(value, str) and value in text for value in required
    ) and all(
        isinstance(value, str) and value not in text for value in forbidden
    )


def _independent_verifier(worktree: Path, task: PhaseBTask) -> bool:
    if task.verification_argv is None:
        return True
    result = subprocess.run(
        list(task.verification_argv),
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return result.returncode == 0


def _run_arm(
    *,
    repo: Path,
    runtime: Path,
    task: PhaseBTask,
    arm: str,
) -> AgentTaskRecord:
    state_root = runtime / "state" / task.task_id / arm
    shutil.rmtree(state_root, ignore_errors=True)
    source_repo = runtime / "fixtures" / task.task_id / arm / "source"
    base_revision, fixture_sha = make_fixture(source_repo, task)
    trace = ACITrace()
    tiered, coordinator = _service(
        repo=repo,
        task=task,
        arm=arm,
        runtime=runtime,
        source_repo=source_repo,
        base_revision=base_revision,
        trace=trace,
    )
    repo_id = _phase_b_repo_id(task, arm)

    run_signature = inspect.signature(coordinator.run)
    if "allow_mutation" not in run_signature.parameters:
        raise RuntimeError("v34_phase_b_transient_mutation_parameter_missing")

    started = time.perf_counter()
    try:
        result = coordinator.run(
            user_id=USER_ID,
            repo_id=repo_id,
            instruction=task.instruction,
            lane=ModelLane.QUALITY,
            session_id=None,
            max_steps=6,
            max_chars=6000,
            verification_argv=task.verification_argv,
            wait_timeout_seconds=300.0,
            allow_mutation=task.mutation_expected,
            apply_to_repository=False,
            persistent_session=True,
        )
    except ServiceTierError as exc:
        checkpoint = _failure_checkpoint(coordinator)
        telemetry = checkpoint.get("telemetry")
        session_id = checkpoint.get("session_id")
        model_id = checkpoint.get("model_id")
        if (
            not isinstance(telemetry, Mapping)
            or not isinstance(session_id, str)
            or not isinstance(model_id, str)
        ):
            raise TypeError(
                "v34_phase_b_failure_checkpoint_fields_invalid"
            ) from exc
        result = {
            "status": "FAILED",
            "session_id": session_id,
            "repo_id": repo_id,
            "model_id": model_id,
            "content": "",
            "telemetry": dict(telemetry),
            "failure_category": exc.category,
        }
        print(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "arm": arm,
                    "agent_failure_category": exc.category,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    if not isinstance(result, dict):
        raise TypeError("v34_phase_b_agent_result_invalid")
    session_id = result.get("session_id")
    if not isinstance(session_id, str):
        raise TypeError("v34_phase_b_session_id_missing")
    binding = tiered.sandboxes.require_active(session_id, user_id=USER_ID)
    worktree = Path(binding.worktree_root).resolve(strict=True)
    sandbox_root = Path(tiered.sandboxes.sandbox_root).resolve(strict=True)
    try:
        worktree.relative_to(sandbox_root)
        contained = True
    except ValueError:
        contained = False

    content = result.get("content")
    rendered = content if isinstance(content, str) else ""
    answer_ok = all(
        term.casefold() in rendered.casefold()
        for term in task.expected_answer_terms
    )
    oracle_ok = _oracle_ok(task, worktree)
    independent_verified = _independent_verifier(worktree, task)
    verifier_passed = (
        independent_verified
        if task.mutation_expected
        else None
    )
    status = result.get("status")
    correct = (
        status == "SUCCESS"
        and oracle_ok
        and (answer_ok if task.kind != "mutation" else True)
        and (verifier_passed is True if task.mutation_expected else True)
    )
    input_tokens, output_tokens, tool_rounds, usage_complete = _telemetry(result)
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        model_id = "INVALID"

    record = AgentTaskRecord(
        task_id=task.task_id,
        kind=task.kind,
        correct=bool(correct),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_rounds=tool_rounds,
        wall_seconds=elapsed,
        failed=status != "SUCCESS",
        verifier_passed=verifier_passed,
        repository_coverage=_expected_fraction(
            trace.paths,
            task.expected_paths,
        ),
        relevance=_relevance(trace.paths, task.expected_paths),
        usage_complete=usage_complete,
        security_preserved=contained,
        model_id=model_id,
        fixture_sha256=fixture_sha,
    )
    return record


def _preflight(repo: Path, runtime: Path, task: PhaseBTask) -> JsonObject:
    phase_a_sha = _validate_phase_a(repo)
    policy, route, selected_sha = _quality_route(repo)
    del policy

    parameters = inspect.signature(RecordingBaselineCoordinator.run).parameters
    required = {
        "allow_mutation",
        "persistent_session",
        "apply_to_repository",
        "verification_argv",
        "wait_timeout_seconds",
    }
    missing = sorted(required - set(parameters))
    if missing:
        raise RuntimeError(f"v34_phase_b_run_signature_missing:{missing}")

    preflight_runtime = runtime / "preflight-runtime"
    shutil.rmtree(preflight_runtime, ignore_errors=True)
    source = runtime / "preflight" / "source"
    revision, fixture_sha = make_fixture(source, task)
    trace = ACITrace()
    tiered, _coordinator = _service(
        repo=repo,
        task=task,
        arm="baseline",
        runtime=preflight_runtime,
        source_repo=source,
        base_revision=revision,
        trace=trace,
    )
    binding = tiered.sandboxes.create(
        user_id=USER_ID,
        repo_id=_phase_b_repo_id(task, "baseline"),
        session_id="v34-phase-b-preflight",
        tool_policy=AgentToolPolicy(
            policy_id="v34-phase-b-preflight",
            allowed_tools=("read_file",),
            network_enabled=False,
            max_commands=2,
            max_wall_seconds=60,
        ),
        task_title="Step 3.4 Phase B preflight",
    )
    worktree = Path(binding.worktree_root).resolve(strict=True)
    worktree.relative_to(Path(tiered.sandboxes.sandbox_root).resolve(strict=True))
    return {
        "status": "PASS",
        "laplace_revision": LAPLACE_V34_PHASE_B_BASELINE,
        "swe_agent_revision": SWE_AGENT_REFERENCE_COMMIT,
        "phase_a_result_sha256": phase_a_sha,
        "quality_model_id": route.model_id,
        "quality_endpoint": route.endpoint,
        "selected_profile_sha256": selected_sha,
        "fixture_sha256": fixture_sha,
        "fixture_revision": revision,
        "authorized_worktree": str(worktree),
        "transient_mutation_parameter": True,
        "production_wiring_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("benchmarks/v34_swe_agent_aci/tasks_phase_b.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runtime/v34/phase-b/result.json"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-info-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    head = git(repo, "rev-parse", "HEAD")
    if head != LAPLACE_V34_PHASE_B_BASELINE:
        raise RuntimeError("v34_phase_b_laplace_revision_mismatch")
    tasks_path = args.tasks if args.tasks.is_absolute() else repo / args.tasks
    tasks = load_phase_b_tasks(tasks_path)
    phase_a_sha = _validate_phase_a(repo)
    runtime = repo / ".runtime/v34/phase-b"
    runtime.mkdir(parents=True, exist_ok=True)

    if args.preflight_only:
        print(json.dumps(_preflight(repo, runtime, tasks[0]), indent=2, sort_keys=True))
        return 0

    _preflight(repo, runtime, tasks[0])
    _policy, route, selected_sha = _quality_route(repo)
    manifest_sha = manifest_sha256(tasks_path)
    partial_path = runtime / "result.partial.json"
    baseline, candidate, partial_schema = _load_partial(
        partial_path,
        tasks=tasks,
        current_model_id=route.model_id,
        phase_a_sha=phase_a_sha,
        task_manifest_sha=manifest_sha,
        selected_profile_sha=selected_sha,
    )
    completed_pairs = len(baseline)
    if args.resume_info_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "partial_schema": partial_schema,
                    "completed_pairs": completed_pairs,
                    "remaining_pairs": len(tasks) - completed_pairs,
                    "next_task_id": (
                        tasks[completed_pairs].task_id
                        if completed_pairs < len(tasks)
                        else None
                    ),
                    "quality_model_id": route.model_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if completed_pairs:
        print(
            json.dumps(
                {
                    "resume": True,
                    "partial_schema": partial_schema,
                    "completed_pairs": completed_pairs,
                    "remaining_pairs": len(tasks) - completed_pairs,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for index, task in enumerate(
        tasks[completed_pairs:],
        start=completed_pairs,
    ):
        arm_order = (
            ("baseline", "candidate")
            if index % 2 == 0
            else ("candidate", "baseline")
        )
        pair: dict[str, AgentTaskRecord] = {}
        for arm in arm_order:
            record = _run_arm(
                repo=repo,
                runtime=runtime,
                task=task,
                arm=arm,
            )
            pair[arm] = record
            print(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "arm": arm,
                        "correct": record.correct,
                        "input_tokens": record.input_tokens,
                        "output_tokens": record.output_tokens,
                        "tool_rounds": record.tool_rounds,
                        "usage_complete": record.usage_complete,
                        "wall_seconds": round(record.wall_seconds, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        before = pair["baseline"]
        after = pair["candidate"]
        if before.fixture_sha256 != after.fixture_sha256:
            raise RuntimeError(
                f"v34_phase_b_fixture_pair_mismatch:{task.task_id}"
            )
        baseline.append(before)
        candidate.append(after)

        _write_partial(
            partial_path,
            baseline=baseline,
            candidate=candidate,
            phase_a_sha=phase_a_sha,
            task_manifest_sha=manifest_sha,
            selected_profile_sha=selected_sha,
            quality_model_id=route.model_id,
        )

    decision = assess_phase_b(baseline, candidate)
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result: JsonObject = {
        "schema": "laplace.v34.swe_aci_phase_b_results.v1",
        "phase": "B_REAL_AGENT",
        "laplace_revision": LAPLACE_V34_PHASE_B_BASELINE,
        "swe_agent_revision": SWE_AGENT_REFERENCE_COMMIT,
        "phase_a_result_sha256": phase_a_sha,
        "task_manifest_sha256": manifest_sha,
        "selected_profile_sha256": selected_sha,
        "quality_model_id": route.model_id,
        "quality_endpoint": route.endpoint,
        "arm_order_policy": "alternating_baseline_candidate_by_task_index",
        "token_source": "model_reported_per_request",
        "baseline_records": [item.as_json() for item in baseline],
        "candidate_records": [item.as_json() for item in candidate],
        "decision": decision,
        "production_wiring_changed": False,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"output={output}")
    return 2 if decision.get("decision") == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
