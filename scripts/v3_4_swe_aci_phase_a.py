#!/usr/bin/env python3
"""Run the reproducible Step 3.4 SWE-agent ACI primitive A/B."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, median
from typing import TypeAlias, cast

from research_workspace.bounded_aci import BoundedACIError, BoundedRepositoryACI
from research_workspace.swe_aci_ab import (
    LAPLACE_V34_BASELINE,
    PHASE_A_REPETITIONS,
    SWE_AGENT_REFERENCE_COMMIT,
    PrimitiveTask,
    PrimitiveTrial,
    SweAgentACIShadow,
    assess_phase_a,
    load_phase_a_tasks,
)
from research_workspace.upstream_ab import char4_tokens, manifest_sha256

JsonObject: TypeAlias = dict[str, object]


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise RuntimeError(f"v34_fixture_git_failed:{' '.join(args)}:{detail}")
    return result.stdout.strip()


def make_fixture(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()

    alpha_lines = [
        "# v34 controlled fixture",
        "ALPHA_TOP = 'present'",
        "",
        "def alpha_helper() -> str:",
        "    return ALPHA_TOP",
    ]
    alpha_lines.extend(f"alpha_padding_{index:03d} = {index}" for index in range(6, 60))
    alpha_lines.extend(
        [
            "ALPHA_MIDDLE = 'present'",
            "",
            "def alpha_middle() -> str:",
            "    return ALPHA_MIDDLE",
            "",
            "shared_target = alpha_middle",
        ]
    )
    alpha_lines.extend(f"alpha_tail_{index:03d} = {index}" for index in range(67, 221))
    (root / "src/alpha.py").write_text(
        "\n".join(alpha_lines) + "\n",
        encoding="utf-8",
    )

    beta_lines = [
        "BETA_TOP = 'present'",
        "",
        "def beta_helper() -> str:",
        "    return BETA_TOP",
        "",
        "shared_target = beta_helper",
    ]
    beta_lines.extend(f"beta_padding_{index:03d} = {index}" for index in range(7, 181))
    (root / "src/beta.py").write_text(
        "\n".join(beta_lines) + "\n",
        encoding="utf-8",
    )

    (root / "src/gamma.py").write_text(
        "UNIQUE_MARKER = 'gamma'\n"
        "shared_target = UNIQUE_MARKER\n",
        encoding="utf-8",
    )
    (root / "src/edit_target.py").write_text(
        "VALUE = 1\n\n"
        "def read_value() -> int:\n"
        "    return VALUE\n",
        encoding="utf-8",
    )
    (root / "tests/test_fixture.py").write_text(
        "from src.alpha import alpha_middle, shared_target\n\n"
        "def test_fixture() -> None:\n"
        "    assert alpha_middle()\n"
        "    assert shared_target\n",
        encoding="utf-8",
    )
    (root / "docs.txt").write_text(
        "controlled benchmark fixture\n",
        encoding="utf-8",
    )

    git(root, "init", "-q")
    git(root, "config", "user.name", "Laplace v3.4 Benchmark")
    git(root, "config", "user.email", "v34@example.invalid")
    git(root, "add", "--all")
    git(root, "commit", "-qm", "v34 fixture")
    outside = root.parent / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")


def encoded(value: object) -> str:
    if isinstance(value, Mapping):
        return json.dumps(
            dict(value),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return str(value)


def expected_path_hits(text: str, expected: tuple[str, ...]) -> float:
    if not expected:
        return 1.0
    return sum(path in text for path in expected) / len(expected)


def expected_term_hits(text: str, expected: tuple[str, ...]) -> float:
    if not expected:
        return 1.0
    lowered = text.casefold()
    return sum(term.casefold() in lowered for term in expected) / len(expected)


def verify_python_fixture(repo: Path) -> bool:
    python = Path(__file__).resolve().parents[1] / ".venv/bin/python"
    compile_result = subprocess.run(
        [str(python), "-m", "py_compile", "src/edit_target.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    diff_result = subprocess.run(
        ["git", "diff", "--check"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return compile_result.returncode == 0 and diff_result.returncode == 0


def execute(
    aci: BoundedRepositoryACI,
    task: PrimitiveTask,
    repo: Path,
) -> tuple[object, bool, bool, bool]:
    """Return observation, expected-task correctness, verifier success, security."""
    operation = task.operation
    payload = task.payload

    try:
        if operation == "read":
            result = aci.read_region(
                path=cast(str, payload["path"]),
                start_line=cast(int, payload["start_line"]),
                end_line=cast(int, payload["end_line"]),
            )
            rendered = encoded(result)
            correct = (
                expected_term_hits(rendered, task.expected_terms) == 1.0
                and expected_path_hits(rendered, task.expected_paths) == 1.0
            )
            return result, correct, True, True

        if operation in {"search", "search_no_match"}:
            result = aci.search_text(
                query=cast(str, payload["query"]),
                glob=cast(str, payload["glob"]),
            )
            rendered = encoded(result)
            if operation == "search_no_match":
                raw_matches = result.get("matches")
                files = result.get("files")
                correct = (
                    (isinstance(raw_matches, list) and not raw_matches)
                    or (isinstance(files, list) and not files)
                )
            else:
                correct = expected_path_hits(
                    rendered,
                    task.expected_paths,
                ) == 1.0
            return result, correct, True, True

        if operation == "edit_valid":
            result = aci.edit_region(
                path=cast(str, payload["path"]),
                old_text=cast(str, payload["old_text"]),
                new_text=cast(str, payload["new_text"]),
            )
            current = (repo / cast(str, payload["path"])).read_text(
                encoding="utf-8"
            )
            verified = verify_python_fixture(repo)
            correct = (
                cast(str, payload["new_text"]) in current and verified
            )
            return result, correct, verified, True

        if operation == "edit_invalid_python":
            before = (repo / cast(str, payload["path"])).read_text(
                encoding="utf-8"
            )
            try:
                result = aci.edit_region(
                    path=cast(str, payload["path"]),
                    old_text=cast(str, payload["old_text"]),
                    new_text=cast(str, payload["new_text"]),
                )
            except BoundedACIError as exc:
                after = (repo / cast(str, payload["path"])).read_text(
                    encoding="utf-8"
                )
                verified = verify_python_fixture(repo)
                correct = (
                    exc.category == "aci_swe_python_syntax_invalid"
                    and before == after
                    and verified
                )
                return {
                    "error": exc.category,
                    "evidence": exc.evidence,
                }, correct, verified, True
            verified = verify_python_fixture(repo)
            return result, False, verified, True

        if operation == "mutation_denied":
            before = (repo / cast(str, payload["path"])).read_text(
                encoding="utf-8"
            )
            try:
                result = aci.edit_region(
                    path=cast(str, payload["path"]),
                    old_text=cast(str, payload["old_text"]),
                    new_text=cast(str, payload["new_text"]),
                )
            except BoundedACIError as exc:
                after = (repo / cast(str, payload["path"])).read_text(
                    encoding="utf-8"
                )
                correct = (
                    exc.category == "aci_mutation_not_allowed"
                    and before == after
                )
                return {"error": exc.category}, correct, True, correct
            return result, False, True, False

        if operation == "path_escape":
            try:
                result = aci.read_region(
                    path=cast(str, payload["path"]),
                    start_line=1,
                    end_line=1,
                )
            except BoundedACIError as exc:
                correct = exc.category == "aci_path_escape"
                return {"error": exc.category}, correct, True, correct
            return result, False, True, False

        if operation == "git_metadata":
            try:
                result = aci.read_region(
                    path=cast(str, payload["path"]),
                    start_line=1,
                    end_line=1,
                )
            except BoundedACIError as exc:
                correct = exc.category == "aci_git_metadata_forbidden"
                return {"error": exc.category}, correct, True, correct
            return result, False, True, False

        raise RuntimeError(f"v34_phase_a_unknown_operation:{operation}")
    except BoundedACIError as exc:
        return {
            "unexpected_error": exc.category,
            "evidence": exc.evidence,
        }, False, False, False


def one_arm(
    *,
    task: PrimitiveTask,
    provider: str,
    root: Path,
) -> PrimitiveTrial:
    outputs: list[str] = []
    correctness: list[float] = []
    verifier: list[float] = []
    security: list[bool] = []
    walls: list[float] = []
    failures: list[float] = []
    coverage: list[float] = []
    relevance: list[float] = []

    for repetition in range(PHASE_A_REPETITIONS):
        repo = root / task.task_id / f"rep-{repetition}"
        make_fixture(repo)
        mutate = task.operation in {"edit_valid", "edit_invalid_python"}
        cls = SweAgentACIShadow if provider == "candidate" else BoundedRepositoryACI
        aci = cls(
            repo,
            owner_user_id=f"v34-{provider}",
            session_id=f"{task.task_id}-{repetition}",
            allow_mutation=mutate,
            required_verification_argv=("pytest", "tests/test_fixture.py", "-q")
            if mutate
            else None,
        )
        started = time.perf_counter()
        result, correct, verified, safe = execute(aci, task, repo)
        failed = False
        elapsed = time.perf_counter() - started
        rendered = encoded(result)
        outputs.append(rendered)
        correctness.append(float(correct))
        verifier.append(float(verified))
        security.append(safe)
        walls.append(elapsed)
        failures.append(float(failed))
        coverage.append(expected_path_hits(rendered, task.expected_paths))
        # For edit/guard tasks, task correctness is the semantic relevance.
        relevance.append(
            float(correct)
            if task.primitive in {"syntax_preflight", "governance"}
            else expected_term_hits(rendered, task.expected_terms)
        )

    return PrimitiveTrial(
        task_id=task.task_id,
        primitive=task.primitive,
        correctness=fmean(correctness),
        context_tokens=fmean(char4_tokens(item) for item in outputs),
        wall_seconds=median(walls),
        failure=fmean(failures),
        verifier_success=fmean(verifier),
        repository_coverage=fmean(coverage),
        relevance=fmean(relevance),
        security_preserved=all(security),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("benchmarks/v34_swe_agent_aci/tasks.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runtime/v34/phase-a/result.json"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    revision_result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if (
        revision_result.returncode != 0
        or revision_result.stdout.strip() != LAPLACE_V34_BASELINE
    ):
        raise RuntimeError("v34_phase_a_baseline_mismatch")

    tasks_path = args.tasks if args.tasks.is_absolute() else repo / args.tasks
    tasks = load_phase_a_tasks(tasks_path)
    manifest_hash = manifest_sha256(tasks_path)

    runtime = repo / ".runtime/v34/phase-a"
    baseline_root = runtime / "baseline"
    candidate_root = runtime / "candidate"
    shutil.rmtree(baseline_root, ignore_errors=True)
    shutil.rmtree(candidate_root, ignore_errors=True)
    baseline_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)

    baseline = [
        one_arm(task=task, provider="baseline", root=baseline_root)
        for task in tasks
    ]
    candidate = [
        one_arm(task=task, provider="candidate", root=candidate_root)
        for task in tasks
    ]
    assessment = assess_phase_a(baseline, candidate)

    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result: JsonObject = {
        "schema": "laplace.v34.swe_aci_phase_a_results.v2",
        "laplace_revision": LAPLACE_V34_BASELINE,
        "swe_agent_revision": SWE_AGENT_REFERENCE_COMMIT,
        "task_manifest_sha256": manifest_hash,
        "fixture": "repo_local_generated_git_fixture_v2",
        "repetitions": PHASE_A_REPETITIONS,
        "baseline_trials": [item.as_json() for item in baseline],
        "candidate_trials": [item.as_json() for item in candidate],
        "assessment": assessment,
        "final_adoption_decision": None,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "phase": "A_PRIMITIVE_ONLY",
                "assessment": assessment["assessment"],
                "promising_primitives": assessment["promising_primitives"],
                "governance_safe": assessment["governance_safe"],
                "tasks": 12,
                "repetitions": PHASE_A_REPETITIONS,
                "output": str(output),
                "laplace_revision": LAPLACE_V34_BASELINE,
                "swe_agent_revision": SWE_AGENT_REFERENCE_COMMIT,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
