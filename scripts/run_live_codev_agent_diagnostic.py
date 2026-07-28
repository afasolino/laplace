#!/usr/bin/env python3
"""Diagnose the real CodeV validated-patch path in an isolated repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess  # nosec B404 - fixed read-only Git command
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from research_workspace.agent_sandbox import AgentSandboxManager, AgentToolPolicy
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.service_tiers import (
    ChatBackend,
    LanePolicy,
    LocalOpenAIChatBackend,
    ModelLane,
    ModelRoute,
    ServiceTierError,
    TierAuditLog,
    TieredServingService,
    ValidatedPatchAgentBackend,
)
from research_workspace.serving_profile_runtime import observe_gpu
from research_workspace.user_capabilities import CapabilityTier, UserCapabilityStore

from run_live_production_gpu_certification import (
    CODEV_ENDPOINT,
    CODEV_ID,
    STABLE,
    OwnedCodeV,
    _wait_gpu_release,
)
from run_registered_live_gpu_smoke import _git_fixture


class RecordingChatBackend:
    """Retain the synthetic model response only in the diagnostic artifact."""

    def __init__(self, delegate: ChatBackend) -> None:
        self.delegate = delegate
        self.responses: list[dict[str, object]] = []

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        route: ModelRoute,
        tools: Sequence[Mapping[str, object]],
        request_id: str,
    ) -> dict[str, object]:
        response = self.delegate.complete(
            messages=messages,
            route=route,
            tools=tools,
            request_id=request_id,
        )
        self.responses.append(dict(response))
        return response


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)

    result: dict[str, object] = {
        "schema_version": 1,
        "status": "FAIL",
        "initial_gpu": asdict(observe_gpu()),
    }
    if result["initial_gpu"]["compute_pids"]:  # type: ignore[index]
        raise RuntimeError("GPU has unowned compute processes")
    stable_status = subprocess.run(  # nosec B603 B607 - fixed read-only Git query
        ["git", "status", "--short"],
        cwd=STABLE,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if stable_status.returncode != 0 or stable_status.stdout.strip():
        raise RuntimeError("stable checkout is not clean")

    codev = OwnedCodeV(output)
    codev_started = False
    return_code = 2
    try:
        result["owned_codev"] = codev.start()
        codev_started = True
        result["codev_ready"] = codev.wait_ready()
        result["ready_gpu"] = asdict(observe_gpu())
        with tempfile.TemporaryDirectory(
            prefix="laplace-live-codev-agent-"
        ) as temporary:
            temporary_root = Path(temporary)
            repository = temporary_root / "authorized-repository"
            revision = _git_fixture(repository)
            users = UserCapabilityStore(temporary_root / "state/users.sqlite3")
            authorizations = RepositoryAuthorizationStore(
                temporary_root / "state/repositories.sqlite3"
            )
            sandboxes = AgentSandboxManager(
                temporary_root / "state/worktrees",
                authorizations,
            )
            users.set_user("usr_live_plus", CapabilityTier.PLUS)
            authorizations.register("live-systemverilog-fixture", repository)
            authorizations.grant(
                "usr_live_plus",
                "live-systemverilog-fixture",
                base_revision=revision,
            )
            recording = RecordingChatBackend(
                LocalOpenAIChatBackend(timeout_seconds=300)
            )
            service = TieredServingService(
                users=users,
                sandboxes=sandboxes,
                lane_policy=LanePolicy(
                    routes={
                        ModelLane.QUALITY: ModelRoute(
                            ModelLane.QUALITY,
                            "unused-quality-route",
                            "http://127.0.0.1:8201",
                            0,
                        ),
                        ModelLane.STANDARD: ModelRoute(
                            ModelLane.STANDARD,
                            "unused-standard-route",
                            "http://127.0.0.1:8201",
                            10,
                        ),
                        ModelLane.ECONOMY: ModelRoute(
                            ModelLane.ECONOMY,
                            CODEV_ID,
                            CODEV_ENDPOINT,
                            20,
                            context_limit=16_384,
                            output_limit=2_048,
                        ),
                    }
                ),
                chat_backend=recording,
                agent_backend=ValidatedPatchAgentBackend(recording),
                audit_log=TierAuditLog(temporary_root / "state/tier-audit.jsonl"),
            )
            session_id = "live-codev-diagnostic"
            service.create_agent_session(
                user_id="usr_live_plus",
                repo_id="live-systemverilog-fixture",
                session_id=session_id,
                tool_policy=AgentToolPolicy(
                    policy_id="live-codev-diagnostic-v1",
                    allowed_tools=(
                        "read_file",
                        "apply_patch",
                        "run_validation",
                    ),
                    network_enabled=False,
                ),
            )
            try:
                agent_result = service.agent(
                    user_id="usr_live_plus",
                    session_id=session_id,
                    lane=ModelLane.ECONOMY,
                    domain="systemverilog",
                    instruction=(
                        "The file rtl/example.sv contains `assign y = a;`. Modify "
                        "only that line to `assign y = ~a;`. Return the requested "
                        "strict JSON edit object with path rtl/example.sv, exact "
                        "old text, and exact replacement text."
                    ),
                )
            except ServiceTierError as exc:
                result["failure_category"] = exc.category
                result["failure_evidence"] = exc.evidence
            else:
                agent_response = agent_result.get("response")
                if not isinstance(agent_response, dict):
                    agent_response = {}
                diff = str(agent_response.get("diff", ""))
                checks = {
                    "effective_model_exact": agent_result.get("model_id") == CODEV_ID,
                    "expected_path_modified": "rtl/example.sv" in diff,
                    "expected_rtl_change": "assign y = ~a;" in diff,
                    "verification_passed": (
                        agent_response.get("verification_status") == "PASSED"
                    ),
                }
                result["checks"] = checks
                if all(checks.values()):
                    result["status"] = "PASS"
                    return_code = 0
            if recording.responses:
                raw = recording.responses[-1]
                _write_json(output / "synthetic_model_response.json", raw)
                encoded = json.dumps(raw, sort_keys=True).encode("utf-8")
                result["synthetic_model_response"] = {
                    "path": "synthetic_model_response.json",
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "bytes": len(encoded),
                }
    finally:
        result["codev_release"] = (
            codev.stop() if codev_started else {"status": "NOT_STARTED"}
        )
        result["final_gpu"] = _wait_gpu_release(maximum_used_mib=4_000)
        _write_json(output / "diagnostic_results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
