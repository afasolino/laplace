from __future__ import annotations

import json
import os
import sys

import pytest
from pathlib import Path

from research_workspace.prime_agent_gate import PrimeAgentGateError, main, run_gate


def test_advisory_gate_runs_argv_without_shell(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    spec = tmp_path / "gate.json"
    spec.write_text(
        json.dumps(
            {
                "workspace": str(workspace),
                "scratch_root": str(scratch),
                "environment": {"PATH": os.environ.get("PATH", "")},
                "plan": [
                    {
                        "cwd": ".",
                        "argv": [sys.executable, "-c", "print('GATE_OK')"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["--spec", str(spec)]) == 0


def test_advisory_gate_rejects_cwd_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    spec = tmp_path / "gate.json"
    spec.write_text(
        json.dumps(
            {
                "workspace": str(workspace),
                "scratch_root": str(scratch),
                "environment": {"PATH": os.environ.get("PATH", "")},
                "plan": [{"cwd": "../", "argv": [sys.executable, "-c", "pass"]}],
            }
        ),
        encoding="utf-8",
    )
    assert main(["--spec", str(spec)]) == 2


def test_advisory_gate_runs_in_disposable_copy(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "source.txt").write_text("original\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    spec = tmp_path / "gate.json"
    spec.write_text(
        json.dumps(
            {
                "workspace": str(workspace),
                "scratch_root": str(scratch),
                "environment": {"PATH": os.environ.get("PATH", "")},
                "plan": [
                    {
                        "cwd": ".",
                        "argv": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('source.txt').write_text('changed')",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["--spec", str(spec)]) == 0
    assert (workspace / "source.txt").read_text(encoding="utf-8") == "original\n"


def test_advisory_gate_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (workspace / "escape").symlink_to(outside)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    spec = tmp_path / "gate.json"
    spec.write_text(
        json.dumps(
            {
                "workspace": str(workspace),
                "scratch_root": str(scratch),
                "plan": [{"cwd": ".", "argv": [sys.executable, "-c", "print('ok')"]}],
                "environment": {"PATH": os.environ.get("PATH", "")},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PrimeAgentGateError, match="prime_gate_absolute_symlink_forbidden"):
        run_gate(spec)
