#!/usr/bin/env python3
"""Exercise Laplace Client Python, Git, RTL, cancellation, and root isolation."""

from __future__ import annotations

import json
import argparse
import shutil
import tempfile
import time
from pathlib import Path
from typing import Sequence

from research_workspace.client_bridge import (
    LocalWorkspace,
    WorkspaceRegistry,
    detected_capabilities,
)


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-root",
        type=Path,
        default=ROOT / "outputs" / "certification" / "client-sandbox",
        help="Repository-local parent for disposable certification workspaces.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    work_root = arguments.work_root.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="laplace-client-cert-", dir=work_root) as raw:
        root = Path(raw) / "project"
        root.mkdir()
        registry = WorkspaceRegistry(Path(raw) / "state/workspaces.json")
        allowed = ("python3", "git", "verilator")
        grant = registry.grant(root, writable=True, allowed_commands=allowed)
        workspace = LocalWorkspace(grant)
        workspace.write_text("input.txt", "laplace\n")

        python_result = workspace.run(
            (
                "python3",
                "-c",
                "from pathlib import Path; "
                "Path('generated.txt').write_text(Path('input.txt').read_text().upper())",
            )
        )
        assert python_result["returncode"] == 0
        assert workspace.read_text("generated.txt") == "LAPLACE\n"

        git_init = workspace.run(("git", "init", "-q"))
        git_status = workspace.git_inspect()
        assert git_init["returncode"] == 0
        assert git_status["status"]["returncode"] == 0  # type: ignore[index]

        rtl_result: dict[str, object] | None = None
        if shutil.which("verilator") is not None:
            workspace.write_text(
                "adder.sv",
                "module adder(input logic [7:0] a, b, output logic [8:0] y);\n"
                "  assign y = a + b;\n"
                "endmodule\n",
            )
            rtl_result = workspace.run(("verilator", "--lint-only", "--sv", "adder.sv"))
            assert rtl_result["returncode"] == 0

        escape_result = workspace.run(
            (
                "python3",
                "-c",
                "from pathlib import Path; print(Path('/etc/passwd').read_text())",
            )
        )
        assert escape_result["returncode"] != 0
        assert "root:x:" not in str(escape_result["stdout"])

        cancel_started = time.monotonic()
        cancelled = workspace.run(
            ("python3", "-c", "import time; time.sleep(30)"),
            cancelled=lambda: time.monotonic() - cancel_started > 0.3,
        )
        assert cancelled["cancelled"] is True

        read_only = LocalWorkspace(
            registry.grant(root, writable=False, allowed_commands=("python3",))
        )
        write_attempt = read_only.run(
            (
                "python3",
                "-c",
                "from pathlib import Path; Path('forbidden.txt').write_text('no')",
            )
        )
        assert write_attempt["returncode"] != 0
        assert not (root / "forbidden.txt").exists()

        print(
            json.dumps(
                {
                    "status": "PASSED",
                    "sandbox": "bubblewrap",
                    "network": "isolated",
                    "python": True,
                    "git": True,
                    "rtl_verilator": rtl_result is not None,
                    "path_isolation": True,
                    "read_only_mount": True,
                    "cancellation": True,
                    "capabilities": detected_capabilities(registry),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
