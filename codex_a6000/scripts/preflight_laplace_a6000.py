from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(command: list[str], cwd: Path, timeout_s: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-20000:],
        }
    except FileNotFoundError as exc:
        return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": (exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-20000:] if isinstance(exc.stderr, str) else "timeout",
        }


def find_repo_root(start: Path) -> Path:
    result = run(["git", "rev-parse", "--show-toplevel"], start)
    if result["returncode"] != 0:
        raise SystemExit("Run preflight from inside the cloned Laplace Git repository.")
    return Path(str(result["stdout"]).strip()).resolve()


def executable_probe(name: str, args: list[str], root: Path) -> dict[str, Any]:
    path = shutil.which(name)
    result: dict[str, Any] = {"path": path}
    if path:
        result["probe"] = run([name, *args], root)
    return result


def main() -> int:
    root = find_repo_root(Path.cwd())
    overlay_prompt = root / "CODEX_LAPLACE_A6000_PYTHON_SYSTEMVERILOG_PROMPT.md"
    config = root / "codex_a6000" / "PROJECT_CONFIG.json"
    required_overlay = [
        overlay_prompt,
        config,
        root / "codex_a6000" / "reference_sources" / "python_sources.yaml",
        root / "codex_a6000" / "reference_sources" / "systemverilog_sources.yaml",
        root / "codex_a6000" / "templates" / "python_task_spec.schema.json",
        root / "codex_a6000" / "templates" / "systemverilog_task_spec.schema.json",
        root / "codex_a6000" / "templates" / "paired_benchmark_manifest.schema.json",
        root / "codex_a6000" / "benchmarks" / "paired_task_catalog.yaml",
    ]
    missing_overlay = [
        str(path.relative_to(root)) for path in required_overlay if not path.is_file()
    ]
    if missing_overlay:
        raise SystemExit(f"Overlay is incomplete. Missing: {missing_overlay}")

    required_root = [
        "README.md",
        "AGENTS.md",
        "CODEX_PROMPT.md",
        "PROJECT_CONFIG.yaml",
        "PROMPTS.md",
        "pyproject.toml",
    ]
    core_modules = [
        "src/research_workspace/laplace_cli.py",
        "src/research_workspace/cli.py",
        "src/research_workspace/projects.py",
        "src/research_workspace/retrieval.py",
        "src/research_workspace/llm.py",
        "src/research_workspace/chat.py",
        "src/research_workspace/laplace_server.py",
    ]
    observed_modules = [
        "src/research_workspace/core.py",
        "src/research_workspace/library.py",
        "src/research_workspace/documents.py",
        "src/research_workspace/drafting.py",
        "src/research_workspace/draft_workflow.py",
        "src/research_workspace/analysis.py",
        "src/research_workspace/extraction.py",
        "src/research_workspace/probe.py",
        "src/research_workspace/real_benchmark.py",
        "src/research_workspace/api.py",
        "src/research_workspace/ui.py",
        "src/research_workspace/acquisition.py",
        "src/research_workspace/online.py",
        "src/research_workspace/optional.py",
    ]
    observed_tests = [
        "tests/test_chat.py",
        "tests/test_laplace.py",
        "tests/test_workspace.py",
    ]

    runtime = root / "codex_a6000" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    report_path = runtime / "preflight.json"

    expected_venv_linux = (root / ".venv").resolve()
    active_prefix = Path(sys.prefix).resolve()

    report: dict[str, Any] = {
        "schema_version": 3,
        "repo_root": str(root),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "platform": platform.platform(),
            "prefix": str(active_prefix),
            "expected_repo_venv": str(expected_venv_linux),
            "using_repo_venv": active_prefix == expected_venv_linux,
        },
        "git": {
            "status": run(["git", "status", "--short", "--branch"], root),
            "commit": run(["git", "rev-parse", "HEAD"], root),
            "remote": run(["git", "remote", "-v"], root),
        },
        "repository_shape": {
            "root_files": {name: (root / name).is_file() for name in required_root},
            "core_modules": {name: (root / name).is_file() for name in core_modules},
            "observed_modules": {name: (root / name).is_file() for name in observed_modules},
            "observed_tests": {name: (root / name).is_file() for name in observed_tests},
        },
        "overlay": {str(path.relative_to(root)): path.is_file() for path in required_overlay},
        "imports": {},
        "cuda": {},
        "tools": {},
        "quality_commands": {},
    }

    import_names = [
        "torch",
        "fastapi",
        "pydantic",
        "pytest",
        "yaml",
        "httpx",
        "hypothesis",
        "coverage",
    ]
    for name in import_names:
        report["imports"][name] = importlib.util.find_spec(name) is not None

    if report["imports"]["torch"]:
        import torch

        cuda: dict[str, Any] = {
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cuda.update(
                {
                    "device_0": props.name,
                    "vram_gib": props.total_memory / 1024**3,
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                }
            )
        report["cuda"] = cuda
    else:
        report["cuda"] = {"available": False, "error": "torch is not installed in .venv"}

    probes: dict[str, list[str]] = {
        "nvidia-smi": ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        "git": ["--version"],
        "ruff": ["--version"],
        "mypy": ["--version"],
        "pytest": ["--version"],
        "codex": ["--version"],
        "ollama": ["--version"],
        "verilator": ["--version"],
        "iverilog": ["-V"],
        "verible-verilog-lint": ["--version"],
        "verible-verilog-format": ["--version"],
        "yosys": ["-V"],
        "sby": ["--version"],
        "surelog": ["--version"],
        "docker": ["--version"],
        "podman": ["--version"],
        "vivado": ["-version"],
        "quartus_sh": ["--version"],
        "vsim": ["-version"],
        "xrun": ["-version"],
        "genus": ["-version"],
    }
    for tool, args in probes.items():
        report["tools"][tool] = executable_probe(tool, args, root)

    quality_commands = {
        "pytest_collect": [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        "ruff_format_check": [sys.executable, "-m", "ruff", "format", "--check", "src", "tests"],
        "ruff_check": [sys.executable, "-m", "ruff", "check", "src", "tests"],
        "mypy": [sys.executable, "-m", "mypy", "src"],
    }
    for key, command in quality_commands.items():
        report["quality_commands"][key] = run(command, root, timeout_s=180)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    missing_core = [
        path
        for group_name in ("root_files", "core_modules")
        for path, present in report["repository_shape"][group_name].items()
        if not present
    ]
    if missing_core:
        raise SystemExit(
            f"Clone does not match the expected Laplace baseline. Missing: {missing_core}"
        )
    if sys.version_info < (3, 11):
        raise SystemExit(f"Python 3.11+ is required; active version is {sys.version.split()[0]}.")
    if not report["python"]["using_repo_venv"]:
        raise SystemExit(f"Activate the repository .venv first. Active prefix: {active_prefix}")
    required_imports = ["fastapi", "pydantic", "pytest", "yaml", "httpx"]
    missing_imports = [name for name in required_imports if not report["imports"].get(name)]
    if missing_imports:
        raise SystemExit(f"Required control-plane imports are missing: {missing_imports}")
    if not report["cuda"].get("available", False):
        raise SystemExit("CUDA is unavailable in the active .venv.")
    device = str(report["cuda"].get("device_0", ""))
    vram = float(report["cuda"].get("vram_gib", 0.0))
    if "A6000" not in device or vram < 45.0:
        raise SystemExit(
            f"Expected RTX A6000-class hardware; found {device!r} with {vram:.2f} GiB."
        )
    if not report["tools"]["codex"].get("path"):
        raise SystemExit("Codex CLI is not on PATH. Install and authenticate it before continuing.")

    print(f"Preflight PASS: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
