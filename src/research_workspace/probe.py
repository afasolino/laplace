from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class ProbeItem:
    status: str
    command: list[str] | None
    raw_output: str
    parsed: dict[str, object]


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _run(command: list[str], runner: Runner) -> ProbeItem:
    if shutil.which(command[0]) is None:
        return ProbeItem("unavailable", command, "executable not found", {})
    try:
        result = runner(command)
        return ProbeItem(
            "detected" if result.returncode == 0 else "unsupported",
            command,
            (result.stdout + result.stderr).strip(),
            {"returncode": result.returncode},
        )
    except PermissionError as exc:
        return ProbeItem("permission_blocked", command, str(exc), {})
    except (OSError, subprocess.SubprocessError) as exc:
        return ProbeItem("unsupported", command, str(exc), {})


def default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)


def _ollama_service_probe() -> dict[str, object]:
    endpoint = "http://127.0.0.1:11434"
    result: dict[str, object] = {"endpoint": endpoint, "loopback_only": False}
    try:
        with urllib.request.urlopen(endpoint + "/api/tags", timeout=5) as response:
            value = json.loads(response.read())
        result["api_status"] = "detected"
        result["models"] = [
            item.get("name") for item in value.get("models", []) if isinstance(item, dict)
        ]
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        result["api_status"] = "unavailable"
        result["api_error"] = str(exc)
    try:
        netstat = subprocess.run(
            ["cmd", "/c", "netstat -ano | findstr 11434"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        lines = [line.strip() for line in netstat.stdout.splitlines() if line.strip()]
        result["listener_raw"] = lines
        result["loopback_only"] = bool(lines) and all(
            "127.0.0.1:11434" in line or "[::1]:11434" in line for line in lines
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["listener_error"] = str(exc)
    try:
        process = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-Process -Name ollama -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Path)",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        result["process_path"] = process.stdout.strip() or None
    except (OSError, subprocess.SubprocessError) as exc:
        result["process_path_error"] = str(exc)
    result["status"] = (
        "detected"
        if result.get("api_status") == "detected" and result.get("loopback_only")
        else "unsupported"
    )
    return result


def collect_probe(runner: Runner = default_runner) -> dict[str, object]:
    disk = shutil.disk_usage(Path.cwd())
    base: dict[str, object] = {
        "os": {"status": "detected", "value": platform.platform()},
        "python": {"status": "detected", "value": sys.version},
        "cpu": {"status": "detected", "value": platform.processor() or platform.machine()},
        "ram": {"status": "unsupported", "note": "portable stdlib RAM probe unavailable"},
        "disk": {"status": "detected", "free_bytes": disk.free, "total_bytes": disk.total},
    }
    commands = {
        "nvidia": [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,pstate,power.draw",
            "--format=csv,noheader,nounits",
        ],
        "ollama": ["ollama", "--version"],
        "llama_cpp": ["llama-cli", "--version"],
        "amd_npu": [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-PnpDevice | Where-Object FriendlyName -Match 'NPU|Ryzen AI' | Format-List",
        ],
        "onnxruntime": [
            sys.executable,
            "-c",
            "import onnxruntime as o; print(o.get_available_providers())",
        ],
    }
    for name, command in commands.items():
        base[name] = asdict(_run(command, runner))
    if runner is default_runner:
        service = _ollama_service_probe()
        base["ollama_service"] = service
        ollama_item = base.get("ollama")
        if service.get("status") == "detected" and isinstance(ollama_item, dict):
            ollama_item["status"] = "detected_via_loopback_service"
            ollama_item["parsed"] = {
                "process_path": service.get("process_path"),
                "models": service.get("models"),
            }
    return base


def write_probe(root: Path, probe: dict[str, object]) -> tuple[Path, Path]:
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    json_path = out / "system_probe.json"
    report_path = out / "system_probe.md"
    json_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    lines = ["# System probe", "", "No unavailable property is inferred.", ""]
    for key, value in probe.items():
        status = value.get("status", "unknown") if isinstance(value, dict) else "unknown"
        lines.append(f"- **{key}:** `{status}` — `{json.dumps(value, ensure_ascii=False)}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, report_path
