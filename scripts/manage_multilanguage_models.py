#!/usr/bin/env python3
"""Inspect pinned model profiles and emit safe user-authorized preparation commands."""

from __future__ import annotations

import argparse
import json

# Subprocess use is limited to a fixed local nvidia-smi observation command.
import subprocess  # nosec B404
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from research_workspace.model_artifacts import (
    load_model_artifacts,
    validate_local_artifacts,
    validate_profile_alignment,
    validate_quantization_lock,
    validate_serving_environments,
    write_artifact_manifest,
)


EXPERIMENT = Path("codex_a6000/experiments/multilanguage_dual_model_ablation_v1")


def commands(root: Path) -> dict[str, object]:
    experiment = root / EXPERIMENT
    records = load_model_artifacts(experiment / "model_artifacts.json")
    download: dict[str, str] = {}
    quantize: dict[str, str] = {}
    for artifact_id in ("phase1_main", "phase2_main", "phase2_rtl_worker"):
        record = records[artifact_id]
        repository_id = str(record["source_repository"]).removeprefix("https://huggingface.co/")
        download[artifact_id] = (
            "uvx --from huggingface_hub==1.23.0 hf download "
            f"{repository_id} --revision {record['source_revision']} "
            f"--local-dir {record['source_path']}"
        )
        if artifact_id != "phase1_main":
            quantize[artifact_id] = (
                ".venv-quantization/bin/python scripts/quantize_multilanguage_model.py "
                f"--artifact {artifact_id} --acknowledge-unverified-procedure"
            )
    return {
        "lock_serving_environment": (
            "uv pip compile --python 3.11 --generate-hashes "
            "codex_a6000/experiments/multilanguage_dual_model_ablation_v1/"
            "serving_requirements.in --output-file .models/serving_requirements.lock"
        ),
        "create_serving_environment": (
            "uv venv --python 3.11 .venv-vllm && "
            "uv pip sync --python .venv-vllm/bin/python .models/serving_requirements.lock"
        ),
        "lock_quantization_environment": (
            "uv pip compile --python 3.11 --generate-hashes "
            "codex_a6000/experiments/multilanguage_dual_model_ablation_v1/"
            "quantization_requirements.in --output-file "
            ".models/quantization_requirements.lock"
        ),
        "create_quantization_environment": (
            "uv venv --python 3.11 .venv-quantization && "
            "uv pip sync --python .venv-quantization/bin/python "
            ".models/quantization_requirements.lock"
        ),
        "download": download,
        "quantize": quantize,
        "verification": {
            artifact_id: (
                ".venv/bin/python scripts/manage_multilanguage_models.py manifest "
                f"--artifact {artifact_id}"
            )
            for artifact_id in ("phase1_main", "phase2_main", "phase2_rtl_worker")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "validate-metadata",
            "check",
            "commands",
            "manifest",
            "endpoint",
            "gpu",
            "server-profile",
            "environment",
            "ready",
            "validate-quantization-lock",
        ),
    )
    parser.add_argument("--artifact")
    parser.add_argument("--lock", type=Path, default=Path(".models/quantization_requirements.lock"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    experiment = root / EXPERIMENT
    if arguments.command == "validate-quantization-lock":
        result = validate_quantization_lock(experiment, (root / arguments.lock).resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "QUANTIZATION_LOCK_COMPATIBLE" else 2
    if arguments.command == "validate-metadata":
        result: object = validate_profile_alignment(experiment)
    elif arguments.command == "check":
        result = validate_local_artifacts(experiment)
    elif arguments.command == "commands":
        result = commands(root)
    elif arguments.command == "manifest":
        if arguments.artifact is None:
            parser.error("manifest requires --artifact")
        result = {
            "status": "WRITTEN",
            "path": str(write_artifact_manifest(experiment, arguments.artifact)),
        }
    elif arguments.command == "ready":
        if arguments.artifact is None:
            parser.error("ready requires --artifact")
        local = validate_local_artifacts(experiment)
        artifacts = local.get("artifacts")
        selected = (
            next(
                (
                    item
                    for item in artifacts
                    if isinstance(item, dict) and item.get("artifact_id") == arguments.artifact
                ),
                None,
            )
            if isinstance(artifacts, list)
            else None
        )
        if selected is None:
            parser.error("unknown artifact")
        result = dict(selected)
        result["status"] = "READY" if selected.get("available") is True else "NOT_READY"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if selected.get("available") is True else 2
    elif arguments.command == "server-profile":
        if arguments.artifact is None:
            parser.error("server-profile requires --artifact")
        records = load_model_artifacts(experiment / "model_artifacts.json")
        record = records.get(arguments.artifact)
        if record is None or not isinstance(record.get("serving"), dict):
            parser.error("unknown artifact or malformed serving profile")
        serving = record["serving"]
        if not isinstance(serving, dict):
            parser.error("serving profile is malformed")
        endpoint = urlsplit(str(serving["endpoint"]))
        extra_args = serving.get("extra_args")
        if not isinstance(extra_args, list) or not all(
            isinstance(item, str) for item in extra_args
        ):
            parser.error("serving extra_args must be a string list")
        values = [
            str(serving["executable"]),
            str(record["output_path"]),
            str(record["served_model_name"]),
            str(endpoint.port),
            str(serving["gpu_memory_utilization"]),
            str(serving["context_tokens"]),
            str(serving["max_num_seqs"]),
            *extra_args,
        ]
        if any("\n" in item for item in values):
            parser.error("serving profile values must be single-line")
        print("\n".join(values))
        return 0
    elif arguments.command == "environment":
        if arguments.artifact is None:
            parser.error("environment requires --artifact")
        result = validate_serving_environments(experiment, {arguments.artifact})
        print(json.dumps(result, indent=2, sort_keys=True))
        environments = result.get("environments")
        ready = (
            isinstance(environments, list)
            and len(environments) == 1
            and isinstance(environments[0], dict)
            and environments[0].get("available") is True
        )
        return 0 if ready else 2
    elif arguments.command == "endpoint":
        if arguments.artifact is None:
            parser.error("endpoint requires --artifact")
        records = load_model_artifacts(experiment / "model_artifacts.json")
        record = records.get(arguments.artifact)
        if record is None:
            parser.error("unknown artifact")
        serving = record["serving"]
        if not isinstance(serving, dict):
            parser.error("serving profile is malformed")
        endpoint = str(serving["endpoint"])
        try:
            with urllib.request.urlopen(endpoint + "/v1/models", timeout=5) as response:  # nosec B310
                payload: object = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            result = {"status": "UNAVAILABLE", "endpoint": endpoint, "error": str(exc)}
        else:
            data = payload.get("data") if isinstance(payload, dict) else None
            identities = (
                [
                    item["id"]
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
                if isinstance(data, list)
                else []
            )
            expected = str(record["served_model_name"])
            result = {
                "status": "AVAILABLE" if expected in identities else "MODEL_MISMATCH",
                "endpoint": endpoint,
                "expected": expected,
                "served_models": identities,
            }
    else:
        completed = subprocess.run(  # nosec B603, B607
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        result = {
            "status": "OBSERVED" if completed.returncode == 0 else "UNAVAILABLE",
            "command": "nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits",
            "output": completed.stdout.strip(),
            "error": completed.stderr.strip(),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
