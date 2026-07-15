"""Pinned local model-artifact metadata and fail-closed file verification."""

from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from .engineering import EngineeringError, JsonObject, _write_json_atomic
from .model_routing import load_dual_model_configuration


_ARTIFACT_IDS = {"phase1_main", "phase2_main", "phase2_rtl_worker"}
_TOP_KEYS = {"schema_version", "quantization_environment", "calibration", "artifacts"}
_ARTIFACT_KEYS = {
    "artifact_id",
    "phase",
    "role",
    "source_repository",
    "source_revision",
    "licence_identifier",
    "source_path",
    "output_path",
    "served_model_name",
    "availability_policy",
    "quantization",
    "serving",
    "verification",
}
_ARTIFACT_IDENTITY = {
    "phase1_main": ("phase1", "main", "installed_official_artifact"),
    "phase2_main": ("phase2", "main", "requires_local_quantization"),
    "phase2_rtl_worker": ("phase2", "rtl_worker", "requires_local_quantization"),
}


def _read_object(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineeringError(f"Cannot read model metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EngineeringError(f"Model metadata {path} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_artifacts(path: Path) -> dict[str, JsonObject]:
    value = _read_object(path.resolve())
    if set(value) != _TOP_KEYS or value.get("schema_version") != 1:
        raise EngineeringError("Model-artifact profile keys or schema version are invalid")
    environment = value.get("quantization_environment")
    calibration = value.get("calibration")
    if not isinstance(environment, dict) or not isinstance(calibration, dict):
        raise EngineeringError("Model quantization environment and calibration must be objects")
    expected_environment = {
        "python",
        "llmcompressor",
        "llmcompressor_revision",
        "transformers",
        "datasets",
        "huggingface_hub",
        "vllm",
    }
    if set(environment) != expected_environment:
        raise EngineeringError("Quantization environment keys are invalid")
    expected_calibration = {"repository", "revision", "split", "samples", "shuffle_seed"}
    if set(calibration) != expected_calibration:
        raise EngineeringError("Calibration profile keys are invalid")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise EngineeringError("Exactly three model artifacts are required")
    records: dict[str, JsonObject] = {}
    for raw in artifacts:
        if not isinstance(raw, dict) or set(raw) != _ARTIFACT_KEYS:
            raise EngineeringError("Model artifact keys are invalid")
        record = dict(raw)
        artifact_id = record.get("artifact_id")
        revision = record.get("source_revision")
        if artifact_id not in _ARTIFACT_IDS or artifact_id in records:
            raise EngineeringError("Model artifact id is invalid or duplicated")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise EngineeringError(f"Model artifact {artifact_id} revision must be exact")
        expected_phase, expected_role, expected_policy = _ARTIFACT_IDENTITY[str(artifact_id)]
        if (
            record.get("phase") != expected_phase
            or record.get("role") != expected_role
            or record.get("availability_policy") != expected_policy
        ):
            raise EngineeringError(f"Model artifact {artifact_id} identity fields are invalid")
        for key in (
            "source_repository",
            "licence_identifier",
            "source_path",
            "output_path",
            "served_model_name",
        ):
            if not isinstance(record.get(key), str) or not str(record[key]).strip():
                raise EngineeringError(f"Model artifact {artifact_id} {key} is invalid")
        for key in ("quantization", "serving", "verification"):
            if not isinstance(record.get(key), dict):
                raise EngineeringError(f"Model artifact {artifact_id} {key} must be an object")
        serving = cast(JsonObject, record["serving"])
        required_serving = {
            "backend",
            "backend_version",
            "environment_path",
            "executable",
            "endpoint",
            "context_tokens",
            "max_output_tokens",
            "kernel",
            "gpu_memory_utilization",
            "max_num_seqs",
            "extra_args",
        }
        allowed_serving = required_serving | {"language_model_only"}
        if not required_serving.issubset(serving) or not set(serving).issubset(allowed_serving):
            raise EngineeringError(f"Model artifact {artifact_id} serving keys are invalid")
        parsed_endpoint = urlsplit(str(serving.get("endpoint", "")))
        environment_path = Path(str(serving.get("environment_path", "")))
        executable_path = Path(str(serving.get("executable", "")))
        if (
            parsed_endpoint.scheme != "http"
            or parsed_endpoint.hostname != "127.0.0.1"
            or parsed_endpoint.port is None
            or parsed_endpoint.path not in {"", "/"}
        ):
            raise EngineeringError(f"Model artifact {artifact_id} endpoint must be loopback-only")
        if (
            not environment_path.is_absolute()
            or not executable_path.is_absolute()
            or executable_path.parent.parent != environment_path
        ):
            raise EngineeringError(
                f"Model artifact {artifact_id} serving environment paths are invalid"
            )
        gpu_fraction = serving.get("gpu_memory_utilization")
        extra_args = serving.get("extra_args")
        if (
            not isinstance(serving.get("context_tokens"), int)
            or not isinstance(serving.get("max_output_tokens"), int)
            or not isinstance(serving.get("max_num_seqs"), int)
            or not isinstance(gpu_fraction, (int, float))
            or not 0 < gpu_fraction < 1
            or not isinstance(extra_args, list)
            or not all(isinstance(item, str) and item for item in extra_args)
        ):
            raise EngineeringError(f"Model artifact {artifact_id} serving values are invalid")
        verification = cast(JsonObject, record["verification"])
        if set(verification) != {"required_files", "quantization_config", "artifact_manifest"}:
            raise EngineeringError(f"Model artifact {artifact_id} verification keys are invalid")
        required_files = verification.get("required_files")
        if (
            not isinstance(required_files, list)
            or not required_files
            or not all(isinstance(item, str) and item for item in required_files)
            or not isinstance(verification.get("quantization_config"), dict)
            or not isinstance(verification.get("artifact_manifest"), str)
        ):
            raise EngineeringError(f"Model artifact {artifact_id} verification values are invalid")
        records[str(artifact_id)] = record
    if set(records) != _ARTIFACT_IDS:
        raise EngineeringError("Model artifact ids are incomplete")
    return records


def validate_serving_environments(
    experiment_root: Path,
    artifact_ids: set[str] | None = None,
    *,
    probe_cli: bool = True,
) -> JsonObject:
    """Validate pinned vLLM environments without importing CUDA or loading a model."""
    records = load_model_artifacts(experiment_root / "model_artifacts.json")
    selected = artifact_ids or set(records)
    unknown = selected - set(records)
    if unknown:
        raise EngineeringError(f"Unknown model artifact ids: {sorted(unknown)}")
    results: list[JsonObject] = []
    for artifact_id in sorted(selected):
        record = records[artifact_id]
        serving = cast(JsonObject, record["serving"])
        environment = Path(str(serving["environment_path"])).expanduser().resolve()
        executable = Path(str(serving["executable"])).expanduser().resolve()
        python = environment / "bin" / "python"
        expected_version = str(serving["backend_version"])
        errors: list[str] = []
        detected_version: str | None = None
        cli_returncode: int | None = None
        cli_error: str | None = None
        arguments_supported: bool | None = None
        missing_arguments: list[str] = []
        if not environment.is_dir():
            errors.append("environment_missing")
        if not python.is_file():
            errors.append("python_missing")
        if not executable.is_file() or not (executable.stat().st_mode & 0o111):
            errors.append("vllm_executable_missing")
        if not errors:
            try:
                metadata = subprocess.run(  # nosec B603
                    [
                        str(python),
                        "-c",
                        "from importlib.metadata import version; print(version('vllm'))",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                detected_version = metadata.stdout.strip() if metadata.returncode == 0 else None
                if detected_version != expected_version:
                    errors.append("vllm_version_mismatch")
                if probe_cli:
                    cli = subprocess.run(  # nosec B603
                        [str(executable), "--version"],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=30,
                    )
                    cli_returncode = cli.returncode
                    if cli.returncode != 0:
                        lines = (cli.stderr or cli.stdout).strip().splitlines()
                        cli_error = lines[-1] if lines else "vllm --version returned no diagnostics"
                        errors.append("vllm_cli_unusable")
                    else:
                        help_result = subprocess.run(  # nosec B603
                            [str(executable), "serve", "--help"],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=30,
                        )
                        required_arguments = {
                            "--host",
                            "--port",
                            "--served-model-name",
                            "--tensor-parallel-size",
                            "--max-model-len",
                            "--max-num-seqs",
                            "--gpu-memory-utilization",
                            "--enable-prefix-caching",
                            "--enable-chunked-prefill",
                            *(
                                argument
                                for argument in cast(list[str], serving["extra_args"])
                                if argument.startswith("--")
                            ),
                        }
                        help_text = help_result.stdout + help_result.stderr
                        missing_arguments = sorted(
                            argument for argument in required_arguments if argument not in help_text
                        )
                        arguments_supported = help_result.returncode == 0 and not missing_arguments
                        if not arguments_supported:
                            errors.append("vllm_serving_arguments_unsupported")
            except (OSError, subprocess.TimeoutExpired) as exc:
                cli_error = str(exc)
                errors.append("vllm_environment_probe_failed")
        results.append(
            {
                "artifact_id": artifact_id,
                "available": not errors,
                "environment_path": str(environment),
                "executable": str(executable),
                "expected_version": expected_version,
                "detected_version": detected_version,
                "cli_returncode": cli_returncode,
                "cli_probe": "COMPLETE" if probe_cli else "NOT_REQUESTED",
                "cli_error": cli_error,
                "arguments_supported": arguments_supported,
                "missing_arguments": missing_arguments,
                "errors": errors,
            }
        )
    return {
        "status": "SERVING_ENVIRONMENTS_READY"
        if all(item["available"] is True for item in results)
        else "SERVING_ENVIRONMENTS_INCOMPLETE",
        "environments": results,
    }


def validate_profile_alignment(experiment_root: Path) -> JsonObject:
    records = load_model_artifacts(experiment_root / "model_artifacts.json")
    arm_a = load_dual_model_configuration(experiment_root / "arm_a_models.json")
    arm_b = load_dual_model_configuration(experiment_root / "arm_b_models.json")
    arm_c = load_dual_model_configuration(experiment_root / "arm_c_models.json")
    candidates = {
        "phase1_main": arm_a.main,
        "phase2_main": arm_b.main,
        "phase2_rtl_worker": arm_c.rtl_worker,
    }
    errors: list[str] = []
    for artifact_id, candidate in candidates.items():
        record = records[artifact_id]
        if candidate is None:
            errors.append(f"{artifact_id}:configured candidate is missing")
            continue
        expected = {
            "model_path": record["output_path"],
            "model": record["served_model_name"],
            "revision": record["source_revision"],
            "endpoint": cast(JsonObject, record["serving"]).get("endpoint"),
        }
        actual = {
            "model_path": candidate.model_path,
            "model": candidate.model,
            "revision": candidate.revision,
            "endpoint": candidate.endpoint,
        }
        for key, value in expected.items():
            if actual[key] != value:
                errors.append(f"{artifact_id}:{key}:configured={actual[key]!r}:expected={value!r}")
    if arm_b.main != arm_c.main:
        errors.append("phase2_main:arms B and C are not identical")
    return {
        "status": "VALID_MODEL_PROFILES" if not errors else "FAILED",
        "artifact_ids": sorted(records),
        "errors": errors,
    }


def _validate_artifact_manifest(
    artifact_id: str,
    output_path: Path,
    manifest_path: Path,
    source_revision: str,
    *,
    verify_hashes: bool,
) -> JsonObject:
    if not manifest_path.is_file():
        return {"status": "MISSING", "path": str(manifest_path)}
    value = _read_object(manifest_path)
    expected_keys = {"schema_version", "artifact_id", "source_revision", "created_at", "files"}
    if set(value) != expected_keys or value.get("schema_version") != 1:
        return {"status": "INVALID", "path": str(manifest_path), "reason": "schema"}
    if value.get("artifact_id") != artifact_id or value.get("source_revision") != source_revision:
        return {"status": "INVALID", "path": str(manifest_path), "reason": "identity"}
    files = value.get("files")
    if not isinstance(files, list) or not files:
        return {"status": "INVALID", "path": str(manifest_path), "reason": "empty_files"}
    errors: list[str] = []
    for raw in files:
        if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
            errors.append("malformed_record")
            continue
        relative = raw.get("path")
        if not isinstance(relative, str):
            errors.append("invalid_path")
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"unsafe_path:{relative}")
            continue
        source = output_path / relative_path
        if not source.is_file():
            errors.append(f"missing:{relative}")
        elif source.stat().st_size != raw.get("size") or (
            verify_hashes and _sha256(source) != raw.get("sha256")
        ):
            errors.append(f"hash:{relative}")
    return {
        "status": ("VERIFIED" if verify_hashes else "METADATA_VERIFIED")
        if not errors
        else "INVALID",
        "path": str(manifest_path),
        "files": len(files),
        "errors": errors,
    }


def _nested_quantized_weights(value: object) -> bool:
    if isinstance(value, dict):
        if (
            value.get("num_bits") == 4
            and value.get("group_size") == 128
            and value.get("symmetric") is True
            and value.get("type") == "int"
        ):
            return True
        return any(_nested_quantized_weights(item) for item in value.values())
    if isinstance(value, list):
        return any(_nested_quantized_weights(item) for item in value)
    return False


def _quantization_config_status(artifact_id: str, config: JsonObject) -> str:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return "QUANTIZATION_MISMATCH"
    if artifact_id == "phase1_main":
        expected = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
        }
        return (
            "VERIFIED"
            if all(quantization.get(key) == value for key, value in expected.items())
            else "QUANTIZATION_MISMATCH"
        )
    method = str(quantization.get("quant_method", "")).replace("_", "-")
    return (
        "VERIFIED"
        if method == "compressed-tensors"
        and quantization.get("format") == "pack-quantized"
        and _nested_quantized_weights(quantization)
        else "QUANTIZATION_MISMATCH"
    )


def validate_local_artifacts(
    experiment_root: Path,
    artifact_ids: set[str] | None = None,
    *,
    verify_hashes: bool = True,
) -> JsonObject:
    records = load_model_artifacts(experiment_root / "model_artifacts.json")
    selected = artifact_ids or set(records)
    unknown = selected - set(records)
    if unknown:
        raise EngineeringError(f"Unknown model artifact ids: {sorted(unknown)}")
    results: list[JsonObject] = []
    for artifact_id in (
        candidate
        for candidate in ("phase1_main", "phase2_main", "phase2_rtl_worker")
        if candidate in selected
    ):
        record = records[artifact_id]
        output = Path(str(record["output_path"])).expanduser().resolve()
        verification = cast(JsonObject, record["verification"])
        raw_required = verification.get("required_files")
        required = [str(item) for item in raw_required] if isinstance(raw_required, list) else []
        missing = [relative for relative in required if not (output / relative).is_file()]
        config_status = "NOT_CHECKED"
        config_path = output / "config.json"
        if config_path.is_file():
            config = _read_object(config_path)
            config_status = _quantization_config_status(artifact_id, config)
        manifest_path = Path(str(verification.get("artifact_manifest"))).expanduser().resolve()
        manifest = _validate_artifact_manifest(
            artifact_id,
            output,
            manifest_path,
            str(record["source_revision"]),
            verify_hashes=verify_hashes,
        )
        available = (
            output.is_dir()
            and not missing
            and config_status == "VERIFIED"
            and manifest.get("status") in {"VERIFIED", "METADATA_VERIFIED"}
        )
        results.append(
            {
                "artifact_id": artifact_id,
                "output_path": str(output),
                "availability_policy": record["availability_policy"],
                "available": available,
                "missing_required_files": missing,
                "config_status": config_status,
                "artifact_manifest": manifest,
            }
        )
    return {
        "status": "ALL_MODEL_ARTIFACTS_AVAILABLE"
        if all(bool(item["available"]) for item in results)
        else "MODEL_ARTIFACTS_INCOMPLETE",
        "artifacts": results,
    }


def write_artifact_manifest(experiment_root: Path, artifact_id: str) -> Path:
    records = load_model_artifacts(experiment_root / "model_artifacts.json")
    if artifact_id not in records:
        raise EngineeringError(f"Unknown artifact id: {artifact_id}")
    record = records[artifact_id]
    output = Path(str(record["output_path"])).expanduser().resolve()
    if not output.is_dir():
        raise EngineeringError(f"Model output directory is missing: {output}")
    verification = cast(JsonObject, record["verification"])
    manifest = Path(str(verification["artifact_manifest"])).expanduser().resolve()
    files = [
        {
            "path": str(path.relative_to(output)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.resolve() != manifest
    ]
    if not files:
        raise EngineeringError("Cannot manifest an empty model artifact")
    _write_json_atomic(
        manifest,
        {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "source_revision": record["source_revision"],
            "created_at": datetime.now(UTC).isoformat(),
            "files": files,
        },
        readonly=True,
    )
    return manifest
