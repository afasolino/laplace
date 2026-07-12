from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when unsafe or invalid configuration is supplied."""


@dataclass(frozen=True)
class Settings:
    root: Path
    bind_host: str
    context_tokens: int
    concurrency: int
    model: str | None
    embedding_model: str
    model_endpoint: str
    raw: dict[str, Any]

    @property
    def documents_dir(self) -> Path:
        return self.root / str(self.raw["storage"]["source_documents"])

    @property
    def parsed_dir(self) -> Path:
        return self.root / str(self.raw["storage"]["parsed_documents"])

    @property
    def database(self) -> Path:
        return self.root / str(self.raw["storage"]["metadata_database"])


def load_settings(path: Path | str = "PROJECT_CONFIG.yaml") -> Settings:
    config_path = Path(os.getenv("RW_CONFIG", str(path))).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load configuration: {exc}") from exc
    if not isinstance(raw, dict) or not raw.get("project", {}).get("local_only"):
        raise ConfigurationError("project.local_only must be true")
    runtime = raw.get("runtime", {})
    host = os.getenv("RW_BIND_HOST", str(runtime.get("bind_host", "127.0.0.1")))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigurationError("Only loopback bind hosts are allowed")
    context = int(os.getenv("RW_CONTEXT_TOKENS", runtime.get("default_context_tokens", 8192)))
    concurrency = int(os.getenv("RW_CONCURRENCY", runtime.get("default_concurrency", 1)))
    if context < 512 or context > 8192 or concurrency != 1:
        raise ConfigurationError("context must be 512..8192 and concurrency must be one")
    return Settings(
        root=config_path.parent,
        bind_host=host,
        context_tokens=context,
        concurrency=concurrency,
        model=os.getenv("RW_MODEL") or str(raw["models"]["main_text"]["preferred_family"]),
        embedding_model=os.getenv(
            "RW_EMBEDDING_MODEL", str(raw["models"]["embeddings"]["preferred_family"])
        ),
        model_endpoint=os.getenv("RW_MODEL_ENDPOINT", "http://127.0.0.1:11434"),
        raw=raw,
    )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            sort_keys=True,
        )


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def deterministic_run_id(operation: str, inputs: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"operation": operation, "inputs": inputs}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def write_manifest(
    settings: Settings, operation: str, inputs: dict[str, Any], artifacts: list[Path]
) -> Path:
    run_id = deterministic_run_id(operation, inputs)
    target = settings.root / "outputs" / "manifests" / f"{run_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "operation": operation,
        "inputs": inputs,
        "artifacts": [str(p.resolve().relative_to(settings.root)) for p in artifacts],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def ensure_layout(settings: Settings) -> None:
    for key, value in settings.raw["storage"].items():
        target = settings.root / str(value)
        if key == "metadata_database" or target.suffix in {".db", ".sqlite"}:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
