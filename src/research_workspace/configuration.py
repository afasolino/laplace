"""Strict layered configuration with per-setting provenance and redaction."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, cast
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_ENVIRONMENT_OVERRIDES = {
    "LAPLACE_CONFIG_MODE": ("operating_mode",),
    "LAPLACE_CONFIG_BIND_HOST": ("security", "bind_host"),
    "LAPLACE_CONFIG_STATE_ROOT": ("storage", "state_root"),
    "LAPLACE_CONFIG_LOG_LEVEL": ("logging", "level"),
}


class ConfigurationV7Error(ValueError):
    """Configuration failed closed without exposing a secret value."""


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SecuritySettings(StrictConfig):
    local_only: Literal[True] = True
    bind_host: Literal["127.0.0.1", "localhost", "::1"] = "127.0.0.1"
    allow_model_downloads: Literal[False] = False
    allow_telemetry: Literal[False] = False
    redact_paths: Literal[True] = True


class StorageSettings(StrictConfig):
    state_root: str = ".runtime"
    minimum_free_bytes: int = Field(default=536_870_912, ge=0)


class LoggingSettings(StrictConfig):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    structured: Literal[True] = True


class GovernanceSettings(StrictConfig):
    per_user_bytes: int = Field(default=5 * 1024**3, ge=1)
    global_bytes: int = Field(default=100 * 1024**3, ge=1)
    soft_delete_days: int = Field(default=30, ge=0, le=3650)
    audit_retention_days: int = Field(default=365, ge=1, le=36500)

    @model_validator(mode="after")
    def global_not_smaller(self) -> GovernanceSettings:
        if self.global_bytes < self.per_user_bytes:
            raise ValueError("global quota cannot be smaller than per-user quota")
        return self


class ProviderSettings(StrictConfig):
    provider_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")
    display_name: str = Field(min_length=1, max_length=160)
    provider_type: Literal["fixture", "ollama", "vllm"]
    endpoint: str
    model_id: str = Field(min_length=1, max_length=320)
    lifecycle: Literal["fixture", "owned", "unowned"]
    enabled: bool = True
    timeout_seconds: float = Field(default=10, ge=0.1, le=30)
    context_limit: int = Field(default=8192, ge=256, le=10_000_000)
    output_limit: int = Field(default=4096, ge=1, le=1_000_000)
    embedding_support: bool = False

    @model_validator(mode="after")
    def safe_endpoint_and_lifecycle(self) -> ProviderSettings:
        if self.provider_type == "fixture":
            if self.endpoint != "fixture://in-memory" or self.lifecycle != "fixture":
                raise ValueError("fixture provider endpoint/lifecycle is invalid")
            return self
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or self.lifecycle == "fixture"
        ):
            raise ValueError("provider endpoint must be a credential-free loopback HTTP origin")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("provider endpoint port is invalid") from exc
        return self


class RouteSettings(StrictConfig):
    route_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")
    display_name: str = Field(min_length=1, max_length=160)
    provider_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")
    model_id: str = Field(min_length=1, max_length=320)
    lane: Literal["quality", "standard", "economy", "codev", "fixture"]
    enabled: bool = True


class SecretReferences(StrictConfig):
    session_key_env: str = Field(default="LAPLACE_SESSION_KEY", pattern=r"^[A-Z][A-Z0-9_]+$")
    owner_hmac_key_env: str = Field(
        default="LAPLACE_OWNER_HMAC_KEY", pattern=r"^[A-Z][A-Z0-9_]+$"
    )
    backup_key_env: str = Field(default="LAPLACE_BACKUP_KEY", pattern=r"^[A-Z][A-Z0-9_]+$")


class LaplaceConfiguration(StrictConfig):
    schema_version: Literal[1] = 1
    operating_mode: Literal["desktop", "server"] = "desktop"
    security: SecuritySettings = SecuritySettings()
    storage: StorageSettings = StorageSettings()
    logging: LoggingSettings = LoggingSettings()
    governance: GovernanceSettings = GovernanceSettings()
    providers: tuple[ProviderSettings, ...]
    routes: tuple[RouteSettings, ...]
    secrets: SecretReferences = SecretReferences()

    @field_validator("providers", "routes", mode="before")
    @classmethod
    def yaml_sequences_to_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def references_are_complete(self) -> LaplaceConfiguration:
        provider_ids = [item.provider_id for item in self.providers]
        route_ids = [item.route_id for item in self.routes]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("provider IDs must be unique")
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route IDs must be unique")
        known = set(provider_ids)
        if any(route.provider_id not in known for route in self.routes):
            raise ValueError("route references an unknown provider")
        enabled = {item.provider_id for item in self.providers if item.enabled}
        if any(route.enabled and route.provider_id not in enabled for route in self.routes):
            raise ValueError("enabled route references a disabled provider")
        return self


@dataclass(frozen=True)
class EffectiveConfiguration:
    configuration: LaplaceConfiguration
    setting_provenance: dict[str, str]
    secret_classification: dict[str, str]

    def effective(self) -> Mapping[str, object]:
        return self.configuration.model_dump(mode="json")

    def sanitized(self) -> Mapping[str, object]:
        return cast(
            Mapping[str, object],
            _redact(self.configuration.model_dump(mode="json")),
        )

    def provenance(self) -> Mapping[str, str]:
        return dict(self.setting_provenance)

    def diagnostic(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "configuration": self.sanitized(),
            "setting_provenance": dict(sorted(self.setting_provenance.items())),
            "secret_classification": dict(sorted(self.secret_classification.items())),
        }


def _built_in_defaults() -> dict[str, object]:
    return {
        "schema_version": 1,
        "operating_mode": "desktop",
        "security": {
            "local_only": True,
            "bind_host": "127.0.0.1",
            "allow_model_downloads": False,
            "allow_telemetry": False,
            "redact_paths": True,
        },
        "storage": {"state_root": ".runtime", "minimum_free_bytes": 536_870_912},
        "logging": {"level": "INFO", "structured": True},
        "governance": {
            "per_user_bytes": 5 * 1024**3,
            "global_bytes": 100 * 1024**3,
            "soft_delete_days": 30,
            "audit_retention_days": 365,
        },
        "providers": [
            {
                "provider_id": "fixture",
                "display_name": "Deterministic fixture provider",
                "provider_type": "fixture",
                "endpoint": "fixture://in-memory",
                "model_id": "fixture-model-v1",
                "lifecycle": "fixture",
                "enabled": True,
                "timeout_seconds": 10.0,
                "context_limit": 8192,
                "output_limit": 4096,
                "embedding_support": False,
            }
        ],
        "routes": [
            {
                "route_id": "fixture",
                "display_name": "Deterministic fixture route",
                "provider_id": "fixture",
                "model_id": "fixture-model-v1",
                "lane": "fixture",
                "enabled": True,
            }
        ],
        "secrets": {
            "session_key_env": "LAPLACE_SESSION_KEY",
            "owner_hmac_key_env": "LAPLACE_OWNER_HMAC_KEY",
            "backup_key_env": "LAPLACE_BACKUP_KEY",
        },
    }


def _read_yaml(path: Path, *, require_private: bool) -> dict[str, object]:
    resolved = path.resolve()
    try:
        metadata = resolved.stat()
        if metadata.st_size > _MAX_CONFIG_BYTES:
            raise ConfigurationV7Error("configuration_file_too_large")
        if require_private and os.name != "nt":
            if stat.S_IMODE(metadata.st_mode) & 0o022:
                raise ConfigurationV7Error("deployment_configuration_permissions_unsafe")
        raw: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except ConfigurationV7Error:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationV7Error("configuration_file_unreadable") from exc
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ConfigurationV7Error("configuration_root_must_be_mapping")
    return {str(key): value for key, value in raw.items()}


def _deep_merge(
    target: dict[str, object],
    update: Mapping[str, object],
) -> dict[str, object]:
    result = dict(target)
    for key, value in update.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        flattened: dict[str, object] = {}
        for key in sorted(value):
            current = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], current))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = {}
        for index, item in enumerate(value):
            current = f"{prefix}[{index}]"
            flattened.update(_flatten(item, current))
        return flattened or {prefix: []}
    return {prefix: value}


def _set_nested(target: dict[str, object], path: tuple[str, ...], value: object) -> None:
    current = target
    for part in path[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            nested = {}
            current[part] = nested
        current = nested
    current[path[-1]] = value


def _portable_path_kind(value: str) -> str:
    if "\x00" in value:
        raise ConfigurationV7Error("configuration_path_contains_nul")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute():
        return "windows-absolute"
    if posix.is_absolute():
        return "posix-absolute"
    if ".." in windows.parts or ".." in posix.parts:
        raise ConfigurationV7Error("configuration_path_traversal")
    return "relative"


def _redact(value: object, *, key_path: str = "") -> object:
    lowered = key_path.casefold()
    if any(marker in lowered for marker in ("secret", "password", "token", "credential", "key")):
        return "<secret-reference-redacted>"
    if key_path.endswith("state_root") and isinstance(value, str):
        kind = _portable_path_kind(value)
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return {"path_class": kind, "path_digest": digest}
    if isinstance(value, Mapping):
        return {
            str(key): _redact(item, key_path=f"{key_path}.{key}" if key_path else str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, key_path=f"{key_path}[]") for item in value]
    return value


def load_configuration(
    *,
    repository_configuration: Path | None = None,
    deployment_configuration: Path | None = None,
    environment: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, object] | None = None,
) -> EffectiveConfiguration:
    """Merge exactly defaults < repository < deployment < env < CLI."""

    merged = _built_in_defaults()
    provenance = {key: "secure_built_in_defaults" for key in _flatten(merged)}
    for source, path, private in (
        ("versioned_repository_configuration", repository_configuration, False),
        ("external_deployment_configuration", deployment_configuration, True),
    ):
        if path is None:
            continue
        update = _read_yaml(path, require_private=private)
        merged = _deep_merge(merged, update)
        provenance.update({key: source for key in _flatten(update)})

    selected_environment = dict(os.environ if environment is None else environment)
    unknown_environment = sorted(
        key
        for key in selected_environment
        if key.startswith("LAPLACE_CONFIG_") and key not in _ENVIRONMENT_OVERRIDES
    )
    if unknown_environment:
        raise ConfigurationV7Error("unknown_environment_override")
    environment_update: dict[str, object] = {}
    for name, setting_path in _ENVIRONMENT_OVERRIDES.items():
        if name not in selected_environment:
            continue
        raw: object = selected_environment[name]
        _set_nested(environment_update, setting_path, raw)
    if environment_update:
        merged = _deep_merge(merged, environment_update)
        provenance.update(
            {key: "environment_override" for key in _flatten(environment_update)}
        )

    explicit = dict(cli_overrides or {})
    if explicit:
        merged = _deep_merge(merged, explicit)
        provenance.update({key: "explicit_cli_argument" for key in _flatten(explicit)})
    try:
        configuration = LaplaceConfiguration.model_validate(merged)
    except ValidationError as exc:
        raise ConfigurationV7Error("configuration_schema_invalid") from exc
    _portable_path_kind(configuration.storage.state_root)
    secret_classes = {
        "secrets.session_key_env": "secret_reference",
        "secrets.owner_hmac_key_env": "secret_reference",
        "secrets.backup_key_env": "secret_reference",
        "providers[].endpoint": "sensitive_transport_configuration",
        "storage.state_root": "sensitive_path",
    }
    return EffectiveConfiguration(configuration, provenance, secret_classes)


def write_diagnostic_export(path: Path, configuration: EffectiveConfiguration) -> Path:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(configuration.diagnostic(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
        if os.name != "nt":
            os.chmod(resolved, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved
