"""Deterministic, installed-feature-aware local vLLM serving profiles."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence, TypeAlias
from urllib.parse import urlsplit

JsonObject: TypeAlias = dict[str, object]
KVCacheDType: TypeAlias = Literal[
    "auto",
    "bfloat16",
    "fp8",
    "fp8_per_token_head",
    "int8_per_token_head",
]


class ServingProfileError(RuntimeError):
    """A profile cannot be resolved safely on the installed serving build."""

    def __init__(self, category: str, evidence: JsonObject) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence


@dataclass(frozen=True)
class InstalledServingCapabilities:
    """Immutable summary parsed from the exact installed ``vllm serve`` help."""

    version: str
    flags: frozenset[str]
    help_sha256: str

    @classmethod
    def from_help(cls, *, version: str, help_text: str) -> InstalledServingCapabilities:
        normalized = help_text.replace("\r\n", "\n")
        flags = frozenset(re.findall(r"(?<![A-Za-z0-9_])(--[a-z0-9][a-z0-9-]*)", normalized))
        if "--max-model-len" not in flags or "--port" not in flags:
            raise ServingProfileError(
                "invalid_installed_help",
                {"reason": "required vLLM flags are absent", "version": version},
            )
        return cls(
            version=version.strip(),
            flags=flags,
            help_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True)
class ServingProfile:
    """Strict profile schema; capability and quality policy live elsewhere."""

    profile_id: str
    model_route: Literal["quality", "standard", "economy"]
    model_path: str
    served_model_name: str
    port: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_dtype: KVCacheDType
    kv_cache_memory_bytes: int | None
    enable_prefix_caching: bool
    prefix_hash_algorithm: Literal["sha256", "sha256_cbor_64bit"]
    enable_chunked_prefill: bool
    scheduling_policy: Literal["fcfs", "priority"]
    cpu_offload_gb: float
    cpu_offload_params: tuple[str, ...]
    offload_backend: Literal["auto", "uva", "prefetch"]
    offload_group_size: int
    offload_num_in_group: int
    offload_prefetch_step: int
    kv_offloading_size: float | None
    kv_offloading_backend: Literal["native", "lmcache"]
    gpu_memory_utilization: float
    startup_timeout: int
    request_timeout: int
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"P[0-9]+(?:_[a-z0-9_]+)?", self.profile_id):
            raise ValueError("invalid profile_id")
        model_path = Path(self.model_path)
        if not self.model_path or "\x00" in self.model_path:
            raise ValueError("model_path must be a non-empty path")
        if not model_path.is_absolute() and (
            model_path == Path(".") or ".." in model_path.parts
        ):
            raise ValueError("relative model_path must remain inside the repository")
        if not self.served_model_name or any(char.isspace() for char in self.served_model_name):
            raise ValueError("served_model_name must be a non-empty token")
        if not 1 <= self.port <= 65_535:
            raise ValueError("invalid port")
        if self.max_model_len < 2_048 or self.max_num_seqs < 1:
            raise ValueError("invalid context or sequence capacity")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError("max_num_batched_tokens is too small")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if self.kv_cache_memory_bytes is not None and self.kv_cache_memory_bytes <= 0:
            raise ValueError("kv_cache_memory_bytes must be positive")
        if self.cpu_offload_gb < 0 or self.kv_offloading_size is not None and self.kv_offloading_size <= 0:
            raise ValueError("offload sizes must be positive")
        if self.cpu_offload_params and self.cpu_offload_gb <= 0:
            raise ValueError("selective CPU offload requires cpu_offload_gb")
        if self.offload_backend == "prefetch":
            if self.offload_group_size <= 0:
                raise ValueError("prefetch offload requires a positive group size")
            if not 1 <= self.offload_num_in_group <= self.offload_group_size:
                raise ValueError("invalid prefetch group selection")
        if self.startup_timeout <= 0 or self.request_timeout <= 0:
            raise ValueError("timeouts must be positive")
        for segment in self.cpu_offload_params:
            if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", segment):
                raise ValueError("offload parameter targets must be exact name segments")
        if any(not arg.startswith("--") or "\n" in arg for arg in self.extra_args):
            raise ValueError("extra_args must contain fixed long flags")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ServingProfile:
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - expected
        missing = expected - set(value) - {"extra_args"}
        if unknown or missing:
            raise ServingProfileError(
                "invalid_profile_schema",
                {"unknown_fields": sorted(unknown), "missing_fields": sorted(missing)},
            )
        converted = dict(value)
        for key in ("cpu_offload_params", "extra_args"):
            raw = converted.get(key, ())
            if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
                raise ServingProfileError("invalid_profile_schema", {"field": key})
            converted[key] = tuple(raw)
        try:
            return cls(**converted)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ServingProfileError(
                "invalid_profile_schema",
                {"error": str(exc), "profile_id": value.get("profile_id")},
            ) from exc

    def to_json(self) -> JsonObject:
        value = asdict(self)
        value["cpu_offload_params"] = list(self.cpu_offload_params)
        value["extra_args"] = list(self.extra_args)
        return value


@dataclass(frozen=True)
class ResolvedServingProfile:
    profile: ServingProfile
    command: tuple[str, ...]
    installed_version: str
    installed_help_sha256: str
    resolution_sha256: str

    def to_json(self) -> JsonObject:
        return {
            **self.profile.to_json(),
            "command": list(self.command),
            "installed_version": self.installed_version,
            "installed_help_sha256": self.installed_help_sha256,
            "resolution_sha256": self.resolution_sha256,
        }


def load_profiles(config_root: Path) -> tuple[ServingProfile, ...]:
    """Load all profiles in stable order, rejecting duplicate identities and ports."""

    profiles: list[ServingProfile] = []
    for path in sorted(config_root.resolve().glob("*.json")):
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ServingProfileError("invalid_profile_schema", {"path": str(path)})
        profiles.append(ServingProfile.from_mapping(raw))
    if not profiles:
        raise ServingProfileError("missing_profiles", {"config_root": str(config_root)})
    ids = [profile.profile_id for profile in profiles]
    ports = [profile.port for profile in profiles]
    if len(ids) != len(set(ids)) or len(ports) != len(set(ports)):
        raise ServingProfileError(
            "duplicate_profile_identity",
            {"profile_ids": ids, "ports": ports},
        )
    return tuple(sorted(profiles, key=lambda item: item.profile_id))


def _required_flags(profile: ServingProfile) -> frozenset[str]:
    flags = {
        "--served-model-name",
        "--host",
        "--port",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--kv-cache-dtype",
        "--gpu-memory-utilization",
        "--scheduling-policy",
        "--generation-config",
    }
    if profile.kv_cache_memory_bytes is not None:
        flags.add("--kv-cache-memory-bytes")
    flags.add(
        "--enable-prefix-caching"
        if profile.enable_prefix_caching
        else "--no-enable-prefix-caching"
    )
    if profile.enable_prefix_caching:
        flags.add("--prefix-caching-hash-algo")
    flags.add(
        "--enable-chunked-prefill"
        if profile.enable_chunked_prefill
        else "--no-enable-chunked-prefill"
    )
    if profile.cpu_offload_gb:
        flags.update(("--cpu-offload-gb", "--offload-backend"))
    if profile.cpu_offload_params:
        flags.add("--cpu-offload-params")
    if profile.offload_backend == "prefetch":
        flags.update(
            (
                "--offload-group-size",
                "--offload-num-in-group",
                "--offload-prefetch-step",
            )
        )
    if profile.kv_offloading_size is not None:
        flags.update(("--kv-offloading-size", "--kv-offloading-backend"))
    flags.update(argument.partition("=")[0] for argument in profile.extra_args)
    return frozenset(flags)


def resolve_profile(
    profile: ServingProfile,
    capabilities: InstalledServingCapabilities,
    *,
    executable: Path,
    require_model: bool = True,
    repository_root: Path | None = None,
) -> ResolvedServingProfile:
    """Validate a profile against the installed build and emit one exact argv."""

    missing = sorted(_required_flags(profile) - capabilities.flags)
    if missing:
        raise ServingProfileError(
            "unsupported_profile",
            {
                "profile_id": profile.profile_id,
                "installed_version": capabilities.version,
                "missing_flags": missing,
            },
        )
    configured_model_path = Path(profile.model_path).expanduser()
    if configured_model_path.is_absolute():
        model_path = configured_model_path.resolve()
    else:
        if repository_root is None:
            raise ServingProfileError(
                "relative_model_path_requires_repository_root",
                {"profile_id": profile.profile_id, "model_path": profile.model_path},
            )
        repo = repository_root.resolve()
        model_path = (repo / configured_model_path).resolve()
        try:
            model_path.relative_to(repo)
        except ValueError as exc:
            raise ServingProfileError(
                "relative_model_path_escape",
                {"profile_id": profile.profile_id, "model_path": profile.model_path},
            ) from exc
    if require_model and not model_path.is_dir():
        raise ServingProfileError(
            "missing_model",
            {"profile_id": profile.profile_id, "model_path": str(model_path)},
        )
    if not executable.is_absolute():
        raise ServingProfileError("invalid_executable", {"path": str(executable)})
    command = [
        str(executable),
        "serve",
        str(model_path),
        "--served-model-name",
        profile.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(profile.port),
        "--max-model-len",
        str(profile.max_model_len),
        "--max-num-seqs",
        str(profile.max_num_seqs),
        "--max-num-batched-tokens",
        str(profile.max_num_batched_tokens),
        "--kv-cache-dtype",
        profile.kv_cache_dtype,
        "--gpu-memory-utilization",
        str(profile.gpu_memory_utilization),
        "--scheduling-policy",
        profile.scheduling_policy,
        "--generation-config",
        "vllm",
    ]
    if profile.kv_cache_memory_bytes is not None:
        command.extend(("--kv-cache-memory-bytes", str(profile.kv_cache_memory_bytes)))
    if profile.enable_prefix_caching:
        command.extend(
            ("--enable-prefix-caching", "--prefix-caching-hash-algo", profile.prefix_hash_algorithm)
        )
    else:
        command.append("--no-enable-prefix-caching")
    command.append(
        "--enable-chunked-prefill"
        if profile.enable_chunked_prefill
        else "--no-enable-chunked-prefill"
    )
    if profile.cpu_offload_gb:
        command.extend(("--cpu-offload-gb", str(profile.cpu_offload_gb)))
        command.extend(("--offload-backend", profile.offload_backend))
    if profile.cpu_offload_params:
        command.append("--cpu-offload-params")
        command.extend(profile.cpu_offload_params)
    if profile.offload_backend == "prefetch":
        command.extend(("--offload-group-size", str(profile.offload_group_size)))
        command.extend(("--offload-num-in-group", str(profile.offload_num_in_group)))
        command.extend(("--offload-prefetch-step", str(profile.offload_prefetch_step)))
    if profile.kv_offloading_size is not None:
        command.extend(("--kv-offloading-size", str(profile.kv_offloading_size)))
        command.extend(("--kv-offloading-backend", profile.kv_offloading_backend))
    command.extend(profile.extra_args)
    payload = {
        "profile": profile.to_json(),
        "command": command,
        "installed_version": capabilities.version,
        "installed_help_sha256": capabilities.help_sha256,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ResolvedServingProfile(
        profile=profile,
        command=tuple(command),
        installed_version=capabilities.version,
        installed_help_sha256=capabilities.help_sha256,
        resolution_sha256=digest,
    )


def endpoint_for(profile: ServingProfile) -> str:
    endpoint = f"http://127.0.0.1:{profile.port}"
    parsed = urlsplit(endpoint)
    if parsed.hostname != "127.0.0.1":
        raise ServingProfileError("invalid_local_endpoint", {"endpoint": endpoint})
    return endpoint


def resolve_all(
    profiles: Sequence[ServingProfile],
    capabilities: InstalledServingCapabilities,
    *,
    executable: Path,
    require_model: bool = True,
    repository_root: Path | None = None,
) -> tuple[ResolvedServingProfile, ...]:
    return tuple(
        resolve_profile(
            profile,
            capabilities,
            executable=executable,
            require_model=require_model,
            repository_root=repository_root,
        )
        for profile in profiles
    )
