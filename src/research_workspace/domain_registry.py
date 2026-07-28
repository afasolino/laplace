"""Versioned, server-owned domain routing registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

JsonObject: TypeAlias = dict[str, object]
Surface = Literal["chat", "agent", "research"]


class DomainRegistryError(RuntimeError):
    """A requested domain is unknown, disabled, or unavailable for the surface."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


@dataclass(frozen=True)
class DomainDefinition:
    domain_id: str
    display_name: str
    description: str
    surfaces: tuple[Surface, ...]
    eligible_model_routes: tuple[str, ...]
    eligible_verification_tools: tuple[str, ...]
    accepted_reference_code_extensions: tuple[str, ...]
    default: bool = False
    enabled: bool = True
    degraded_reason: str | None = None

    def public(self) -> JsonObject:
        value = asdict(self)
        value["available_in"] = {
            surface: surface in self.surfaces
            for surface in ("chat", "agent", "research")
        }
        value["state"] = (
            "disabled"
            if not self.enabled
            else ("degraded" if self.degraded_reason else "enabled")
        )
        return value


class DomainRegistry:
    """Immutable registry used for both selector rendering and request validation."""

    schema_version = 1
    registry_revision = "domains-v1"

    def __init__(
        self, domains: tuple[DomainDefinition, ...] | None = None
    ) -> None:
        configured = domains or (
            DomainDefinition(
                domain_id="general",
                display_name="Auto / General",
                description=(
                    "General questions with automatic local routing; no engineering "
                    "verification tool is implied."
                ),
                surfaces=("chat", "research"),
                eligible_model_routes=("quality", "standard", "economy-fallback"),
                eligible_verification_tools=(),
                accepted_reference_code_extensions=(),
                default=True,
            ),
            DomainDefinition(
                domain_id="python",
                display_name="Python",
                description=(
                    "Python engineering requests with the installed deterministic "
                    "Python validation gates."
                ),
                surfaces=("chat", "agent", "research"),
                eligible_model_routes=("quality", "standard", "economy-fallback"),
                eligible_verification_tools=("pytest", "ruff", "mypy", "compileall"),
                accepted_reference_code_extensions=(".py", ".pyi"),
            ),
            DomainDefinition(
                domain_id="json",
                display_name="Structured JSON",
                description=(
                    "Structured JSON answers checked by deterministic response "
                    "validation; this does not imply a repository Agent domain."
                ),
                surfaces=("chat", "research"),
                eligible_model_routes=("quality", "standard", "economy-fallback"),
                eligible_verification_tools=("json-schema",),
                accepted_reference_code_extensions=(),
            ),
            DomainDefinition(
                domain_id="systemverilog",
                display_name="SystemVerilog",
                description=(
                    "Verilog/SystemVerilog engineering requests with eligible HDL "
                    "routing and installed verification gates."
                ),
                surfaces=("chat", "agent", "research"),
                eligible_model_routes=("quality", "standard", "economy-codev"),
                eligible_verification_tools=("iverilog", "verilator", "yosys"),
                accepted_reference_code_extensions=(".v", ".vh", ".sv", ".svh"),
            ),
        )
        identifiers = [item.domain_id for item in configured]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate domain ID")
        if sum(item.default and item.enabled for item in configured) != 1:
            raise ValueError("domain registry requires one enabled default")
        self._domains = {item.domain_id: item for item in configured}

    def require(self, domain_id: str, *, surface: Surface) -> DomainDefinition:
        value = self._domains.get(domain_id)
        if value is None:
            raise DomainRegistryError(
                "unknown_domain", {"domain_id": domain_id, "surface": surface}
            )
        if not value.enabled:
            raise DomainRegistryError(
                "domain_disabled", {"domain_id": domain_id, "surface": surface}
            )
        if surface not in value.surfaces:
            raise DomainRegistryError(
                "domain_unavailable_for_surface",
                {"domain_id": domain_id, "surface": surface},
            )
        return value

    def public(self, *, surface: Surface | None = None) -> JsonObject:
        domains = [
            item.public()
            for item in self._domains.values()
            if surface is None or surface in item.surfaces
        ]
        return {
            "schema_version": self.schema_version,
            "registry_revision": self.registry_revision,
            "default_domain_id": next(
                item.domain_id
                for item in self._domains.values()
                if item.default and item.enabled
            ),
            "domains": domains,
        }


DEFAULT_DOMAIN_REGISTRY = DomainRegistry()
