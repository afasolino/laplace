#!/usr/bin/env python3
"""Run a loopback-only authenticated Zetsu transport fixture for CLI certification."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import uvicorn

from research_workspace.operator_api import (
    AuthCredential,
    OperatorApiSettings,
    OperatorAuth,
    create_operator_app,
)
from research_workspace.operator_service import OperatorService
from research_workspace.service_tiers import LanePolicy, ModelLane, ModelRoute
from research_workspace.user_capabilities import Capability, CapabilityTier


class _FixtureTiered:
    def __init__(self) -> None:
        self.lane_policy = LanePolicy(
            routes={
                ModelLane.QUALITY: ModelRoute(
                    ModelLane.QUALITY,
                    "fixture-qwen38-not-live",
                    "http://127.0.0.1:8206",
                    0,
                    32768,
                    4096,
                ),
                ModelLane.STANDARD: ModelRoute(
                    ModelLane.STANDARD,
                    "fixture-qwen38-not-live",
                    "http://127.0.0.1:8206",
                    10,
                    16384,
                    2048,
                ),
                ModelLane.ECONOMY: ModelRoute(
                    ModelLane.ECONOMY,
                    "fixture-codev-not-live",
                    "http://127.0.0.1:8103",
                    20,
                    8192,
                    2048,
                ),
            },
            quality_reserved_slots=1,
            standard_capacity=4,
            economy_capacity=4,
        )

    def effective_capabilities(self, _user_id: str) -> frozenset[Capability]:
        return frozenset(
            {Capability.CHAT, Capability.AGENT, Capability.PERSONAL_CORPUS}
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--token-env-var", default="LAPLACE_ZETSU_TOKEN")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    token = os.environ.get(arguments.token_env_var, "")
    if len(token) < 24:
        raise RuntimeError(f"missing_fixture_token:{arguments.token_env_var}")
    repository = Path(__file__).resolve().parents[1]
    app = create_operator_app(
        OperatorService(repository, arguments.state_root),
        OperatorAuth(
            {
                token: AuthCredential(
                    "read", "zetsu-fixture-owner", CapabilityTier.PLUS
                )
            }
        ),
        settings=OperatorApiSettings(
            port=arguments.port,
            allowed_origins=(f"http://127.0.0.1:{arguments.port}",),
            allowed_hosts=("127.0.0.1",),
            bearer_api_enabled=True,
            fixture_mode=True,
        ),
        tiered=_FixtureTiered(),  # type: ignore[arg-type]
    )
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=arguments.port,
        log_level="warning",
        access_log=False,
        proxy_headers=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
