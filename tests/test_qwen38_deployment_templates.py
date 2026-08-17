from __future__ import annotations

import configparser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _unit(name: str) -> configparser.ConfigParser:
    path = ROOT / "deploy/systemd" / name
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read_string(path.read_text(encoding="utf-8"))
    return parser


def test_model_units_are_persistent_private_and_readiness_ordered() -> None:
    aggregate = _unit("laplace-model-servers.service.example")
    codev = _unit("laplace-codev.service.example")
    qwen = _unit("laplace-qwen.service.example")
    operator = _unit("laplace-operator.service.example")

    assert "laplace-codev.service" in aggregate["Unit"]["Requires"]
    assert "laplace-qwen.service" in aggregate["Unit"]["Requires"]
    assert codev["Service"]["Type"] == "notify"
    assert qwen["Service"]["Type"] == "notify"
    assert codev["Service"]["Restart"] == "on-failure"
    assert qwen["Service"]["Restart"] == "on-failure"
    assert "run_codev_service.py" in codev["Service"]["ExecStart"]
    assert "run_selected_quality_service.py" in qwen["Service"]["ExecStart"]
    assert "laplace-codev.service" in qwen["Unit"]["Requires"]
    assert "127.0.0.1" in operator["Service"]["ExecStart"]
    assert "--enable-bearer-api" in operator["Service"]["ExecStart"]


def test_https_ingress_proxies_only_to_operator_and_preserves_streaming() -> None:
    caddy = (ROOT / "deploy/caddy/Caddyfile.example").read_text(encoding="utf-8")

    assert "reverse_proxy 127.0.0.1:8765" in caddy
    assert "flush_interval -1" in caddy
    assert "Strict-Transport-Security" in caddy
    assert "8102" not in caddy
    assert "8103" not in caddy
