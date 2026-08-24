from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from research_workspace import zetsu_runtime
from research_workspace.zetsu_config import ZetsuConfigError
from research_workspace.zetsu_runtime import (
    bearer_token_file,
    build_service_specs,
    load_local_plus_token,
    start_local_runtime,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_plus_token(state_root: Path) -> None:
    path = bearer_token_file(state_root)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "tokens": {
                    "secret-plus": {
                        "role": "read",
                        "user_id": "plus-local",
                        "capability_tier": "plus",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def test_service_specs_use_certified_order_and_cu129(tmp_path: Path) -> None:
    vllm = ROOT / ".venv-vllm-cu129/bin/vllm"
    ffmpeg = ROOT / ".runtime/ffmpeg7/lib"
    specs = build_service_specs(
        ROOT,
        tmp_path,
        python=ROOT / ".venv/bin/python",
        vllm=vllm,
        ffmpeg_lib=ffmpeg,
    )

    assert [item.name for item in specs] == ["codev", "qwen", "operator"]
    assert specs[0].environment["LAPLACE_VLLM_EXECUTABLE"] == str(vllm)
    assert str(vllm) in specs[1].argv
    assert specs[2].argv[-1] == str(bearer_token_file(tmp_path))
    assert specs[0].expected_model == "laplace-codev-r1-rl-qwen-7b-w4a16"
    assert specs[1].expected_model == "laplace-quality-qwen38-mtp"


def test_local_plus_token_requires_private_file(tmp_path: Path) -> None:
    path = bearer_token_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "tokens": {
                    "secret-plus": {
                        "role": "read",
                        "user_id": "plus-local",
                        "capability_tier": "plus",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    assert load_local_plus_token(tmp_path) == "secret-plus"

    os.chmod(path, 0o640)
    with pytest.raises(ZetsuConfigError, match="local_bearer_token_file_permissions"):
        load_local_plus_token(tmp_path)


def test_runtime_dry_run_materializes_all_commands_without_starting(tmp_path: Path) -> None:
    result = start_local_runtime(ROOT, tmp_path, dry_run=True)
    assert result["status"] == "DRY_RUN"
    assert result["service_order"] == ["codev", "qwen", "operator"]
    assert result["vllm"] == str(ROOT / ".venv-vllm-cu129/bin/vllm")
    commands = result["commands"]
    assert isinstance(commands, dict)
    assert "run_codev_service.py" in " ".join(commands["codev"])
    assert "run_selected_quality_service.py" in " ".join(commands["qwen"])
    assert "research_workspace.operator_server" in " ".join(commands["operator"])


def test_runtime_interrupt_rolls_back_only_new_supervisors(
    tmp_path: Path, monkeypatch
) -> None:
    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    next_pid = iter((101, 102))
    stopped: list[tuple[int, tuple[str, ...]]] = []

    _write_plus_token(tmp_path)

    monkeypatch.setattr(zetsu_runtime, "_probe", lambda _: False)
    monkeypatch.setattr(zetsu_runtime, "_spawn", lambda _: Process(next(next_pid)))

    def wait_ready(spec, process, *, deadline):  # noqa: ANN001
        del process, deadline
        if spec.name == "qwen":
            raise KeyboardInterrupt

    monkeypatch.setattr(zetsu_runtime, "_wait_ready", wait_ready)
    monkeypatch.setattr(
        zetsu_runtime,
        "_stop_started",
        lambda process, markers: stopped.append((process.pid, markers)) or True,
    )

    with pytest.raises(KeyboardInterrupt):
        start_local_runtime(ROOT, tmp_path)

    assert [item[0] for item in stopped] == [102, 101]


def test_runtime_replaces_only_recorded_incompatible_operator(
    tmp_path: Path, monkeypatch
) -> None:
    _write_plus_token(tmp_path)
    probes: list[str] = []
    zetsu_probes = 0
    replacements: list[tuple[str, Path, Path]] = []

    def probe(spec) -> bool:  # noqa: ANN001
        probes.append(spec.name)
        return True

    def probe_zetsu(token: str) -> None:
        nonlocal zetsu_probes
        assert token == "secret-plus"
        zetsu_probes += 1
        if zetsu_probes == 1:
            raise ZetsuConfigError("operator_zetsu_http_404")

    def replace(spec, *, repository: Path, state_root: Path) -> int:  # noqa: ANN001
        replacements.append((spec.name, repository, state_root))
        return 591372

    monkeypatch.setattr(zetsu_runtime, "_probe", probe)
    monkeypatch.setattr(zetsu_runtime, "_probe_zetsu", probe_zetsu)
    monkeypatch.setattr(zetsu_runtime, "_replace_incompatible_owned_operator", replace)

    result = start_local_runtime(ROOT, tmp_path)

    assert result["status"] == "READY"
    assert result["replaced_incompatible_operator_pid"] == 591372
    assert replacements == [("operator", ROOT.resolve(), tmp_path.resolve())]
    assert probes == ["operator", "codev", "qwen", "operator"]
    assert zetsu_probes == 2
