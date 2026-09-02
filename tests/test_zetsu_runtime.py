from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from research_workspace import zetsu_runtime
from research_workspace.zetsu_config import ZetsuConfigError
from research_workspace.zetsu_runtime import (
    ProcessIdentity,
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
    assert specs[1].expected_model == "laplace-quality-qwen38-mtp8"


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


def test_runtime_dry_run_nocodev_has_qwen_operator_topology(tmp_path: Path) -> None:
    result = start_local_runtime(ROOT, tmp_path, dry_run=True, codev_enabled=False)
    assert result["status"] == "DRY_RUN"
    assert result["topology"] == "nocodev"
    assert result["codev"] == "intentionally_disabled"
    assert result["service_order"] == ["qwen", "operator"]
    commands = result["commands"]
    assert isinstance(commands, dict)
    assert "codev" not in commands
    assert "--codev-disabled" in commands["operator"]


def test_runtime_interrupt_rolls_back_only_new_supervisors(
    tmp_path: Path, monkeypatch
) -> None:
    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    next_pid = iter((101, 102))
    rollback_records: list[dict[str, object]] = []

    _write_plus_token(tmp_path)

    monkeypatch.setattr(zetsu_runtime, "_probe", lambda _: False)
    monkeypatch.setattr(zetsu_runtime, "_validate_paths", lambda *_, **__: None)
    monkeypatch.setattr(zetsu_runtime, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(zetsu_runtime, "_spawn", lambda _: Process(next(next_pid)))
    monkeypatch.setattr(
        zetsu_runtime,
        "_process_identity",
        lambda pid: zetsu_runtime.ProcessIdentity(pid, pid * 10, pid, pid, f"sha-{pid}"),
    )

    def wait_ready(spec, process, *, deadline):  # noqa: ANN001
        del process, deadline
        if spec.name == "qwen":
            raise KeyboardInterrupt

    monkeypatch.setattr(zetsu_runtime, "_wait_ready", wait_ready)
    def terminate(record, **_kwargs):  # noqa: ANN001
        rollback_records.append(dict(record))
        return {"survivors": [], "initial_owned_pids": [101, 102]}

    monkeypatch.setattr(zetsu_runtime, "_terminate_runtime_record", terminate)

    with pytest.raises(KeyboardInterrupt):
        start_local_runtime(ROOT, tmp_path)

    assert len(rollback_records) == 1
    services = rollback_records[0]["services"]
    assert isinstance(services, dict)
    assert list(services) == ["codev", "qwen"]


def test_runtime_refuses_unowned_compatible_or_incompatible_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    _write_plus_token(tmp_path)
    probes: list[str] = []

    def probe(spec) -> bool:  # noqa: ANN001
        probes.append(spec.name)
        return True

    monkeypatch.setattr(zetsu_runtime, "_probe", probe)
    monkeypatch.setattr(zetsu_runtime, "_validate_paths", lambda *_, **__: None)

    with pytest.raises(ZetsuConfigError, match="codev_endpoint_in_use_by_unowned_process"):
        start_local_runtime(ROOT, tmp_path)

    assert probes == ["codev"]


def test_runtime_transitions_from_full_to_nocodev_with_owned_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    _write_plus_token(tmp_path)
    old_identity = ProcessIdentity(7001, 88, 7001, 7001, "old")
    old_record = {
        "schema_version": 2,
        "runtime_id": "old-runtime",
        "boot_id": "boot-a",
        "repository": str(ROOT.resolve()),
        "state_root": str(tmp_path.resolve()),
        "topology": "full",
        "codev": "required",
        "services": {"operator": {"process": zetsu_runtime.asdict(old_identity)}},
    }
    zetsu_runtime._atomic_json(tmp_path / "run/zetsu_services.json", old_record)
    terminated: list[str] = []
    spawned: list[str] = []

    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    next_pid = iter((8001, 8002))
    monkeypatch.setattr(zetsu_runtime, "_validate_paths", lambda *_, **__: None)
    monkeypatch.setattr(zetsu_runtime, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(zetsu_runtime, "_probe", lambda _spec: False)
    monkeypatch.setattr(zetsu_runtime, "_probe_zetsu", lambda _token: None)
    monkeypatch.setattr(zetsu_runtime, "_wait_ready", lambda *_args, **_kwargs: None)

    def owned(record):  # noqa: ANN001
        return {7001: old_identity} if record.get("runtime_id") == "old-runtime" else {}

    def terminate(record, **_kwargs):  # noqa: ANN001
        terminated.append(str(record["runtime_id"]))
        return {"survivors": [], "initial_owned_pids": [7001]}

    def spawn(spec):  # noqa: ANN001
        spawned.append(spec.name)
        return Process(next(next_pid))

    monkeypatch.setattr(zetsu_runtime, "_runtime_owned_processes", owned)
    monkeypatch.setattr(zetsu_runtime, "_terminate_runtime_record", terminate)
    monkeypatch.setattr(zetsu_runtime, "_spawn", spawn)
    monkeypatch.setattr(
        zetsu_runtime,
        "_process_identity",
        lambda pid: ProcessIdentity(pid, pid, pid, pid, f"sha-{pid}"),
    )

    result = start_local_runtime(ROOT, tmp_path, codev_enabled=False)
    assert result["status"] == "READY"
    assert result["topology"] == "nocodev"
    assert terminated == ["old-runtime"]
    assert spawned == ["qwen", "operator"]
