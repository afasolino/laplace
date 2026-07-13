from __future__ import annotations

import json
from pathlib import Path

from path_cli import write_json


def test_write_json_returns_project_relative_output(tmp_path: Path) -> None:
    result = write_json(tmp_path, "reports/result.json", {"answer": 3})
    assert result == tmp_path / "reports" / "result.json"
    assert json.loads(result.read_text(encoding="utf-8")) == {"answer": 3}
