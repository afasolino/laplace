from pathlib import Path

from writer import write_json_output


def test_valid_nested_output_is_written(tmp_path: Path) -> None:
    created = write_json_output(tmp_path, "reports/result.json", {"ok": True})
    assert created == tmp_path / "reports" / "result.json"
    assert created.read_text(encoding="utf-8") == '{"ok": true}'
