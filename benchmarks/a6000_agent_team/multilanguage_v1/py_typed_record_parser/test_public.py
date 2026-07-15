from record_parser import Record, parse_record


def test_parses_basic_record() -> None:
    assert parse_record("name=alpha,count=2") == Record("alpha", 2)
