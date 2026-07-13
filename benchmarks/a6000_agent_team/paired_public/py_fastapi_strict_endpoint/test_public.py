from __future__ import annotations

from endpoint import SquareRequest, square


def test_square_preserves_existing_valid_contract() -> None:
    assert square(SquareRequest(value=9)) == {"result": 81}
