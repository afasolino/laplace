import asyncio

import pytest

from deadline import run_with_deadline


def test_positive_deadline_returns_completed_result() -> None:
    assert asyncio.run(run_with_deadline(lambda: asyncio.sleep(0, result=7), 1.0)) == 7


def test_non_positive_deadline_is_rejected() -> None:
    with pytest.raises(ValueError):
        asyncio.run(run_with_deadline(lambda: asyncio.sleep(0), 0))
