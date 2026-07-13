from __future__ import annotations

import asyncio

import pytest

from job_runner import JobResult, run_job


def test_successful_job_returns_typed_result() -> None:
    async def value() -> int:
        return 7

    assert asyncio.run(run_job(value, 1)) == JobResult(value=7, timed_out=False)


def test_nonpositive_timeout_is_rejected() -> None:
    async def value() -> int:
        return 7

    with pytest.raises(ValueError):
        asyncio.run(run_job(value, 0))
