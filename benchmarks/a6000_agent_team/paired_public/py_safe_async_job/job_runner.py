"""Public fixture for a bounded asynchronous job runner task."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar


T = TypeVar("T")


class JobTimeout(TimeoutError):
    """A job did not finish before its caller-provided deadline."""


@dataclass(frozen=True)
class JobResult:
    value: object
    timed_out: bool


async def run_job(factory: Callable[[], Awaitable[T]], timeout_seconds: float) -> JobResult:
    """Run a coroutine factory with a positive timeout and cancellation cleanup."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    # Intentional seeded defect: this ignores the timeout and leaks a job on cancellation.
    return JobResult(value=await factory(), timed_out=False)
