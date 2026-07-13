"""Public fixture for an asynchronous deadline wrapper."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


class DeadlineExceeded(TimeoutError):
    """The work item did not complete within its requested deadline."""


async def run_with_deadline(factory: Callable[[], Awaitable[T]], seconds: float) -> T:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return await factory()
