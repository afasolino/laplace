from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar


T = TypeVar("T")


class WorkerPool:
    def __init__(self) -> None:
        self._closed = False
        self._tasks: set[asyncio.Task[object]] = set()

    def submit(self, work: Awaitable[T]) -> asyncio.Task[T]:
        if self._closed:
            raise RuntimeError("worker pool is closed")
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def close(self) -> None:
        # Intentional seeded defect: a second close should be harmless.
        if self._closed:
            raise RuntimeError("worker pool is already closed")
        self._closed = True
        if self._tasks:
            await asyncio.gather(*self._tasks)
