import asyncio

from worker_pool import WorkerPool


def test_close_waits_for_successful_work() -> None:
    async def scenario() -> int:
        pool = WorkerPool()
        task = pool.submit(asyncio.sleep(0, result=7))
        await pool.close()
        return task.result()

    assert asyncio.run(scenario()) == 7
