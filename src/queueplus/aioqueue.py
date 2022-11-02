import asyncio
from asyncio import Queue
from typing import AsyncGenerator, Callable, Coroutine, Generator, Optional, Union


class AioQueue(Queue):
    async def wait_for_consumer(self):
        """
        Queue.join is a terrible name for what it does which is wait until all tasks on the queue
        have been processed. This is just a method with a more obvious name that mimics join.
        """
        await self.join()

    def add_consumer(self, callback: Union[Callable, Coroutine]) -> asyncio.Task:
        task = asyncio.create_task(self._consumer(callback))
        return task

    async def _consumer(self, callback: Union[Callable, Coroutine]):
        while True:
            val = await self.get()
            if asyncio.iscoroutinefunction(callback):
                await callback(val)
            else:
                callback(val)  # type: ignore
            self.task_done()

    def collect(self, transform: Optional[Callable] = None):
        return [
            transform(self.get_nowait()) if transform else self.get_nowait()
            for _ in range(self.qsize())
        ]

    async def __aiter__(self) -> AsyncGenerator:
        for _ in range(self.qsize()):
            row = await self.get()
            yield row

    def __iter__(self) -> Generator:
        for _ in range(self.qsize()):
            yield self.get_nowait()
