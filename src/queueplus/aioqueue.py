import asyncio
from asyncio import Queue
from typing import AsyncGenerator, Callable, Coroutine, Generator, Optional, Type, Union

from queueplus.datatypes import DataT
from queueplus.violations import RaiseOnViolation, ViolationStrategy


class AioQueue(Queue):
    async def wait_for_consumer(self):
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

    def __len__(self) -> int:
        return self.qsize()

    def __iter__(self) -> Generator:
        for _ in range(self.qsize()):
            yield self.get_nowait()


class TypedAioQueue(AioQueue):
    def __init__(
        self, model: DataT, violations_strategy: Type[ViolationStrategy] = RaiseOnViolation
    ):
        self._model = model
        self._check_for_violation = violations_strategy()
        super().__init__()

    def _put(self, item: dict):
        new = self._check_for_violation.run(item, self._model)
        if new is not None:
            return super()._put(new)
