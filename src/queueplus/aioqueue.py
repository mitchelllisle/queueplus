import asyncio
from asyncio import Queue
from typing import AsyncGenerator, Callable, Coroutine, Generator, Optional, Type, Union

from bloom_filter import BloomFilter

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

    def _put(self, item: DataT):
        new = self._check_for_violation.run(item, self._model)
        if new is not None:
            return super()._put(new)


class BloomFilterQueue(AioQueue):
    def __init__(self, max_elements: int, error_rate: float = 0.1):
        self.bloom = BloomFilter(max_elements=max_elements, error_rate=error_rate)
        super().__init__()

    def item_exists(self, key: str) -> bool:
        return key in self.bloom

    def _put(self, item: DataT) -> None:
        if item not in self.bloom:
            self.bloom.add(item)
            super()._put(item)


class ConditionalQueue(AioQueue):
    def __init__(self, check: Callable[[DataT], bool]):
        self._checker = check
        super().__init__()

    def _put(self, item: DataT):
        if self._checker(item):
            return super()._put(item)
