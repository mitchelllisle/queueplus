import asyncio
from asyncio import Queue
from typing import AsyncGenerator, Callable, Coroutine, Generator, Optional, Set, Type, Union

from bloom_filter import BloomFilter

from queueplus.datatypes import DataT
from queueplus.violations import RaiseOnViolation, ViolationStrategy


class AioQueue(Queue):
    async def wait_for_consumer(self):
        """
        Asynchronously wait for all items in the queue to be processed.

        This method is a coroutine. It will block until all items in the queue
        have been processed and acknowledged by the `task_done()` method.
        """
        await self.join()

    def add_consumer(self, callback: Union[Callable, Coroutine]) -> asyncio.Task:
        """
        Add a consumer task to process items from the queue.

        Args:
            callback (Union[Callable, Coroutine]): A function or coroutine that
            will be called with each item from the queue.

        Returns:
            asyncio.Task: The task object representing the consumer task.
        """
        task = asyncio.create_task(self._consumer(callback))
        return task

    async def _consumer(self, callback: Union[Callable, Coroutine]):
        """
        An internal asynchronous method that represents a consumer task.

        This method continuously gets items from the queue and applies the
        provided callback to each item.

        Args:
            callback (Union[Callable, Coroutine]): A function or coroutine that
            will be called with each item from the queue.
        """
        while True:
            val = await self.get()
            if asyncio.iscoroutinefunction(callback):
                await callback(val)
            else:
                callback(val)  # type: ignore
            self.task_done()

    def collect(self, transform: Optional[Callable] = None):
        """
        Collect and optionally transform all items currently in the queue.

        Args:
            transform (Optional[Callable]): An optional function that will be
            applied to each item in the queue.

        Returns:
            list: A list of items from the queue, transformed if a transform
            function is provided.
        """
        return [
            transform(self.get_nowait()) if transform else self.get_nowait()
            for _ in range(self.qsize())
        ]

    async def __aiter__(self) -> AsyncGenerator:
        """
        Asynchronously iterate over items in the queue.

        This method allows the queue to be used as an asynchronous iterator.

        Yields:
            Each item from the queue.
        """
        for _ in range(self.qsize()):
            row = await self.get()
            yield row

    def __len__(self) -> int:
        """
        Return the number of items in the queue.

        Returns:
            int: The number of items in the queue.
        """
        return self.qsize()

    def __iter__(self) -> Generator:
        """
        Synchronously iterate over items in the queue.

        This method allows the queue to be used as a synchronous iterator,
        yielding items as they are available without blocking.

        Yields:
            Each item from the queue.
        """
        for _ in range(self.qsize()):
            yield self.get_nowait()


class TypedAioQueue(AioQueue):
    def __init__(
        self, model: DataT, violations_strategy: Type[ViolationStrategy] = RaiseOnViolation
    ):
        """
        Initialize a TypedAioQueue with a data model and a violation strategy.

        Args:
            model (DataT): The data type model that all items in the queue should conform to.
            violations_strategy (Type[ViolationStrategy], optional): A strategy to handle
            violations of the data type model. Defaults to RaiseOnViolation, which raises
            an exception on violation.

        The TypedAioQueue extends the AioQueue by adding type checking for each item
        put into the queue. It uses the specified model to validate the data and applies
        the violation strategy if the data does not conform to the model.
        """
        self._model = model
        self._check_for_violation = violations_strategy()
        super().__init__()

    def _put(self, item: DataT):
        """
        Put an item into the queue after validating its type against the model.

        This method extends the '_put' method of AioQueue. It first checks the item
        against the provided model, applying the violation strategy if necessary. If the
        item is valid or transformed by the violation strategy, it's then placed into
        the queue.

        Args:
            item (DataT): The item to be placed in the queue.

        The method leverages the provided model for type checking, ensuring that all items
        in the queue adhere to the specified data structure. The method uses the
        violation strategy to handle any discrepancies between the item and the model.
        """
        new = self._check_for_violation.run(item, self._model)
        if new is not None:
            return super()._put(new)


class BloomFilterQueue(AioQueue):
    def __init__(self, max_elements: int, error_rate: float = 0.1):
        """
        Initialize a BloomFilterQueue with a specified size and error rate.

        Args:
            max_elements (int): The maximum number of elements the Bloom filter is expected to store.
            error_rate (float, optional): The desired probability of false positives. Defaults to 0.1.

        This class extends AioQueue by incorporating a Bloom filter. The Bloom filter is
        used to efficiently test whether an element is present in the queue with a certain
        allowable rate of false positives. This is particularly useful for large datasets
        where memory efficiency is crucial, and a small percentage of false positives is acceptable.
        """
        self.bloom = BloomFilter(max_elements=max_elements, error_rate=error_rate)
        super().__init__()

    def item_exists(self, key: str) -> bool:
        """
        Check if an item exists in the Bloom filter.

        Args:
            key (str): The item to check for existence in the Bloom filter.

        Returns:
            bool: True if the item might exist in the queue, False if it definitely does not.

        This method provides a way to check for the possible presence of an item in the queue.
        Due to the nature of Bloom filters, this check can yield false positives but not false negatives.
        """
        return key in self.bloom

    def _put(self, item: DataT) -> None:
        """
        Put an item into the queue and Bloom filter, if it's not already present.

        Args:
            item (DataT): The item to be placed in the queue.

        This method overrides the '_put' method of AioQueue. Before placing the item in the queue,
        it checks if the item is already in the Bloom filter. If the item is not in the filter,
        it's added to both the Bloom filter and the queue. This ensures that only unique items
        are stored in the queue, reducing redundancy and enhancing memory efficiency.
        """
        if item not in self.bloom:
            self.bloom.add(item)
            super()._put(item)


class ConditionalQueue(AioQueue):
    def __init__(self, check: Callable[[DataT], bool]):
        """
        Initialize a ConditionalQueue with a given condition check function.

        Args:
            check (Callable[[DataT], bool]): A function that takes an item of type DataT
            and returns a boolean indicating whether the item meets a certain condition.

        The ConditionalQueue extends AioQueue by adding a condition for items to be put in the queue.
        Only items that satisfy the given condition (as determined by the `check` function)
        are added to the queue.
        """
        self._checker = check
        super().__init__()

    def _put(self, item: DataT):
        """
        Put an item into the queue if it meets the specified condition.

        Args:
            item (DataT): The item to be potentially placed in the queue.

        This method overrides the '_put' method of AioQueue. It applies the condition
        check to each item. If the item satisfies the condition (i.e., `check` function returns True),
        the item is placed in the queue using the base class's `_put` method.
        """
        if self._checker(item):
            return super()._put(item)


class UniqueQueue(AioQueue):
    def __init__(self, remove_on_get: bool = True):
        """
        Initialize a UniqueQueue which ensures that all items in the queue are unique.

        Args:
            remove_on_get (bool, optional): If True, items will be removed from the
            set of seen items when they are retrieved from the queue. Defaults to True.

        This queue maintains a set of all items that have been put into it, ensuring
        that no duplicate items are stored. The behavior upon getting items from the queue
        can be controlled with the `remove_on_get` parameter.
        """
        super().__init__()
        self._remove_on_get = remove_on_get
        self._seen: Set[DataT] = set()

    def _put(self, item: DataT):
        """
        Put an item into the queue if it has not been seen before.

        Args:
            item (DataT): The item to be potentially placed in the queue.

        Overrides the '_put' method of AioQueue. This implementation first checks if
        the item is in the set of seen items. If it's not, it's added to both the seen
        set and the queue, ensuring uniqueness of items in the queue.
        """
        if item not in self._seen:
            self._seen.add(item)
            return super()._put(item)

    def _get(self):
        """
        Retrieve and return an item from the queue.

        Returns:
            DataT: The next item from the queue.

        Overrides the '_get' method of AioQueue. If `remove_on_get` is set to True,
        the retrieved item is also removed from the set of seen items, making it
        eligible to be added to the queue again in the future.
        """
        item = super()._get()
        self._seen.remove(item) if self._remove_on_get else None
        return item
