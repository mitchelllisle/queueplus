from abc import ABC, abstractmethod
from typing import Any

from queueplus.datatypes import DataT


class ViolationError(TypeError):
    """Class to raise when violations occur"""


class ViolationStrategy(ABC):
    """ViolationStrategy for dealing with TypedAioQueue errors

    If you attempt to put an item on a TypedAioQueue that isn't of the type specified we will
    decide what to do based on the ViolationStrategy provided. Note: This class is not meant
    to be used directly
    """

    @abstractmethod
    def checks(self, item: Any, model: DataT) -> Any:
        pass

    @staticmethod
    def _is_item_of_type(item: Any, model: DataT):
        return isinstance(item, model)

    def run(self, item: Any, model: DataT) -> dict | DataT:
        checked = self.checks(item, model)
        return checked


class RaiseOnViolation(ViolationStrategy):
    name: str = 'raise-error-on-violation'

    def checks(self, item: Any, model: DataT):
        if not self._is_item_of_type(item, model):
            raise ViolationError(
                f'this is a TypedQueue with a strict {self.name} strategy. '
                f'Item must be of type {model.__name__} not {type(item)}'
            )
        return item


class DiscardOnViolation(ViolationStrategy):
    name: str = 'discard-error-on-violation'

    def checks(self, item: Any, model: DataT) -> None:
        if self._is_item_of_type(item, model):
            return item
        return None
