from dataclasses import dataclass
from datetime import date

import pytest
from hypothesis import given
from hypothesis.strategies import builds, characters, dates, integers, lists, text

from queueplus.aioqueue import AioQueue, TypedAioQueue
from queueplus.violations import (
    DiscardOnViolation,
    RaiseOnViolation,
    ViolationError,
    ViolationStrategy,
)


@pytest.mark.asyncio
async def test_adding_to_queue(text_message):
    q = AioQueue()
    await q.put(text_message)
    assert await q.get() == text_message


@pytest.mark.asyncio
async def test_adding_range_to_queue(ranged_message):
    q = AioQueue()
    [await q.put(message) for message in ranged_message]
    count = 0
    async for message in q:
        assert count == message
        count += 1
    assert count == 10


@pytest.mark.asyncio
async def test_consumer_queue(ranged_message):
    inq = AioQueue()
    task = inq.add_consumer(lambda x: outq.put_nowait(x))
    assert task._state == 'PENDING'

    outq = AioQueue()

    [await inq.put(message) for message in ranged_message]
    count = 0
    await inq.wait_for_consumer()

    async for message in outq:
        assert count == message
        count += 1
    assert count == len(ranged_message)


@pytest.mark.asyncio
async def test_async_consumer_queue(ranged_message):
    inq = AioQueue()

    async def write_to_outq(x):
        await outq.put(x)

    task = inq.add_consumer(write_to_outq)
    assert task._state == 'PENDING'

    outq = AioQueue()

    [await inq.put(message) for message in ranged_message]
    count = 0
    await inq.wait_for_consumer()

    async for message in outq:
        assert count == message
        count += 1
    assert count == len(ranged_message)


@pytest.mark.asyncio
async def test_queue_collect(ranged_message):
    inq = AioQueue()
    [await inq.put(message) for message in ranged_message]
    output = inq.collect()
    assert output == ranged_message


def test_generator(ranged_message):
    q = AioQueue()
    [q.put_nowait(i) for i in ranged_message]

    count = 0
    for message in q:
        assert count == message
        count += 1
    assert count == len(ranged_message)


@given(vals=lists(integers()))
@pytest.mark.asyncio
async def test_typed_queue(vals):
    inq = TypedAioQueue(int)
    [await inq.put(message) for message in vals]
    output = inq.collect()
    assert output == vals


@given(char=characters())
@pytest.mark.asyncio
async def test_typed_queue_raise_violation(char: str, ranged_message: list[int]):
    inq = TypedAioQueue(int, violations_strategy=RaiseOnViolation)
    with pytest.raises(ViolationError):
        await inq.put(char)


@given(chars=lists(characters()), ints=lists(integers()))
@pytest.mark.asyncio
async def test_typed_queue_discard_violation(chars: list[str], ints: list[int]):
    inq = TypedAioQueue(int, violations_strategy=DiscardOnViolation)
    vals = chars + ints
    [await inq.put(val) for val in vals]
    output = inq.collect()
    assert output == ints


@dataclass
class User:
    name: str
    age: int
    dob: date


@given(user=builds(User, name=text(), age=integers(), dob=dates()))
@pytest.mark.asyncio
async def test_typed_queue_pydantic(user: User):
    inq = TypedAioQueue(User, violations_strategy=RaiseOnViolation)
    await inq.put(user)
    assert inq.collect() == [user]


def test_violation_strategy():
    with pytest.raises(TypeError):
        ViolationStrategy()
