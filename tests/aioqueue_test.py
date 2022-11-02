import pytest

from queueplus.aioqueue import AioQueue


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
