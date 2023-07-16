##  QueuePlus ➕

> 1️⃣ version: 0.5.2

> ✍️ author: Mitchell Lisle

A Python library that adds functionality to asyncio queues

## Install

```shell
pip install queueplus
```

## Usage

You can use AioQueue with all the same functionality as a regular `asyncio.Queue`.

```python
from queueplus import AioQueue

q = AioQueue()
await q.put("hello world")

message = await q.get()
# hello world
```

With a few extra capabilities

**Iterate over a queue (can be async or not)**
```python
from queueplus import AioQueue
q = AioQueue()

[await q.put(i) for i in range(10)] # in non-async mode you would call q.put_nowait

async for row in q:
    print(row)
```

**Collect all values into a list**
```python
from queueplus import AioQueue
q = AioQueue()

[await q.put(i) for i in range(10)]
messages = q.collect()
# [0, 1, 2, 3, 4 ,5 ,6 ,7, 8, 9]
```

**Create a callback everytime a message is added**
```python
from queueplus import AioQueue
inq = AioQueue()
outq = AioQueue()

async def copy_to_q(message: str):
    await outq.put(message)

inq.add_consumer(copy_to_q)

inq.put("hello world")

await inq.wait_for_consumer()
```

**Enforce a type on a queue, error if violated**
```python
from queueplus import TypedAioQueue, RaiseOnViolation
q = TypedAioQueue(int, violations_strategy=RaiseOnViolation)

[await q.put(i) for i in range(10)]
messages = q.collect()
# [0, 1, 2, 3, 4 ,5 ,6 ,7, 8, 9]

await q.put("hello") # Raises a ViolationError
```

**Enforce a type on a queue, ignore if violated**
```python
from queueplus import TypedAioQueue, DiscardOnViolation
q = TypedAioQueue(int, violations_strategy=DiscardOnViolation)

[await q.put(i) for i in range(10)]
await q.put("hello")

messages = q.collect()
# [0, 1, 2, 3, 4 ,5 ,6 ,7, 8, 9]
```

### Full example
```python
from queueplus import AioQueue
import asyncio

async def main():
    q = AioQueue()
    await q.put("hello")
    await q.put("world")
    
    async for item in q:
        print(item)

if __name__ == "__main__":
    asyncio.run(main())
```
