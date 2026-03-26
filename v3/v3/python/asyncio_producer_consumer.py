import asyncio
import random

async def producer(queue, id):
    for i in range(3):
        item = f"Item {id}-{i}"
        await queue.put(item)
        print(f"Producer {id} produced {item}")
        await asyncio.sleep(random.uniform(0.1, 0.3))
    await queue.put(None)  # sentinel

async def consumer(queue, id):
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"Consumer {id} consumed {item}")
        await asyncio.sleep(random.uniform(0.2, 0.4))
        queue.task_done()

async def main():
    q = asyncio.Queue(maxsize=10)
    producers = [asyncio.create_task(producer(q, i)) for i in range(2)]
    consumers = [asyncio.create_task(consumer(q, i)) for i in range(2)]
    await asyncio.gather(*producers)
    for _ in consumers:
        await q.put(None)  # stop all consumers
    await asyncio.gather(*consumers)

if __name__ == "__main__":
    asyncio.run(main())
