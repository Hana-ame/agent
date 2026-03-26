import asyncio
import random

async def worker(name, delay):
    print(f"{name} started, will take {delay}s")
    await asyncio.sleep(delay)
    print(f"{name} finished")
    return name

async def main():
    tasks = [asyncio.create_task(worker(f"Task{i}", random.uniform(0.5, 1.5))) for i in range(5)]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    print(f"First completed: {done.pop().result()}")
    for task in pending:
        task.cancel()
        print(f"Cancelled {task.get_name()}")
    results = await asyncio.gather(*[worker(f"Gather{i}", 0.3) for i in range(3)], return_exceptions=True)
    print("Gather results:", results)

if __name__ == "__main__":
    asyncio.run(main())
