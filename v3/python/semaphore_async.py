import asyncio
import random

sem = asyncio.Semaphore(2)

async def task(name):
    async with sem:
        print(f"{name} acquired semaphore")
        await asyncio.sleep(random.uniform(0.5, 1))
        print(f"{name} released")
        return name

async def main():
    tasks = [asyncio.create_task(task(f"Task{i}")) for i in range(5)]
    results = await asyncio.gather(*tasks)
    print("All done:", results)

asyncio.run(main())
