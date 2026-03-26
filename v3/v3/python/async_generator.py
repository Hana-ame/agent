import asyncio

async def async_counter(start, end):
    for i in range(start, end):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for num in async_counter(1, 5):
        print(f"Async yielded: {num}")

asyncio.run(main())
