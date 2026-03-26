import asyncio

class AsyncRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    async def __aenter__(self):
        print(f"Entering async range {self.start}-{self.end}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Exiting async range")

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.start >= self.end:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)
        val = self.start
        self.start += 1
        return val

async def main():
    async with AsyncRange(1, 5) as ar:
        async for num in ar:
            print(num)

asyncio.run(main())
