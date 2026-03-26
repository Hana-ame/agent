import asyncio

class AsyncCounter:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.start >= self.end:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)
        self.start += 1
        return self.start - 1

async def main():
    async for i in AsyncCounter(1, 5):
        print(i)

asyncio.run(main())
