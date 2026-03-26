import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource...")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource...")
        await asyncio.sleep(0.1)

    async def do_work(self):
        print("Working...")
        await asyncio.sleep(0.2)

async def main():
    async with AsyncResource() as resource:
        await resource.do_work()

if __name__ == "__main__":
    asyncio.run(main())
