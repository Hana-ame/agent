import asyncio
import contextlib

@contextlib.asynccontextmanager
async def timed(name):
    import time
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{name} took {elapsed:.2f}s")

async def main():
    async with timed("sleep"):
        await asyncio.sleep(0.5)

asyncio.run(main())
