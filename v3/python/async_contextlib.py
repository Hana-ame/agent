import asyncio
import contextlib

@contextlib.asynccontextmanager
async def managed_resource():
    print("Acquiring resource")
    resource = "resource"
    try:
        yield resource
    finally:
        print("Releasing resource")

async def main():
    async with managed_resource() as res:
        print(f"Using {res}")
        await asyncio.sleep(0.1)

asyncio.run(main())
