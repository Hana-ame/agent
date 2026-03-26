import asyncio
import time

def blocking_io():
    time.sleep(1)
    return "Blocking result"

async def main():
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, blocking_io)
    print(result)

asyncio.run(main())
