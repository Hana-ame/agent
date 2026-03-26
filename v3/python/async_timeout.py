import asyncio
import functools

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                return f"Timeout after {seconds}s"
        return wrapper
    return decorator

@timeout(1)
async def slow_operation():
    await asyncio.sleep(2)
    return "Done"

async def main():
    result = await slow_operation()
    print(result)

asyncio.run(main())
