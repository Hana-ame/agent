import asyncio

async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)

async def main():
    print("start")
    await asyncio.gather(
        say_after(1, "hello"),
        say_after(2, "world")
    )
    print("end")

if __name__ == "__main__":
    asyncio.run(main())
