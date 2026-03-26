import contextvars
import asyncio

user = contextvars.ContextVar('user')

async def handler():
    print(f"Handler: {user.get()}")

async def main():
    user.set("Alice")
    await handler()
    # 在另一个任务中，上下文独立
    await asyncio.create_task(handler())

if __name__ == "__main__":
    asyncio.run(main())
