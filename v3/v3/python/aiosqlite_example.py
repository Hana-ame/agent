try:
    import aiosqlite
    import asyncio
except ImportError:
    print("aiosqlite not installed, skipping")
    exit(0)

async def main():
    async with aiosqlite.connect(":memory:") as db:
        await db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 30))
        await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Bob", 25))
        await db.commit()

        async with db.execute("SELECT * FROM users") as cursor:
            async for row in cursor:
                print(f"User: {row[1]} ({row[2]})")

asyncio.run(main())
