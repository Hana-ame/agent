try:
    import asyncio
    import aiofiles
    import tempfile
except ImportError:
    print("aiofiles not installed, skipping")
    exit(0)

async def main():
    async with aiofiles.tempfile.NamedTemporaryFile(mode='w+') as tmp:
        await tmp.write("Hello async file")
        await tmp.seek(0)
        content = await tmp.read()
        print("Content:", content)

asyncio.run(main())
