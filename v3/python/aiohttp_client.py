try:
    import aiohttp
    import asyncio
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("aiohttp not installed, skipping demo")
    exit(0)

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, 'http://example.com')
        print(f"Fetched {len(html)} bytes")

asyncio.run(main())
