try:
    import asyncio
    import aiohttp
except ImportError:
    print("aiohttp not installed, skipping")
    exit(0)

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = ['http://httpbin.org/get', 'http://httpbin.org/headers']
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        for url, content in zip(urls, results):
            print(f"{url}: {len(content)} chars")

asyncio.run(main())
