try:
    import aiohttp
    import asyncio
    from urllib.parse import urljoin
except ImportError:
    print("aiohttp not installed, skipping")
    exit(0)

async def fetch(session, url):
    try:
        async with session.get(url, timeout=5) as response:
            return url, response.status, len(await response.text())
    except Exception as e:
        return url, None, str(e)

async def crawl(urls, limit=5):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls[:limit]]
        for coro in asyncio.as_completed(tasks):
            url, status, size = await coro
            print(f"{url}: status={status}, size={size}")

urls = [
    "http://httpbin.org/get",
    "http://httpbin.org/headers",
    "http://httpbin.org/ip",
    "http://httpbin.org/user-agent",
]
asyncio.run(crawl(urls))
