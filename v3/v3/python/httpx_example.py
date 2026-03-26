try:
    import httpx
    import asyncio
except ImportError:
    print("httpx not installed, skipping")
    exit(0)

async def main():
    async with httpx.AsyncClient() as client:
        response = await client.get('https://httpbin.org/get')
        print(f"Status: {response.status_code}")
        print(f"JSON: {response.json()}")

asyncio.run(main())
