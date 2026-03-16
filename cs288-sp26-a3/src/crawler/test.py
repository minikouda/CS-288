import aiohttp
import asyncio

async def test():
    url = "https://www2.eecs.berkeley.edu/Pubs/TechRpts/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            async with session.get(url, timeout=10) as resp:
                print(resp.status)
                text = await resp.text()
                print(text[:500])  # first 500 chars
        except Exception as e:
            print("Error:", e)

asyncio.run(test())