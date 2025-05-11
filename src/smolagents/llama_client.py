import asyncio
import httpx
import time
import sys
import json


class Client:
    def __init__(self, base_url):
        self.base_url = base_url
        self.completions_url = f"{self.base_url}/v1/chat/completions"

    def check_health(self):
        async def _check_health():
            async with httpx.AsyncClient() as client:
                response = await client.get(self.base_url + "/health", timeout=3)
                return response.json()
        return asyncio.run(_check_health())

    async def stream(self, messages):
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", self.completions_url, json=messages, timeout=None) as response:
                if response.status_code == 200:
                    start_time = time.time()
                    count = 0
                    async for chunk in response.aiter_bytes():
                        temp = chunk.decode().replace('\n', '', 1)
                        try:
                            token = json.loads(temp[6:])["choices"][0]["delta"]["content"]
                        except Exception:
                            print(temp, file=sys.stderr, flush=True)
                        print(token, end="", flush=True)
                        count += 1
                        end_time = time.time()
                    duration = end_time - start_time
                    print(f"Received {count} chunks in {duration:.2f} seconds", file=sys.stderr)
                else:
                    print(f"Error: {response.status_code} - {(await response.aread()).decode()}",
                          file=sys.stderr)

    async def post(self, messages):
        async with httpx.AsyncClient() as client:
            response = await client.post(self.completions_url, json=messages, timeout=None)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Could not get response"}
