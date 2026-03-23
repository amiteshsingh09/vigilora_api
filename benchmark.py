import os
import sys
import time
import asyncio

# add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.routes.legacy_routes import analyze_stored
from app.main import lifespan
from fastapi import FastAPI

async def run():
    app = FastAPI()
    print("Preloading models via lifespan...")
    async with lifespan(app):
        print("Models preloaded. Simulating first request...")
        t0 = time.time()
        res = await analyze_stored()
        t1 = time.time()
        print(f"First request time: {t1 - t0:.2f}s")
        
        print("Simulating second request...")
        t2 = time.time()
        res2 = await analyze_stored()
        t3 = time.time()
        print(f"Second request time: {t3 - t2:.2f}s")

if __name__ == "__main__":
    asyncio.run(run())
