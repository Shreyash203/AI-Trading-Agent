import asyncio
from backtest import run_backtest

async def main():
    await run_backtest('CAMPUS.NS', 10000.0)

if __name__ == '__main__':
    asyncio.run(main())
