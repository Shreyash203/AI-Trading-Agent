import asyncio
from run_elite import run_backtest

async def main():
    await run_backtest('ETERNAL.NS', 10000.0)

if __name__ == '__main__':
    asyncio.run(main())
