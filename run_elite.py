import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import asyncio
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

class TradeSignal(BaseModel):
    signal: str = Field(description="Strictly BUY, HOLD, or SELL")
    reasoning: str = Field(description="One sentence explaining the technical logic.")

def calculate_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    return df

async def run_backtest(ticker, initial_capital=10000.0):
    print(f"\n?? Starting ELITE AI Backtest for {ticker}")
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    df = calculate_indicators(df)
    df = df.dropna()
    test_data = df.tail(30)
    cash = initial_capital
    shares = 0
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=TradeSignal)
    prompt = PromptTemplate(
        input_variables=["price", "sma50", "rsi", "macd", "macd_sig", "bb_lower"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template=(
            "You are an elite quantitative hedge fund algorithm. Your #1 rule is Capital Preservation.\n"
            "Analyze today's technical data:\n"
            "- Price: ${price}\n"
            "- 50-Day MA: ${sma50}\n"
            "- RSI (14): {rsi}\n"
            "- MACD: {macd} (Signal: {macd_sig})\n"
            "- Lower Bollinger Band: ${bb_lower}\n\n"
            "TRADING RULES:\n"
            "1. ONLY output BUY if RSI is < 40 AND MACD crosses above the Signal line (or Price touches Lower Bollinger Band).\n"
            "2. ONLY output SELL if RSI is > 65 OR if Price severely breaks below the 50-Day MA.\n"
            "3. If conditions are not perfectly met, output HOLD. Do not force trades.\n\n"
            "{format_instructions}"
        )
    )
    chain = prompt | llm | parser

    for date, row in test_data.iterrows():
        price = float(row['Close'].iloc[0]) if isinstance(row['Close'], pd.Series) else float(row['Close'])
        sma50 = float(row['SMA_50'].iloc[0]) if isinstance(row['SMA_50'], pd.Series) else float(row['SMA_50'])
        rsi = float(row['RSI'].iloc[0]) if isinstance(row['RSI'], pd.Series) else float(row['RSI'])
        macd = float(row['MACD'].iloc[0]) if isinstance(row['MACD'], pd.Series) else float(row['MACD'])
        macd_sig = float(row['MACD_Signal'].iloc[0]) if isinstance(row['MACD_Signal'], pd.Series) else float(row['MACD_Signal'])
        bb_lower = float(row['BB_Lower'].iloc[0]) if isinstance(row['BB_Lower'], pd.Series) else float(row['BB_Lower'])
        
        try:
            result = await chain.ainvoke({"price": round(price, 2), "sma50": round(sma50, 2), "rsi": round(rsi, 2), "macd": round(macd, 2), "macd_sig": round(macd_sig, 2), "bb_lower": round(bb_lower, 2)})
            signal = result.signal.upper()
        except Exception:
            signal = "HOLD"

        date_str = date.strftime("%Y-%m-%d")
        if signal == "BUY" and cash >= price:
            shares_bought = int(cash // price)
            cash -= shares_bought * price
            shares += shares_bought
            print(f"[{date_str}] ?? BUY  | Bought {shares_bought} shares @ $price:.2f")
        elif signal == "SELL" and shares > 0:
            cash += shares * price
            print(f"[{date_str}] ?? SELL | Sold {shares} shares @ $price:.2f")
            shares = 0
        else:
            pass # Keep terminal clean, only print trades

    final_value = cash + (shares * price)
    profit = final_value - initial_capital
    print("="*40)
    print(f"?? FINAL RESULTS: {ticker}")
    print(f"Value:  $final_value:,.2f")
    print(f"Profit: $profit:,.2f ({(profit/initial_capital)*100:.2f}%)")
    print("="*40)

async def main():
    await run_backtest('PINELABS.NS')
    await run_backtest('TATACAP.NS')

if __name__ == '__main__':
    asyncio.run(main())
