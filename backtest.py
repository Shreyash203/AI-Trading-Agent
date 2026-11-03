import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

class TradeSignal(BaseModel):
    signal: str = Field(description="BUY, HOLD, or SELL")

def calculate_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

async def run_backtest(ticker, initial_capital=10000.0):
    print(f"\n?? Starting AI Backtest for {ticker}")
    print(f"?? Initial Capital: $initial_capital:,.2f\n")
    
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    df = calculate_indicators(df)
    df = df.dropna()
    
    test_data = df.tail(30)
    
    cash = initial_capital
    shares = 0
    
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=TradeSignal)
    prompt = PromptTemplate(
        input_variables=["current_price", "sma_50", "sma_200", "rsi"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template=(
            "You are a quantitative trading algorithm. Analyze today's data:\n"
            "Price: ${current_price}\n"
            "50-Day MA: ${sma_50}\n"
            "200-Day MA: ${sma_200}\n"
            "RSI: {rsi}\n\n"
            "Output a deterministic signal based on technical analysis.\n"
            "{format_instructions}"
        )
    )
    chain = prompt | llm | parser

    for date, row in test_data.iterrows():
        current_price = float(row['Close'].iloc[0]) if isinstance(row['Close'], pd.Series) else float(row['Close'])
        
        try:
            result = await chain.ainvoke({
                "current_price": round(current_price, 2),
                "sma_50": round(float(row['SMA_50'].iloc[0] if isinstance(row['SMA_50'], pd.Series) else row['SMA_50']), 2),
                "sma_200": round(float(row['SMA_200'].iloc[0] if isinstance(row['SMA_200'], pd.Series) else row['SMA_200']), 2),
                "rsi": round(float(row['RSI'].iloc[0] if isinstance(row['RSI'], pd.Series) else row['RSI']), 2)
            })
            signal = result.signal.upper()
        except Exception as e:
            signal = "HOLD"

        date_str = date.strftime("%Y-%m-%d")
        
        if signal == "BUY" and cash >= current_price:
            shares_bought = int(cash // current_price)
            cash -= shares_bought * current_price
            shares += shares_bought
            print(f"[{date_str}] AI Signal: BUY  | Bought {shares_bought} shares at $current_price:.2f")
        elif signal == "SELL" and shares > 0:
            cash += shares * current_price
            print(f"[{date_str}] AI Signal: SELL | Sold {shares} shares at $current_price:.2f")
            shares = 0
        else:
            print(f"[{date_str}] AI Signal: HOLD | Price: $current_price:.2f")

    final_portfolio_value = cash + (shares * current_price)
    profit = final_portfolio_value - initial_capital
    return_pct = (profit / initial_capital) * 100
    
    print("\n" + "="*40)
    print(f"?? BACKTEST RESULTS FOR {ticker}")
    print("="*40)
    print(f"Final Portfolio Value: $final_portfolio_value:,.2f")
    print(f"Total Profit/Loss:     $profit:,.2f ({return_pct:.2f}%)")
    print("="*40)

if __name__ == '__main__':
    import asyncio
    asyncio.run(run_backtest('AAPL', 10000.0))
    asyncio.run(run_backtest('RELIANCE.NS', 10000.0))
