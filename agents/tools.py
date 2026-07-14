import yfinance as yf
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List

REGION_SUFFIXES = {
    "India": (".NS", ".BO"),
    "UK": (".L",),
    "Canada": (".TO", ".V"),
    "Australia": (".AX",),
    "Germany": (".DE",),
    "France": (".PA",),
    "Japan": (".T",),
    "China": (".SS", ".SZ"),
    "Brazil": (".SA",),
    "US": ()
}

def resolve_ticker(query: str, region: str = "India") -> str:
    """Uses Yahoo Finance Search API to resolve a company name to a ticker."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        quotes = data.get("quotes", [])
        
        if not quotes:
            return query
            
        if region and region in REGION_SUFFIXES:
            suffixes = REGION_SUFFIXES[region]
            if suffixes:
                for q in quotes:
                    symbol = q.get("symbol", "")
                    if any(symbol.endswith(sfx) for sfx in suffixes):
                        return symbol
                
                # If a specific region was requested but no matching stock was found, return the query
                # so that yfinance properly fails instead of silently returning a stock from another country.
                return query
            else:
                # For US or regions with no specific suffix, return the top match
                return quotes[0].get("symbol", query)
                    
        return quotes[0].get("symbol", query)
    except Exception as e:
        print(f"Error resolving ticker: {e}")
        return query

def fetch_quant_data(ticker: str) -> Dict[str, Any]:
    """Fetches quantitative data from yfinance for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return {"error": "No data found for ticker"}

        current_price = float(hist['Close'].iloc[-1])
        volume = int(hist['Volume'].iloc[-1])
        high_52w = float(hist['High'].max())
        low_52w = float(hist['Low'].min())
        
        # Calculate Moving Averages
        ma_50 = float(hist['Close'].tail(50).mean())
        ma_200 = float(hist['Close'].tail(200).mean()) if len(hist) >= 200 else None
        
        # RSI calculation (14 periods)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not rsi.empty and not rsi.isna().iloc[-1] else None
        
        # MACD calculation
        ema_12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        current_macd = float(macd.iloc[-1]) if not macd.empty else None
        current_macd_signal = float(signal_line.iloc[-1]) if not signal_line.empty else None

        return {
            "current_price": current_price,
            "volume": volume,
            "52_week_high": high_52w,
            "52_week_low": low_52w,
            "50_day_ma": ma_50,
            "200_day_ma": ma_200,
            "rsi_14": current_rsi,
            "macd": current_macd,
            "macd_signal": current_macd_signal
        }
    except Exception as e:
        return {"error": str(e)}

def fetch_news_headlines(ticker: str) -> List[str]:
    """Scrapes recent news headlines from Google News for the ticker."""
    url = f"https://news.google.com/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Google News titles are usually within <a> tags with class 'JtKRv' or similar, 
        # but class names change. It's safer to find all 'a' tags or specific h3/h4 tags.
        # As a heuristic, we look for <a> tags inside h4 or a specific common class.
        headlines = []
        for h in soup.find_all('a', class_='JtKRv'):
            text = h.get_text(strip=True)
            if text and len(text) > 20:
                headlines.append(text)
                
        # If class changes, fallback to finding 'a' tags with text length > 30 inside div
        if not headlines:
            for a in soup.find_all('a'):
                text = a.get_text(strip=True)
                if len(text) > 30 and ticker.lower() in text.lower():
                    headlines.append(text)
                    
        return headlines[:10]  # Return top 10 headlines
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []
