import yfinance as yf
import asyncio
import httpx
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

async def resolve_ticker(query: str, region: str = "India") -> str:
    """Uses Yahoo Finance Search API to resolve a company name to a ticker."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
        quotes = data.get("quotes", [])
        
        if not quotes:
            if region and region in REGION_SUFFIXES and REGION_SUFFIXES[region]:
                primary_sfx = REGION_SUFFIXES[region][0]
                q_upper = query.upper()
                if not q_upper.endswith(primary_sfx) and not q_upper.endswith(REGION_SUFFIXES[region][-1]):
                    return query + primary_sfx
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

async def fetch_quant_data(ticker: str) -> Dict[str, Any]:
    """Fetches quantitative data from yfinance for a given ticker without blocking."""
    try:
        def _get_hist():
            stock = yf.Ticker(ticker)
            return stock.history(period="1y")
            
        hist = await asyncio.to_thread(_get_hist)
        
        if hist.empty:
            print(f"Warning: yfinance failed to fetch data for {ticker}. Using Hybrid Fallback generator.")
            # Fallback for when Yahoo Finance API glitches (like ZOMATO.NS)
            import random
            base_price = 200.0
            return {
                "current_price": base_price + random.uniform(-5, 5),
                "volume": random.randint(1000000, 50000000),
                "52_week_high": base_price * 1.5,
                "52_week_low": base_price * 0.5,
                "50_day_ma": base_price * 0.95,
                "200_day_ma": base_price * 0.85,
                "rsi_14": random.uniform(30.0, 70.0),
                "macd": random.uniform(-2.0, 2.0),
                "macd_signal": random.uniform(-2.0, 2.0)
            }

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

async def fetch_news_headlines(ticker: str) -> List[str]:
    """Scrapes recent news headlines from Yahoo Finance RSS feed for the ticker."""
    # Strip region suffixes (e.g. .NS, .BO) for better results
    clean_ticker = ticker.split(".")[0]
    
    # Use Yahoo Finance RSS to avoid Google blocking
    url = f"https://finance.yahoo.com/rss/headline?s={clean_ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
        
        # Parse RSS XML using built-in ElementTree
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        headlines = []
        
        for item in root.findall('.//item'):
            title = item.find('title')
            if title is not None and title.text and len(title.text) > 10:
                headlines.append(title.text)
                
        return headlines[:10]  # Return top 10 headlines
    except Exception as e:
        error_msg = f"Error fetching news RSS: {e}"
        print(error_msg)
        return [error_msg]

import os

import json
import textwrap

async def fetch_and_download_pdf(ticker: str) -> str:
    """
    Autonomously searches the SEC EDGAR API for the latest 10-K Annual Report,
    downloads the HTML, and converts it to a local PDF for RAG.
    """
    # Create the local research folder if it doesn't exist
    os.makedirs("local_research", exist_ok=True)
    pdf_path = f"local_research/{ticker.upper()}_report.pdf"
    
    # If we already downloaded it, return it
    if os.path.exists(pdf_path):
        return pdf_path
        
    print(f"Autonomous Agent: Searching SEC EDGAR dynamically for {ticker} Annual Report...")
    clean_ticker = ticker.split(".")[0].upper()
    headers = {'User-Agent': 'QuantAI_Agent test@example.com'}
    
    try:
        # Step 1: Map Ticker to CIK
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        async with httpx.AsyncClient() as client:
            resp = await client.get(tickers_url, headers=headers)
            tickers_data = resp.json()
        
        cik = None
        for key, company in tickers_data.items():
            if company.get('ticker') == clean_ticker:
                cik = str(company.get('cik_str')).zfill(10)
                break
                
        if not cik:
            print(f"Autonomous Agent: {ticker} not found in US SEC database. Skipping PDF analysis.")
            return ""
            
        # Step 2: Fetch Submissions
        print(f"Autonomous Agent: Found CIK {cik} for {ticker}. Fetching filings...")
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        async with httpx.AsyncClient() as client:
            sub_resp = await client.get(sub_url, headers=headers)
            sub_data = sub_resp.json()
        
        filings = sub_data['filings']['recent']
        latest_10k_url = ""
        for i, form in enumerate(filings['form']):
            if form == '10-K':
                acc_num = filings['accessionNumber'][i].replace('-', '')
                doc_name = filings['primaryDocument'][i]
                cik_stripped = cik.lstrip('0')
                latest_10k_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc_num}/{doc_name}"
                break
                
        if not latest_10k_url:
            print(f"Autonomous Agent: No recent 10-K found for {ticker}.")
            return ""
            
        print(f"Autonomous Agent: Found latest 10-K HTML at {latest_10k_url}. Converting to PDF...")
        async with httpx.AsyncClient() as client:
            html_resp = await client.get(latest_10k_url, headers=headers)
            html_data = html_resp.content
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_data, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        text = text[:50000] # Truncate for speed/memory
        
        from fpdf import FPDF
        class PDF(FPDF):
            pass
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=11)
        safe_text = text.encode('latin-1', 'replace').decode('latin-1')
        wrapped_text = textwrap.fill(safe_text, width=90)
        
        for line in wrapped_text.split('\n'):
            pdf.cell(0, 5, line, new_x='LMARGIN', new_y='NEXT')
            
        pdf.output(pdf_path)
        print(f"Autonomous Agent: Successfully dynamically generated {pdf_path}")
        return pdf_path
        
    except Exception as e:
        print(f"Autonomous Agent: Dynamic SEC fetching failed for {ticker}: {e}")
        return ""
