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
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"q": query}, headers=headers)
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
            import yfinance as yf
            stock = yf.Ticker(ticker)
            return stock.history(period="1y")

        hist = await asyncio.to_thread(_get_hist)

        if hist.empty:
            return {"error": "Market data unavailable", "data_status": "unavailable"}

        if len(hist) < 50:
            return {"error": "Insufficient price history", "data_status": "unavailable"}

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

        # Bollinger Bands (Volatility)
        bb_mid = hist['Close'].rolling(window=20).mean()
        bb_std = hist['Close'].rolling(window=20).std()
        bb_lower = bb_mid - (2 * bb_std)
        current_bb_lower = float(bb_lower.iloc[-1]) if not bb_lower.empty else None

        return {
            "current_price": current_price,
            "volume": volume,
            "52_week_high": high_52w,
            "52_week_low": low_52w,
            "50_day_ma": ma_50,
            "200_day_ma": ma_200,
            "rsi_14": current_rsi,
            "macd": current_macd,
            "macd_signal": current_macd_signal,
            "previous_macd": float(macd.iloc[-2]) if len(macd) > 1 else None,
            "previous_macd_signal": float(signal_line.iloc[-2]) if len(signal_line) > 1 else None,
            "as_of": hist.index[-1].isoformat(),
            "source": "Yahoo Finance via yfinance",
            "bb_lower": current_bb_lower
        }
    except Exception as e:
        return {"error": str(e)}

async def fetch_news_headlines(ticker: str) -> List[dict]:
    """Scrapes recent news headlines from Yahoo Finance RSS feed for the ticker."""
    # Strip region suffixes (e.g. .NS, .BO) for better results
    clean_ticker = ticker

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
                link = item.findtext('link')
                published = item.findtext('pubDate')
                if link and published:
                    from email.utils import parsedate_to_datetime
                    from datetime import timezone
                    try:
                        date = parsedate_to_datetime(published)
                        if date.tzinfo is None:
                            date = date.replace(tzinfo=timezone.utc)
                        headlines.append({'title': title.text, 'url': link, 'published_at': date.isoformat()})
                    except (ValueError, TypeError):
                        continue

        return headlines[:10]  # Return top 10 headlines
    except Exception as e:
        error_msg = f"Error fetching news RSS: {e}"
        print(error_msg)
        return []

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
    import time
    metadata_path = os.path.splitext(pdf_path)[0] + '.json'
    if os.path.exists(pdf_path) and os.path.exists(metadata_path) and time.time()-os.path.getmtime(metadata_path) < 86400:
        return pdf_path
    if '.' in ticker:
        return ''

    print(f"Autonomous Agent: Searching SEC EDGAR dynamically for {ticker} Annual Report...")
    clean_ticker = ticker.split(".")[0].upper()
    headers = {'User-Agent': os.getenv('SEC_USER_AGENT', '')}
    if not headers['User-Agent']:
        return ''

    try:
        # Step 1: Map Ticker to CIK
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        async with httpx.AsyncClient() as client:
            resp = await client.get(tickers_url, headers=headers)
            resp.raise_for_status()
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
            sub_resp.raise_for_status()
        sub_data = sub_resp.json()

        filings = sub_data['filings']['recent']
        latest_10k_url = ""
        for i, form in enumerate(filings['form']):
            if form == '10-K':
                acc_num = filings['accessionNumber'][i].replace('-', '')
                doc_name = filings['primaryDocument'][i]
                filing_date = filings["filingDate"][i] + "T00:00:00Z"
                cik_stripped = cik.lstrip('0')
                latest_10k_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc_num}/{doc_name}"
                break

        if not latest_10k_url:
            print(f"Autonomous Agent: No recent 10-K found for {ticker}.")
            return ""

        print(f"Autonomous Agent: Found latest 10-K HTML at {latest_10k_url}. Converting to PDF...")
        async with httpx.AsyncClient() as client:
            html_resp = await client.get(latest_10k_url, headers=headers)
            html_resp.raise_for_status()
        html_data = html_resp.content

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_data, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)


        def write_pdf():
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
            with open(metadata_path, 'w', encoding='utf-8') as meta:
                json.dump({'url': latest_10k_url, 'filing_date': filing_date, 'accession': acc_num, 'content_hash': __import__('hashlib').sha256(text.encode('utf-8')).hexdigest()}, meta)
        await asyncio.to_thread(write_pdf)
        print(f"Autonomous Agent: Successfully dynamically generated {pdf_path}")
        return pdf_path

    except Exception as e:
        print(f"Autonomous Agent: Dynamic SEC fetching failed for {ticker}: {e}")
        return ""
