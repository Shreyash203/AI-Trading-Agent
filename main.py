from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from agents.workflow import app as workflow_app
from agents.tools import resolve_ticker

app = FastAPI(title="AI Financial Sentiment & Algo-Trading Engine", version="1.0.0")

from enum import Enum

class Region(str, Enum):
    INDIA = "India"
    US = "US"
    UK = "UK"
    CANADA = "Canada"
    AUSTRALIA = "Australia"
    GERMANY = "Germany"
    FRANCE = "France"
    JAPAN = "Japan"
    CHINA = "China"
    BRAZIL = "Brazil"

class AnalysisResponse(BaseModel):
    stock_symbol: str
    signal: str
    sentiment_score: float
    reasoning: str

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_ticker(company_name: str, region: Region = Region.INDIA):
    """
    Analyzes a given stock ticker or company name and returns a Buy/Hold/Sell signal.
    """
    try:
        # Resolve friendly name to actual ticker symbol
        resolved_ticker = resolve_ticker(company_name, region=region)
        
        # Ensure ticker is uppercase
        resolved_ticker = resolved_ticker.upper()
        
        initial_state = {"ticker": resolved_ticker}
        
        result = workflow_app.invoke(initial_state)
        
        return AnalysisResponse(
            stock_symbol=result.get("ticker", resolved_ticker),
            signal=result.get("signal", "ERROR"),
            sentiment_score=result.get("sentiment_score", 0.0),
            reasoning=result.get("reasoning", "No reasoning provided.")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
