from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    ticker: str
    quant_data: Optional[Dict[str, Any]]
    news_headlines: Optional[List[str]]
    sentiment_score: Optional[float]
    sentiment_reasoning: Optional[str]
    signal: Optional[str]  # Buy/Hold/Sell
    reasoning: Optional[str]
