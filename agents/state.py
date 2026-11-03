from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    ticker: str
    quant_data: Optional[Dict[str, Any]]
    news_headlines: Optional[List[Dict[str, Any]]]
    sentiment_score: Optional[float]
    sentiment_reasoning: Optional[str]
    signal: Optional[str]  # Buy/Hold/Sell
    reasoning: Optional[str]

    sentiment_status: str
    sentiment_evidence_ids: List[str]
    news_evidence: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    cited_evidence_ids: List[str]
    analysis_status: str
    warnings: List[str]
