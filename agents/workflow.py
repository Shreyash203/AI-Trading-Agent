from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.tools import fetch_quant_data, fetch_news_headlines
from agents.sentiment_agent import analyze_sentiment
from agents.master_agent import analyze_quant_and_sentiment

def data_ingestion_node(state: AgentState) -> AgentState:
    """Fetches quantitative data and news headlines."""
    ticker = state["ticker"]
    
    # Fetch data
    state["quant_data"] = fetch_quant_data(ticker)
    state["news_headlines"] = fetch_news_headlines(ticker)
    
    return state

# Define a new graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("data_ingestion", data_ingestion_node)
workflow.add_node("sentiment_analysis", analyze_sentiment)
workflow.add_node("master_quant", analyze_quant_and_sentiment)

# Define edges
workflow.set_entry_point("data_ingestion")
workflow.add_edge("data_ingestion", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", "master_quant")
workflow.add_edge("master_quant", END)

# Compile the graph
app = workflow.compile()
