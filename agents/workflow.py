from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.tools import fetch_quant_data, fetch_news_headlines, fetch_and_download_pdf
from agents.sentiment_agent import analyze_sentiment
from agents.master_agent import analyze_quant_and_sentiment

from agents.rag import store_news_in_qdrant, has_pdf_for_ticker, store_pdf_in_qdrant

import asyncio

async def data_ingestion_node(state: AgentState) -> AgentState:
    """Fetches quantitative data, news headlines, and deep PDF fundamentals."""
    ticker = state["ticker"]
    
    # Fetch quantitative data, news, and check PDF cache CONCURRENTLY
    quant_data, news, has_pdf = await asyncio.gather(
        fetch_quant_data(ticker),
        fetch_news_headlines(ticker),
        asyncio.to_thread(has_pdf_for_ticker, ticker)
    )
    
    state["quant_data"] = quant_data
    state["news_headlines"] = news
    
    # Handle the Qdrant DB storing and PDF downloads concurrently
    async def handle_news_storage():
        try:
            await asyncio.to_thread(store_news_in_qdrant, ticker, news)
        except Exception as e:
            print(f"Error storing news in Qdrant: {e}")
            
    async def handle_pdf_storage():
        if has_pdf:
            print(f"Autonomous Agent: PDF data already in Qdrant memory for {ticker}. Skipping download!")
        else:
            pdf_path = await fetch_and_download_pdf(ticker)
            if pdf_path:
                try:
                    await asyncio.to_thread(store_pdf_in_qdrant, ticker, pdf_path)
                except Exception as e:
                    print(f"Error storing PDF in Qdrant: {e}")

    # Fire off both storage tasks simultaneously
    await asyncio.gather(
        handle_news_storage(),
        handle_pdf_storage()
    )
    
    return state

# Define a conditional router
def route_after_data(state: AgentState) -> str:
    """Routes to END if data ingestion failed, otherwise proceeds to sentiment."""
    if "error" in state.get("quant_data", {}):
        return "end"
    return "sentiment_analysis"

# Define a new graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("data_ingestion", data_ingestion_node)
workflow.add_node("sentiment_analysis", analyze_sentiment)
workflow.add_node("master_quant", analyze_quant_and_sentiment)

# Define edges
workflow.set_entry_point("data_ingestion")

# Replace static edge with conditional edge
workflow.add_conditional_edges(
    "data_ingestion",
    route_after_data,
    {
        "sentiment_analysis": "sentiment_analysis",
        "end": END
    }
)

workflow.add_edge("sentiment_analysis", "master_quant")
workflow.add_edge("master_quant", END)

# Compile the graph
app = workflow.compile()
