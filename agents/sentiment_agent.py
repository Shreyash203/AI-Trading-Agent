import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from agents.state import AgentState

from agents.rag import retrieve_relevant_news

import asyncio

async def analyze_sentiment(state: AgentState) -> AgentState:
    """Analyzes the sentiment of the news headlines using Groq LLM."""
    ticker = state["ticker"]
    
    # Retrieve relevant news context using RAG
    try:
        headlines_str = await asyncio.to_thread(
            retrieve_relevant_news, ticker, query="bullish bearish financial outlook performance news events", limit=10
        )
    except Exception as e:
        headlines_str = f"Error retrieving context: {e}"
    
    if not headlines_str or headlines_str.startswith("No historical") or headlines_str.startswith("No relevant") or "Error" in headlines_str:
        state["sentiment_score"] = 0.02 # Debugging value
        state["sentiment_reasoning"] = f"RAG DEBUG INFO: {headlines_str}"
        return state

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Updated to latest model
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.5
    )
    
    prompt = PromptTemplate(
        input_variables=["ticker", "headlines"],
        template=(
            "You are a financial sentiment analyst. "
            "Analyze the following recent news headlines for the stock ticker {ticker}.\n\n"
            "Headlines:\n{headlines}\n\n"
            "Provide your response in two parts separated by a pipe character (|).\n"
            "1. A highly precise, fractional sentiment score between -1.0 (extremely negative) and 1.0 (extremely positive). "
            "Do NOT output exactly 0.0. Pick a nuanced fraction (e.g., 0.15, -0.3, 0.6) that reflects the subtle overall tone.\n"
            "2. A brief reasoning (1-2 sentences) for the score.\n"
            "Format: SCORE|REASONING"
        )
    )
    
    chain = prompt | llm
    
    # We already have headlines_str from RAG
    response = await chain.ainvoke({"ticker": ticker, "headlines": headlines_str})
    
    output = response.content.strip()
    
    import re
    try:
        parts = output.split("|")
        # Extract the first floating point number from the first part
        match = re.search(r"[-+]?\d*\.\d+|\d+", parts[0])
        if match:
            score = float(match.group(0))
        else:
            score = 0.01 # Force non-zero fallback
        reasoning = parts[1].strip() if len(parts) > 1 else "No reasoning provided."
    except Exception as e:
        score = 0.01
        reasoning = f"Failed to parse sentiment. Output was: {output}. Error: {e}"
        
    # Strictly prevent flat 0.0
    if score == 0.0:
        score = 0.01
        
    state["sentiment_score"] = score
    state["sentiment_reasoning"] = reasoning
    
    return state
