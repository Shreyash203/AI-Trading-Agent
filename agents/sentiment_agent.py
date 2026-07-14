import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from agents.state import AgentState

def analyze_sentiment(state: AgentState) -> AgentState:
    """Analyzes the sentiment of the news headlines using Groq LLM."""
    ticker = state["ticker"]
    headlines = state.get("news_headlines", [])
    
    if not headlines:
        state["sentiment_score"] = 0.0
        state["sentiment_reasoning"] = "No recent news headlines found."
        return state

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Updated to latest model
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.0
    )
    
    prompt = PromptTemplate(
        input_variables=["ticker", "headlines"],
        template=(
            "You are a financial sentiment analyst. "
            "Analyze the following recent news headlines for the stock ticker {ticker}.\n\n"
            "Headlines:\n{headlines}\n\n"
            "Provide your response in two parts separated by a pipe character (|).\n"
            "1. A sentiment score between -1.0 (extremely negative) and 1.0 (extremely positive).\n"
            "2. A brief reasoning (1-2 sentences) for the score.\n"
            "Format: SCORE|REASONING"
        )
    )
    
    chain = prompt | llm
    
    headlines_str = "\n".join([f"- {h}" for h in headlines])
    response = chain.invoke({"ticker": ticker, "headlines": headlines_str})
    
    output = response.content.strip()
    
    try:
        parts = output.split("|")
        score = float(parts[0].strip())
        reasoning = parts[1].strip() if len(parts) > 1 else "No reasoning provided."
    except ValueError:
        score = 0.0
        reasoning = f"Failed to parse sentiment. Output was: {output}"
        
    state["sentiment_score"] = score
    state["sentiment_reasoning"] = reasoning
    
    return state
