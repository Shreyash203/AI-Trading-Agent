import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from agents.state import AgentState

from agents.rag import retrieve_relevant_news
import asyncio

class SentimentOutput(BaseModel):
    score: float = Field(description="A highly precise, fractional sentiment score between -1.0 (extremely negative) and 1.0 (extremely positive). Do NOT output exactly 0.0.")
    reasoning: str = Field(description="A brief reasoning (1-2 sentences) for the score.")

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
        model="openai/gpt-oss-20b",  # Switched to OpenAI OSS model for context handling
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.5
    )
    
    parser = PydanticOutputParser(pydantic_object=SentimentOutput)
    
    prompt = PromptTemplate(
        input_variables=["ticker", "headlines"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template=(
            "You are a financial sentiment analyst.\n"
            "Analyze the following recent news headlines for the stock ticker {ticker}.\n\n"
            "Headlines:\n{headlines}\n\n"
            "{format_instructions}\n"
        )
    )
    
    chain = prompt | llm | parser
    
    try:
        result = await chain.ainvoke({"ticker": ticker, "headlines": headlines_str})
        score = result.score
        reasoning = result.reasoning
    except Exception as e:
        score = 0.01
        reasoning = f"Failed to parse sentiment. Error: {e}"
        
    # Strictly prevent flat 0.0
    if score == 0.0:
        score = 0.01
        
    state["sentiment_score"] = score
    state["sentiment_reasoning"] = reasoning
    
    return state
