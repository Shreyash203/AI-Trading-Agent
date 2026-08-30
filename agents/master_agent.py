import os
import json
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from agents.state import AgentState

from agents.rag import retrieve_pdf_context

import asyncio

class MasterOutput(BaseModel):
    signal: str = Field(description="Deterministic trading signal: BUY, HOLD, or SELL.")
    reasoning: str = Field(description="A brief reasoning for the signal based on quant, sentiment, and pdf data.")

async def analyze_quant_and_sentiment(state: AgentState) -> AgentState:
    """Combines quantitative data, short-term sentiment, and long-term PDF fundamentals."""
    ticker = state["ticker"]
    quant_data = state.get("quant_data", {})
    sentiment_score = state.get("sentiment_score", 0.0)
    sentiment_reasoning = state.get("sentiment_reasoning", "")
    
    # Retrieve deep PDF insights
    pdf_context = await asyncio.to_thread(retrieve_pdf_context, ticker)
    
    # Check if quant_data has error
    if "error" in quant_data:
        state["signal"] = "ERROR"
        state["reasoning"] = f"Error fetching quant data: {quant_data['error']}"
        return state

    llm = ChatGroq(
        model="openai/gpt-oss-20b",  # Qwen supports massive context windows for RAG
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.0
    )
    
    parser = PydanticOutputParser(pydantic_object=MasterOutput)
    
    prompt = PromptTemplate(
        input_variables=["ticker", "quant_data", "sentiment_score", "sentiment_reasoning", "pdf_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template=(
            "You are an expert quantitative trader and portfolio manager. "
            "Analyze the following short-term and long-term data for the stock ticker {ticker}.\n\n"
            "=== SHORT-TERM DATA ===\n"
            "Quantitative Data:\n{quant_data}\n\n"
            "News Sentiment Score (from -1.0 to 1.0):\n{sentiment_score}\n"
            "News Sentiment Reasoning:\n{sentiment_reasoning}\n\n"
            "=== LONG-TERM FUNDAMENTALS ===\n"
            "Deep insights retrieved from the company's Annual Report/PDF:\n{pdf_context}\n\n"
            "Based on this combined short-term and long-term information, output a deterministic trading signal.\n"
            "{format_instructions}\n"
        )
    )
    
    chain = prompt | llm | parser
    
    # We serialize quant_data for the prompt
    quant_str = json.dumps(quant_data, indent=2)
    
    try:
        result = await chain.ainvoke({
            "ticker": ticker, 
            "quant_data": quant_str,
            "sentiment_score": sentiment_score,
            "sentiment_reasoning": sentiment_reasoning,
            "pdf_context": pdf_context
        })
        signal = result.signal.upper()
        if signal not in ["BUY", "HOLD", "SELL"]:
            signal = "HOLD"
        reasoning = result.reasoning
    except Exception as e:
        signal = "HOLD"
        reasoning = f"Failed to parse JSON response. Error: {e}"
        
    state["signal"] = signal
    state["reasoning"] = reasoning
    
    return state
