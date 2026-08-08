import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from agents.state import AgentState

from agents.rag import retrieve_pdf_context

def analyze_quant_and_sentiment(state: AgentState) -> AgentState:
    """Combines quantitative data, short-term sentiment, and long-term PDF fundamentals."""
    ticker = state["ticker"]
    quant_data = state.get("quant_data", {})
    sentiment_score = state.get("sentiment_score", 0.0)
    sentiment_reasoning = state.get("sentiment_reasoning", "")
    
    # Retrieve deep PDF insights
    pdf_context = retrieve_pdf_context(ticker)
    
    # Check if quant_data has error
    if "error" in quant_data:
        state["signal"] = "ERROR"
        state["reasoning"] = f"Error fetching quant data: {quant_data['error']}"
        return state

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.0
    )
    
    prompt = PromptTemplate(
        input_variables=["ticker", "quant_data", "sentiment_score", "sentiment_reasoning", "pdf_context"],
        template=(
            "You are an expert quantitative trader and portfolio manager. "
            "Analyze the following short-term and long-term data for the stock ticker {ticker}.\n\n"
            "=== SHORT-TERM DATA ===\n"
            "Quantitative Data:\n{quant_data}\n\n"
            "News Sentiment Score (from -1.0 to 1.0):\n{sentiment_score}\n"
            "News Sentiment Reasoning:\n{sentiment_reasoning}\n\n"
            "=== LONG-TERM FUNDAMENTALS ===\n"
            "Deep insights retrieved from the company's Annual Report/PDF:\n{pdf_context}\n\n"
            "Based on this combined short-term and long-term information, output a deterministic trading signal (BUY, HOLD, or SELL) and a brief reasoning.\n"
            "You MUST output your response as a valid JSON object with the following schema exactly:\n"
            '{{"signal": "BUY", "reasoning": "Your reasoning here."}}'
        )
    )
    
    chain = prompt | llm
    
    # We serialize quant_data for the prompt
    quant_str = json.dumps(quant_data, indent=2)
    response = chain.invoke({
        "ticker": ticker, 
        "quant_data": quant_str,
        "sentiment_score": sentiment_score,
        "sentiment_reasoning": sentiment_reasoning,
        "pdf_context": pdf_context
    })
    
    output = response.content.strip()
    
    # Try parsing JSON (cleaning up possible markdown fences)
    if output.startswith("```json"):
        output = output[7:-3].strip()
    elif output.startswith("```"):
        output = output[3:-3].strip()
        
    try:
        result_json = json.loads(output)
        signal = result_json.get("signal", "HOLD").upper()
        reasoning = result_json.get("reasoning", "No reasoning provided.")
    except json.JSONDecodeError:
        signal = "HOLD"
        reasoning = f"Failed to parse JSON response. Output was: {output}"
        
    state["signal"] = signal
    state["reasoning"] = reasoning
    
    return state
