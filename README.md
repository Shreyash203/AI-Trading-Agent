
# AI Financial Sentiment & Algo-Trading Engine

An asynchronous multi-agent financial analysis API built with **LangGraph** and **FastAPI**. This system orchestrates specialized AI agents to synthesize unstructured market news and quantitative stock data, outputting deterministic Buy/Hold/Sell signals.

## 🚀 Features
- **Data Ingestion:** Integrates with `yfinance` to fetch real-time price action, volume, and 52-week highs/lows.
- **News Scraping:** Automatically aggregates the top daily headlines for the requested ticker.
- **Agentic AI Workflow:** Utilizes LangGraph to pass the stock's "State" between a Sentiment Analysis LLM and a Master Quantitative LLM.
- **Structured Outputs:** Enforces strict JSON responses for easy integration into trading dashboards.

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI
- **AI/Orchestration:** LangGraph, LangChain, Groq/Gemini LLMs
- **Data Feeds:** yfinance API

*(Note: Currently in active development. Code will be pushed shortly.)*
