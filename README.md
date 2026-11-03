# Financial Research Agent

FastAPI + LangGraph research screen using market indicators, source-backed news
sentiment and annual filing excerpts. The LLM explains explicit Python screening
rules; it does not guarantee returns or deterministic model reasoning.

## Run

Install requirements.txt. Configure GROQ_API_KEY, GOOGLE_API_KEY, and optionally
LLM_MODEL / EMBEDDING_MODEL. Set SEC_USER_AGENT to your application name and a
real contact email to enable SEC filing lookup. No secrets belong in Git.

Run `uvicorn main:app --host 0.0.0.0 --port 8000`.
POST /analyze?company_name=Apple&region=US keeps the original query interface.
GET /healthz is a lightweight process health check, not provider readiness.

Set ANALYSIS_API_KEY when exposing the service; callers then supply X-API-Key.
The application applies a per-process 10-request/minute/client limit, caps active
analyses at four, coalesces identical in-flight requests and caches complete results
for five minutes with a maximum of 256 entries. For multiple replicas, enforce
shared limits at the gateway and use Qdrant server/cloud through QDRANT_URL and
QDRANT_API_KEY. File-based Qdrant and in-memory limits/cache are single-process.
Cancellations cannot forcibly stop already-running synchronous library calls.

## Response changes

Existing stock_symbol, signal, sentiment_score and reasoning fields remain.
Missing sentiment is now null rather than a fabricated positive number. Clients
must handle this. ERROR + analysis_status=unavailable means insufficient evidence;
it is different from a supported HOLD. Partial results are not cached.

Added fields: analysis_status, sentiment_status, market_as_of, evidence,
cited_evidence_ids and warnings. Evidence includes source URLs, dates and chunk
IDs. PDF page numbers refer to generated text PDFs, not original SEC pagination.
Citations are validated against retrieved IDs, not automatically fact-checked.

News must have a URL and publication timestamp and be within seven days. Filing
retrieval uses a 550-day window. Neither window proves a source remains accurate.
SEC lookup refreshes daily; non-US regional symbols are not mapped to US issuers.
Regional annual-report providers are not yet implemented. No filing is reported
as partial coverage. Old collections are retained; new *_v2_* collections require
re-ingestion to gain date/source metadata. No guessed embedding dimension is used.

Signal rules: SELL above RSI 65 or below 95% of the 50-day mean; otherwise BUY
below RSI 40, with a real MACD crossover or lower-band touch and sentiment >= -0.5;
otherwise HOLD. Required missing/non-finite inputs yield no signal. These thresholds
are explicit research assumptions, not a validated investment strategy.

## Verification

Install pytest separately and run `python -m pytest tests -q`. Tests mock external
providers and exercise failures, date filters, deduplication, rules and citations.
See evals/README.md for evaluation of recorded complete workflow responses.
The legacy backtest does not evaluate this workflow. Before making performance
claims, run live provider smoke tests and a reviewed held-out evaluation dataset.

Request logs include a request ID, status and elapsed time. Per-model token/cost
accounting and distributed tracing are not implemented yet.
