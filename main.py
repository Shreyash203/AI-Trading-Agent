from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import os
import time
import uuid
from collections import OrderedDict, deque
from enum import Enum
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from agents.workflow import app as workflow_app
from agents.tools import resolve_ticker

logger = logging.getLogger(__name__)
app = FastAPI(title='AI Financial Research Engine', version='1.1.0')


class Region(str, Enum):
    INDIA = 'India'
    US = 'US'
    UK = 'UK'
    CANADA = 'Canada'
    AUSTRALIA = 'Australia'
    GERMANY = 'Germany'
    FRANCE = 'France'
    JAPAN = 'Japan'
    CHINA = 'China'
    BRAZIL = 'Brazil'


class Evidence(BaseModel):
    id: str
    text: str
    url: str
    published_at: str | None = None
    page: int | None = None
    page_kind: str | None = None
    chunk_id: int | None = None
    kind: str | None = None
    score: float | None = None


class AnalysisResponse(BaseModel):
    stock_symbol: str
    signal: Literal['BUY', 'HOLD', 'SELL', 'ERROR']
    sentiment_score: float | None = Field(default=None, ge=-1, le=1)
    reasoning: str
    analysis_status: Literal['available', 'partial', 'unavailable'] = 'unavailable'
    sentiment_status: str = 'unavailable'
    market_as_of: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


local_cache = OrderedDict()
CACHE_TTL = 300
CACHE_MAX = 256
MAX_INFLIGHT = 4
REQUEST_TIMEOUT = 120
_flights = {}
_rate = OrderedDict()


@app.middleware('http')
async def request_metrics(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.monotonic()
    # Shared key is optional locally, recommended when exposing a paid-model API.
    key = os.getenv('ANALYSIS_API_KEY')
    if request.url.path == '/analyze':
        import secrets
        if key and not secrets.compare_digest(request.headers.get('X-API-Key', ''), key):
            return JSONResponse({'detail': 'Invalid API key'}, status_code=401)
        identity = request.client.host if request.client else 'unknown'
        now = time.monotonic()
        bucket = _rate.setdefault(identity, deque())
        _rate.move_to_end(identity)
        while bucket and now-bucket[0] >= 60:
            bucket.popleft()
        if len(bucket) >= 10:
            return JSONResponse({'detail': 'Too many requests'}, status_code=429, headers={'Retry-After': '60'})
        bucket.append(now)
        while len(_rate) > 2048:
            _rate.popitem(last=False)
    response = await call_next(request)
    response.headers['X-Request-ID'] = request_id
    logger.info('request id=%s path=%s status=%s duration_ms=%.1f', request_id, request.url.path, response.status_code, (time.monotonic()-start)*1000)
    return response


async def _analyze(company_name, region):
    ticker = (await resolve_ticker(company_name, region=region.value)).upper()
    result = await workflow_app.ainvoke({'ticker': ticker})
    return AnalysisResponse(stock_symbol=result.get('ticker', ticker), signal=result.get('signal', 'ERROR'),
        sentiment_score=result.get('sentiment_score'), reasoning=result.get('reasoning', 'Analysis unavailable.'),
        analysis_status=result.get('analysis_status', 'unavailable'), sentiment_status=result.get('sentiment_status', 'unavailable'),
        market_as_of=result.get('quant_data', {}).get('as_of'), evidence=result.get('evidence', result.get('news_evidence', [])),
        cited_evidence_ids=result.get('cited_evidence_ids', []), warnings=result.get('warnings', []))


@app.post('/analyze', response_model=AnalysisResponse)
async def analyze_ticker(company_name: str, region: Region = Region.INDIA):
    company_name = company_name.strip()
    if not 1 <= len(company_name) <= 100 or any(c in company_name for c in '/\\\x00'):
        raise HTTPException(422, 'Invalid company name or ticker')
    key = (company_name.casefold(), region.value)
    now = time.monotonic()
    if key in local_cache:
        data, stamp = local_cache[key]
        if now-stamp < CACHE_TTL:
            local_cache.move_to_end(key)
            return AnalysisResponse(**data)
        del local_cache[key]
    task = _flights.get(key)
    if task is None:
        if len(_flights) >= MAX_INFLIGHT:
            raise HTTPException(503, 'Analysis capacity reached. Retry shortly.', headers={'Retry-After': '5'})
        async def run():
            try:
                response = await asyncio.wait_for(_analyze(company_name, region), REQUEST_TIMEOUT)
                if response.analysis_status == 'available':
                    local_cache[key] = (response.model_dump(), time.monotonic())
                    while len(local_cache) > CACHE_MAX:
                        local_cache.popitem(last=False)
                return response
            finally:
                _flights.pop(key, None)
        task = asyncio.create_task(run())
        _flights[key] = task
        # Retrieve exceptions even if all clients disconnect during shared work.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    try:
        return await asyncio.shield(task)
    except asyncio.TimeoutError:
        raise HTTPException(504, 'Analysis timed out. Retry later.')
    except Exception:
        logger.exception('Analysis failed')
        raise HTTPException(503, 'Analysis temporarily unavailable.')


@app.get('/healthz')
def healthz():
    return {'status': 'ok'}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
