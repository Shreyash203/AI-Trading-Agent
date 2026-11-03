import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock
import sys
import pytest
from qdrant_client import QdrantClient
from agents import tools, rag, sentiment_agent, master_agent, workflow
from agents.reliability import calculate_signal, validate_citations


def run(coro):
    return asyncio.run(coro)


def quant(**updates):
    return dict(current_price=100., rsi_14=35., **{'50_day_ma':100.}, macd=2., macd_signal=1., previous_macd=0., previous_macd_signal=1., bb_lower=90.) | updates


@pytest.mark.parametrize('sentiment', [None, float('nan'), 2.])
def test_unavailable_sentiment_never_generates_signal(sentiment):
    assert calculate_signal(quant(), sentiment) is None


def test_neutral_is_valid_and_crossover_is_real():
    assert calculate_signal(quant(), 0.) == 'BUY'
    assert calculate_signal(quant(previous_macd=2.), 0.) == 'HOLD'
    assert calculate_signal(quant(rsi_14=70.), 0.) == 'SELL'
    assert calculate_signal(quant(rsi_14=float('nan')), 0.) is None


def test_schema_rejects_out_of_range_sentiment():
    with pytest.raises(ValueError):
        sentiment_agent.SentimentOutput(score=1.2, reasoning='bad', evidence_ids=['a'])
    assert sentiment_agent.SentimentOutput(score=0, reasoning='neutral', evidence_ids=['a']).score == 0


def test_unknown_citations_rejected():
    with pytest.raises(ValueError):
        validate_citations(['invented'], [{'id':'real'}])
    with pytest.raises(ValueError):
        validate_citations([], [{'id':'real'}])


def test_no_random_market_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, 'yfinance', SimpleNamespace(Ticker=lambda _: SimpleNamespace(history=lambda **_: SimpleNamespace(empty=True))))
    result = run(tools.fetch_quant_data('UNKNOWN'))
    assert 'error' in result
    assert 'current_price' not in result


def test_no_fabricated_sentiment_or_web_fallback(monkeypatch):
    monkeypatch.setattr(sentiment_agent, 'retrieve_evidence', lambda *a: [])
    result = run(sentiment_agent.analyze_sentiment({'ticker':'AAPL'}))
    assert result['sentiment_score'] is None
    assert result['sentiment_status'] == 'unavailable'


def test_market_failure_stops_workflow(monkeypatch):
    monkeypatch.setattr(workflow, 'fetch_quant_data', AsyncMock(return_value={'error':'offline'}))
    monkeypatch.setattr(workflow, 'fetch_news_headlines', AsyncMock(return_value=[]))
    result = run(workflow.app.ainvoke({'ticker':'UNKNOWN'}))
    assert result['signal'] == 'ERROR'
    assert result['analysis_status'] == 'unavailable'


class Embeddings:
    def __init__(self): self.calls=0
    def embed_documents(self, texts):
        self.calls += 1
        return [[1.,0.,0.] for _ in texts]
    def embed_query(self, query): return [1.,0.,0.]


def test_news_deduplication_and_date_filtering(monkeypatch):
    from datetime import datetime, timezone
    db, embed = QdrantClient(':memory:'), Embeddings()
    monkeypatch.setattr(rag, 'client', db)
    monkeypatch.setattr(rag, 'embeddings', embed)
    items=[{'title':'recent', 'url':'https://example.com/recent','published_at':'2026-11-02T00:00:00Z'},
           {'title':'old','url':'https://example.com/old','published_at':'2020-01-01T00:00:00Z'},
           {'title':'future','url':'https://example.com/future','published_at':'2030-01-01T00:00:00Z'}]
    rag.store_news_in_qdrant('AAPL', items)
    rag.store_news_in_qdrant('AAPL', items)
    assert embed.calls == 1
    assert db.get_collection(rag.collection('news','AAPL')).points_count == 3
    evidence=rag.retrieve_evidence('AAPL','news','news',as_of=datetime(2026,11,3,tzinfo=timezone.utc))
    assert len(evidence)==1
    assert evidence[0]['url']=='https://example.com/recent'
    db.close()


def test_embedding_failure_does_not_create_wrong_dimensions(monkeypatch):
    db = QdrantClient(':memory:')
    class Broken(Embeddings):
        def embed_documents(self, texts): raise RuntimeError('provider down')
    monkeypatch.setattr(rag, 'client', db)
    monkeypatch.setattr(rag, 'embeddings', Broken())
    with pytest.raises(RuntimeError):
        rag.store_news_in_qdrant('AAPL',[{'title':'a','url':'https://example.com','published_at':'2026-11-02T00:00:00Z'}])
    assert not db.collection_exists(rag.collection('news','AAPL'))
    db.close()


def test_api_singleflight_and_failure_not_cached(monkeypatch):
    import main
    main.local_cache.clear(); main._flights.clear()
    calls=[]
    async def resolve(*args,**kwargs): return 'AAPL'
    async def invoke(state):
        calls.append(state)
        await asyncio.sleep(.01)
        return {'ticker':'AAPL','signal':'ERROR','analysis_status':'unavailable','reasoning':'missing','sentiment_score':None}
    monkeypatch.setattr(main,'resolve_ticker',resolve)
    monkeypatch.setattr(main.workflow_app,'ainvoke',invoke)
    async def check():
        results=await asyncio.gather(main.analyze_ticker('Apple',main.Region.US),main.analyze_ticker('Apple',main.Region.US))
        assert all(r.sentiment_score is None for r in results)
        assert len(calls)==1
        assert not main.local_cache
    run(check())


def test_api_errors_are_sanitized(monkeypatch):
    import main
    from fastapi import HTTPException
    async def broken(*args,**kwargs): raise RuntimeError('secret-token-do-not-expose')
    monkeypatch.setattr(main,'resolve_ticker',broken)
    with pytest.raises(HTTPException) as exc:
        run(main.analyze_ticker('Failure',main.Region.US))
    assert exc.value.status_code==503
    assert 'secret-token' not in exc.value.detail

def test_evaluation_measures_coverage_without_claiming_truth():
    from evals.evaluate import evaluate
    cases=[{'case_id':'a','expected_source_urls':['https://example.com/a']}, {'case_id':'b','must_abstain':True}]
    outputs=[{'case_id':'a','evidence':[{'id':'one','url':'https://example.com/a'}],'cited_evidence_ids':['one']}, {'case_id':'b','signal':'ERROR','analysis_status':'unavailable'}]
    report=evaluate(cases,outputs)
    assert report['source_recall']==1
    assert report['citation_id_validity']==1
    assert report['required_abstention_rate']==1
    with pytest.raises(ValueError):
        evaluate(cases,outputs[:1])

def test_neutral_model_output_remains_neutral(monkeypatch):
    import json
    from langchain_core.runnables import RunnableLambda
    evidence=[{'id':'news-1','url':'https://example.com/news','text':'unchanged outlook'}]
    monkeypatch.setattr(sentiment_agent,'retrieve_evidence',lambda *a:evidence)
    monkeypatch.setattr(sentiment_agent,'ChatGroq',lambda **kw:RunnableLambda(lambda prompt:json.dumps({'score':0,'reasoning':'neutral','evidence_ids':['news-1']})))
    result=run(sentiment_agent.analyze_sentiment({'ticker':'AAPL'}))
    assert result['sentiment_score']==0
    assert result['sentiment_status']=='available'


def test_invented_model_citation_becomes_unavailable(monkeypatch):
    import json
    from langchain_core.runnables import RunnableLambda
    monkeypatch.setattr(sentiment_agent,'retrieve_evidence',lambda *a:[{'id':'real','url':'https://example.com','text':'news'}])
    monkeypatch.setattr(sentiment_agent,'ChatGroq',lambda **kw:RunnableLambda(lambda prompt:json.dumps({'score':.9,'reasoning':'claim','evidence_ids':['invented']})))
    result=run(sentiment_agent.analyze_sentiment({'ticker':'AAPL'}))
    assert result['sentiment_score'] is None


def test_explanation_cannot_replace_rule_signal(monkeypatch):
    import json
    from langchain_core.runnables import RunnableLambda
    news=[{'id':'news','url':'https://example.com','text':'news'}]
    monkeypatch.setattr(master_agent,'retrieve_evidence',lambda *a:[])
    monkeypatch.setattr(master_agent,'ChatGroq',lambda **kw:RunnableLambda(lambda prompt:json.dumps({'reasoning':'Evidence suggests caution','evidence_ids':['news'],'signal':'SELL'})))
    result=run(master_agent.analyze_quant_and_sentiment({'ticker':'AAPL','quant_data':quant(),'sentiment_score':0.,'news_evidence':news}))
    assert result['signal']=='BUY'
    assert result['analysis_status']=='partial'
    assert result['cited_evidence_ids']==['news']


def test_api_success_cache_and_input_validation(monkeypatch):
    import main
    from fastapi import HTTPException
    main.local_cache.clear();main._flights.clear()
    async def resolve(*args,**kwargs): return 'AAPL'
    invoke=AsyncMock(return_value={'ticker':'AAPL','signal':'HOLD','sentiment_score':0.,'analysis_status':'available','reasoning':'neutral'})
    monkeypatch.setattr(main,'resolve_ticker',resolve)
    monkeypatch.setattr(main.workflow_app,'ainvoke',invoke)
    async def check():
        await main.analyze_ticker('Apple',main.Region.US)
        await main.analyze_ticker(' Apple ',main.Region.US)
        assert invoke.await_count==1
        with pytest.raises(HTTPException):
            await main.analyze_ticker('../secret',main.Region.US)
    run(check())

def test_invalid_market_values_do_not_generate_signal():
    assert calculate_signal(quant(current_price=-1),0.) is None
    assert calculate_signal(quant(rsi_14=150),0.) is None
