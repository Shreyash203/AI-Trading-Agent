"""Source-aware, idempotent retrieval. No network calls at import time."""
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, DatetimeRange

client = None
embeddings = None
_lock = threading.RLock()


def _clients():
    global client, embeddings
    with _lock:
        if embeddings is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv('EMBEDDING_MODEL', 'gemini-embedding-2-preview'), google_api_key=os.getenv('GOOGLE_API_KEY'))
        if client is None:
            url = os.getenv('QDRANT_URL')
            client = QdrantClient(url=url, api_key=os.getenv('QDRANT_API_KEY')) if url else QdrantClient(path=os.getenv('QDRANT_PATH', './qdrant_db'))
    return client, embeddings


def collection(kind, ticker):
    model = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-2-preview')
    fingerprint = hashlib.sha256(model.encode()).hexdigest()[:8]
    return f'{kind}_v2_{fingerprint}_{ticker.lower()}'


def _ensure(name, size):
    db, _ = _clients()
    with _lock:
        if not db.collection_exists(name):
            db.create_collection(name, vectors_config=VectorParams(size=size, distance=Distance.COSINE))
        elif db.get_collection(name).config.params.vectors.size != size:
            raise ValueError('Embedding dimension mismatch; reindex with the configured model.')


def _id(key):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def store_news_in_qdrant(ticker, news_items):
    items = [x for x in news_items if isinstance(x, dict) and x.get('title') and x.get('url') and x.get('published_at')]
    if not items:
        return
    db, embed = _clients()
    name = collection('news', ticker)
    ids = [_id(f"{ticker}:news:{x['url']}:{x['title']}") for x in items]
    existing = {str(x.id) for x in db.retrieve(name, ids=ids)} if db.collection_exists(name) else set()
    pairs = [(item, ident) for item, ident in zip(items, ids) if ident not in existing]
    if not pairs:
        return
    vectors = embed.embed_documents([x['title'] for x, _ in pairs])
    if len(vectors) != len(pairs):
        raise ValueError('Incomplete embedding response')
    _ensure(name, len(vectors[0]))
    db.upsert(name, points=[PointStruct(id=ident, vector=v, payload={'text': x['title'], 'url': x['url'], 'published_at': x['published_at'], 'kind': 'news'}) for (x, ident), v in zip(pairs, vectors)], wait=True)


def store_pdf_in_qdrant(ticker, file_path):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    path = Path(file_path)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if not metadata.get('url') or not metadata.get('filing_date'):
        raise ValueError('Document source and filing date are required')
    digest = metadata.get('content_hash') or hashlib.sha256(path.read_bytes()).hexdigest()
    db, embed = _clients()
    name = collection('pdf', ticker)
    if db.collection_exists(name):
        matches, _ = db.scroll(name, scroll_filter=Filter(must=[FieldCondition(key='content_hash', match=MatchValue(value=digest))]), limit=1)
        if matches:
            return
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(PyPDFLoader(str(path)).load())
    if not chunks:
        raise ValueError('Empty document')
    vectors = embed.embed_documents([x.page_content for x in chunks])
    if len(vectors) != len(chunks):
        raise ValueError('Incomplete embedding response')
    _ensure(name, len(vectors[0]))
    points = [PointStruct(id=_id(f'{ticker}:{digest}:{i}'), vector=v, payload={**metadata, 'text': chunk.page_content, 'content_hash': digest, 'chunk_id': i, 'page': chunk.metadata.get('page', 0)+1, 'page_kind': 'generated_text_pdf', 'kind': 'filing'}) for i, (chunk, v) in enumerate(zip(chunks, vectors))]
    db.upsert(name, points=points, wait=True)


def retrieve_evidence(ticker, kind, query, limit=5, as_of=None):
    db, embed = _clients()
    name = collection(kind, ticker)
    if not db.collection_exists(name):
        return []
    now = as_of or datetime.now(timezone.utc)
    key = 'published_at' if kind == 'news' else 'filing_date'
    age = timedelta(days=7 if kind == 'news' else 550)
    filters = Filter(must=[FieldCondition(key=key, range=DatetimeRange(gte=now-age, lte=now))])
    hits = db.query_points(name, query=embed.embed_query(query), query_filter=filters, limit=limit).points
    return [{'id': str(h.id), 'text': h.payload['text'], 'url': h.payload['url'], 'published_at': h.payload.get(key), 'page': h.payload.get('page'), 'page_kind': h.payload.get('page_kind'), 'chunk_id': h.payload.get('chunk_id'), 'kind': h.payload.get('kind'), 'score': h.score} for h in hits if h.payload.get('url') and h.payload.get('text')]


def retrieve_relevant_news(ticker, query='financial outlook', limit=5):
    return json.dumps(retrieve_evidence(ticker, 'news', query, limit))


def retrieve_pdf_context(ticker, query='financial highlights revenue risks', limit=4):
    return json.dumps(retrieve_evidence(ticker, 'pdf', query, limit))
