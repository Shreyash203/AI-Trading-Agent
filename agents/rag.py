import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize Qdrant Client (File-based local storage, no Docker required!)
client = QdrantClient(path="./qdrant_db")

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTOR_SIZE = 384  # Size for all-MiniLM-L6-v2

def init_collection(collection_name: str):
    """Ensure the collection exists in Qdrant."""
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        # Collection doesn't exist, create it
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

def store_news_in_qdrant(ticker: str, news_items: List[str]):
    """Embeds news items and stores them in Qdrant under a ticker-specific collection."""
    if not news_items:
        return
        
    collection_name = f"news_{ticker.lower()}"
    init_collection(collection_name)
    
    points = []
    for text in news_items:
        vector = embeddings.embed_query(text)
        
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text}
            )
        )
        
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )

def retrieve_relevant_news(ticker: str, query: str = "bullish bearish financial outlook performance", limit: int = 5) -> str:
    """Retrieves the most relevant news context from Qdrant for a given ticker."""
    collection_name = f"news_{ticker.lower()}"
    
    try:
        # Check if collection exists first
        client.get_collection(collection_name=collection_name)
    except Exception:
        return "No historical news context found in database."
        
    query_vector = embeddings.embed_query(query)
    
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit
    )
    
    if not search_result:
        return "No relevant news found."
        
    context = []
    for hit in search_result:
        text = hit.payload.get("text", "")
        context.append(f"- {text}")
        
    return "\n".join(context)
