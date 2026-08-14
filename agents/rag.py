import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize Qdrant Client (File-based local storage, no Docker required!)
client = QdrantClient(path="./qdrant_db")

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Dynamically determine vector size to make the architecture model-agnostic
try:
    _dummy_vector = embeddings.embed_query("test")
    VECTOR_SIZE = len(_dummy_vector)
except Exception:
    VECTOR_SIZE = 384  # Fallback

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
    
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    ).points
    
    if not search_result:
        return "No relevant news found."
        
    context = []
    for hit in search_result:
        text = hit.payload.get("text", "")
        context.append(f"- {text}")
        
    return "\n".join(context)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def has_pdf_for_ticker(ticker: str) -> bool:
    """Checks if the PDF collection already exists and has data."""
    collection_name = f"pdf_{ticker.lower()}"
    try:
        info = client.get_collection(collection_name=collection_name)
        return info.points_count > 0
    except Exception:
        return False

def store_pdf_in_qdrant(ticker: str, file_path: str):
    """Loads a PDF, chunks it, and batch embeds it into Qdrant."""
    if not file_path:
        return
        
    print(f"Autonomous Agent: Chunking and Embedding {file_path}...")
    collection_name = f"pdf_{ticker.lower()}"
    init_collection(collection_name)
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    texts = [split.page_content for split in splits]
    
    # Batch embedding is significantly faster!
    vector_list = embeddings.embed_documents(texts)
    
    points = []
    for i, (text, vector) in enumerate(zip(texts, vector_list)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, "source": file_path, "chunk_id": i}
            )
        )
        
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
    print(f"Autonomous Agent: Successfully saved {len(points)} PDF chunks to memory!")

def retrieve_pdf_context(ticker: str, query: str = "financial highlights revenue risks", limit: int = 4) -> str:
    """Retrieves deep fundamental insights from the PDF database."""
    collection_name = f"pdf_{ticker.lower()}"
    
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        return "No deep fundamental PDF data found for this stock."
        
    query_vector = embeddings.embed_query(query)
    
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    ).points
    
    if not search_result:
        return "No relevant insights found in PDF."
        
    context = []
    for hit in search_result:
        text = hit.payload.get("text", "")
        context.append(f"- {text}")
        
    return "\n\n".join(context)
