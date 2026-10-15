from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

@lru_cache()
def clients():
    api_key = settings.effective_gemini_api_key
    if not api_key or not settings.pinecone_api_key:
        raise ValueError("GEMINI_API_KEY and PINECONE_API_KEY must be set in your .env file.")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embed_model,
        google_api_key=api_key,
        output_dimensionality=768
    )
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return embeddings, pc


def ensure_index():
    _, pc = clients()
    index_name = settings.pinecone_index
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(index_name)


def embed(texts):
    embeddings, _ = clients()
    return embeddings.embed_documents(texts)



def search_kb(query, top_k=5):
    index = ensure_index()
    embeddings, _ = clients()
    vector = embeddings.embed_query(query)
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{'page': m.metadata.get('page'), 'text': m.metadata.get('text',''), 'score': round(m.score,3), 'source': 'Gale Encyclopedia (2002)'} for m in result.matches if m.score >= 0.30]