# src/retrieval/query_processor.py

from typing import List, Dict, Optional, Tuple
from src.embeddings.embedder import OllamaEmbedder
from src.retrieval.vector_store import VectorStore

def retrieve_context(
    question: str,
    embedder: OllamaEmbedder,
    vector_store: VectorStore,
    source_filter: Optional[List[str]] = None,
    top_k: int = 5
) -> Tuple[str, List[Dict]]:
    """
    Retrieve context from the vector store for a given question.

    Args:
        question: The user's question.
        embedder: OllamaEmbedder instance to generate query embedding.
        vector_store: VectorStore instance to perform similarity search.
        source_filter: Optional list of source filenames to restrict search.
        top_k: Number of top chunks to retrieve.

    Returns:
        A tuple (context_string, metadata_list) where:
            - context_string: Concatenated text of retrieved chunks, each separated by two newlines.
            - metadata_list: List of dicts with "source" and "page" keys for each retrieved chunk.
        If no results are found or an error occurs, returns ("", []).
    """
    # 1. Embed the question
    query_embedding = embedder.embed(question)
    if not query_embedding:
        return "", []

    # 2. Search the vector store
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        source_filter=source_filter
    )

    if not results:
        return "", []

    # 3. Build context string and metadata list
    context_parts = []
    metadata_list = []
    for doc, meta, score in results:
        context_parts.append(doc)
        # Ensure metadata contains source and page; use fallbacks if missing
        source = meta.get("source", "unknown")
        page = meta.get("page", 0)
        metadata_list.append({"source": source, "page": page})

    context_string = "\n\n".join(context_parts)
    return context_string, metadata_list