# src/retrieval/query_processor.py

from typing import List, Dict, Optional, Tuple
from src.embeddings.embedder import OllamaEmbedder
from src.retrieval.vector_store import VectorStore


def retrieve_context(
    question: str,
    embedder: OllamaEmbedder,
    vector_store: VectorStore,
    source_filter: Optional[List[str]] = None,
    top_k: int = 5,
    max_distance: Optional[float] = 1.0,
) -> Tuple[str, List[Dict]]:
    """
    Retrieve context from the vector store for a given question.

    Chunks whose ChromaDB distance exceeds ``max_distance`` are discarded
    before being passed to the LLM, preventing irrelevant noise from
    degrading answer quality.

    Args:
        question:      The user's question.
        embedder:      OllamaEmbedder instance to generate the query embedding.
        vector_store:  VectorStore instance to perform similarity search.
        source_filter: Optional list of source filenames to restrict the search.
        top_k:         Maximum number of chunks to retrieve before filtering.
        max_distance:  Upper bound on ChromaDB distance (lower = more similar).
                       Chunks with distance > max_distance are dropped.
                       Set to None to disable filtering and accept all top_k
                       results regardless of quality.
                       A sensible starting value for cosine distance is 1.0;
                       tighten to ~0.5–0.7 once you know your embedding space.

    Returns:
        A tuple ``(context_string, metadata_list)`` where:
          - context_string: Concatenated text of retained chunks, separated
                            by two newlines. Empty string if nothing passes
                            the distance filter.
          - metadata_list:  List of dicts with "source", "page", and
                            "distance" keys for each retained chunk.
        Returns ``("", [])`` if the embedding fails or no chunks survive
        the distance threshold.
    """
    # 1. Embed the question
    query_embedding = embedder.embed(question)
    if not query_embedding:
        return "", []

    # 2. Search the vector store (distance filtering happens inside VectorStore)
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        source_filter=source_filter,
        max_distance=max_distance,
    )

    if not results:
        return "", []

    # 3. Build context string and metadata list
    context_parts: List[str] = []
    metadata_list: List[Dict] = []

    for doc, meta, distance in results:
        context_parts.append(doc)
        metadata_list.append({
            "source":   meta.get("source", "unknown"),
            "page":     meta.get("page", 0),
            "distance": round(distance, 4),
        })

    context_string = "\n\n".join(context_parts)
    return context_string, metadata_list