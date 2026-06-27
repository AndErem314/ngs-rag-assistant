# src/retrieval/query_processor.py

import re
from typing import List, Dict, Optional, Tuple
from src.embeddings.embedder import OllamaEmbedder
from src.retrieval.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Protocol name → source filename mapping for auto source-filtering.
# Questions that mention a protocol name get an automatic source filter,
# preventing vector search from pulling chunks from the wrong PDFs.
# ---------------------------------------------------------------------------
PROTOCOL_SOURCE_MAP = {
    "nextera xt":        "Nextera-XT-DNA.pdf",
    "nextera":           "Nextera-XT-DNA.pdf",
    "truseq nano":       "TruSeq-Nano-DNA.pdf",
    "truseq dna pcr":    "TruSeq-DNA-PCR-Free.pdf",
    "truseq stranded":   "TruSeq-Stranded-Total-RNA.pdf",
    "trusight oncology": "TruSight-Oncology-500-v2.pdf",
    "trusight":          "TruSight-Oncology-500-v2.pdf",
}

# Nextera-specific terminology — these terms appear almost exclusively in
# the Nextera XT DNA protocol and serve as strong signals for source detection.
NEXTERA_KEYWORDS = [
    "tagmentation",
    "tagment buffer",
    "neutralize tagment",
    "amplicon tagment",
]


def detect_source_filter(question: str) -> Optional[List[str]]:
    """Detect protocol names in the question and return matching source filenames.

    Returns ``None`` when no protocol name is detected (meaning: search all sources).
    When multiple protocols match, all matching sources are returned.

    Detection uses two tiers:
    1. Protocol name mentions (e.g., "Nextera XT", "TruSeq Nano")
    2. Protocol-specific terminology (e.g., "tagmentation" → Nextera XT)
    """
    question_lower = question.lower()
    matched_sources: List[str] = []

    # Tier 1: protocol name mentions
    for pattern, source in PROTOCOL_SOURCE_MAP.items():
        if pattern in question_lower and source not in matched_sources:
            matched_sources.append(source)

    # Tier 2: Nextera-specific terminology
    if not matched_sources:
        for keyword in NEXTERA_KEYWORDS:
            if keyword in question_lower:
                nextera_source = "Nextera-XT-DNA.pdf"
                if nextera_source not in matched_sources:
                    matched_sources.append(nextera_source)
                break  # one keyword is enough

    return matched_sources if matched_sources else None


def retrieve_context(
    question: str,
    embedder: OllamaEmbedder,
    vector_store: VectorStore,
    source_filter: Optional[List[str]] = None,
    top_k: int = 15,
    max_distance: Optional[float] = 0.35,
    hybrid: bool = True,
) -> Tuple[str, List[Dict]]:
    """
    Retrieve context from the vector store for a given question.

    Chunks whose ChromaDB distance exceeds ``max_distance`` are discarded
    before being passed to the LLM, preventing irrelevant noise from
    degrading answer quality.

    When ``source_filter`` is not explicitly provided, the function
    auto-detects protocol names in the question text and applies the
    corresponding source filter.  This prevents vector search from
    returning chunks from unrelated protocol PDFs when the question
    is protocol-specific.

    Args:
        question:      The user's question.
        embedder:      OllamaEmbedder instance to generate the query embedding.
        vector_store:  VectorStore (ChromaDB, primary) or PgvectorStore
                       (pgvector, secondary) instance to perform similarity search.
        source_filter: Optional list of source filenames to restrict the search.
                       When ``None``, auto-detected from the question text.
                       Pass an explicit empty list ``[]`` to disable both
                       auto-detection and filtering.
        top_k:         Maximum number of chunks to retrieve before filtering.
        max_distance:  Upper bound on ChromaDB distance (lower = more similar).
                       Chunks with distance > max_distance are dropped.
                       Set to None to disable filtering and accept all top_k
                       results regardless of quality.
                       Defaults to 0.35 — tuned for the embedding space of
                       haybu/mxbai-embed-large-latest on NGS protocol PDFs.
        hybrid:        If True, combine vector search with BM25 keyword search
                       for better recall on protocol-specific terms (volumes,
                       temperatures, named reagents).

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

    # 2. Auto-detect source filter if none explicitly provided
    if source_filter is None:
        source_filter = detect_source_filter(question)
    elif len(source_filter) == 0:
        source_filter = None  # empty list → no filter

    # 3. Search the vector store (distance filtering happens inside VectorStore)
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        source_filter=source_filter,
        max_distance=max_distance,
        hybrid=hybrid,
        query_text=question if hybrid else None,
    )

    if not results:
        return "", []

    # 4. Build context string and metadata list
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