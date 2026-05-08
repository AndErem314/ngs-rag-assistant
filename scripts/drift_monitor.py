"""
NGS RAG Drift Monitor

Tracks embedding quality and retrieval accuracy drift over time using validation question sets.
Logs metrics to JSON for historical comparison.

Usage:
    # Run once manually (saves to drift_metrics.json)
    python scripts/drift_monitor.py

    # With custom settings
    python scripts/drift_monitor.py --questions validation/questions/custom.json --output weekly_drift.json

    # With hybrid search and table-aware chunking
    python scripts/drift_monitor.py --hybrid --strategy table_aware
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion import chunk_document_with_strategy, ChunkingStrategy
from src.ingestion.pdf_parser import extract_pages
from src.embeddings.embedder import OllamaEmbedder
from src.retrieval.vector_store import VectorStore
from src.observability import MetricsCollector

# Default reference texts for embedding drift checks (stable NGS terms)
DEFAULT_REFERENCE_TEXTS = [
    "DNA extraction from blood samples",
    "PCR amplification of target regions",
    "Illumina TruSight Oncology 500 protocol",
    "Hybridization capture for NGS libraries",
    "Sequencing run quality control metrics",
]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_previous_metrics(output_path: str) -> Dict[str, Any]:
    """Load previous drift metrics if file exists."""
    if not os.path.exists(output_path):
        return {}
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                return data[-1]  # Return most recent entry
            return {}
    except Exception:
        return {}


def compute_embedding_drift(
    embedder: OllamaEmbedder,
    reference_texts: List[str],
    previous_embeddings: Dict[str, List[float]] = None,
) -> Dict[str, Any]:
    """Compute embedding drift for reference texts."""
    current_embeddings = {}
    drift_results = {}

    for text in reference_texts:
        emb = embedder.embed(text)
        if not emb:
            continue
        current_embeddings[text] = emb

        if previous_embeddings and text in previous_embeddings:
            prev_emb = previous_embeddings[text]
            similarity = cosine_similarity(emb, prev_emb)
            drift_results[text] = {
                "similarity": similarity,
                "drifted": similarity < 0.95,  # Threshold for drift alert
            }

    return {
        "current_embeddings": current_embeddings,
        "drift_results": drift_results,
        "avg_similarity": (
            sum(r["similarity"] for r in drift_results.values()) / len(drift_results)
            if drift_results else 1.0
        ),
    }


def run_retrieval_test(
    pdf_path: str,
    questions_path: str,
    strategy: str = "basic",
    hybrid: bool = False,
    embedder_model: str = "haybu/mxbai-embed-large-latest:latest",
    ollama_host: str = "http://localhost:11434",
    metrics_collector: object = None,
) -> Dict[str, Any]:
    """Run retrieval accuracy test (reused logic from test_retrieval_accuracy.py)."""
    # Load questions
    with open(questions_path, 'r') as f:
        questions = json.load(f)

    # Extract pages
    pages = extract_pages(pdf_path)

    # Chunk document
    strategy_enum = ChunkingStrategy(strategy)
    if strategy == "table_aware":
        chunks = chunk_document_with_strategy(
            pages=pages,
            source_filename=os.path.basename(pdf_path),
            pdf_path=pdf_path,
            strategy=strategy_enum,
        )
    else:
        chunks = chunk_document_with_strategy(
            pages=pages,
            source_filename=os.path.basename(pdf_path),
            strategy=strategy_enum,
        )

    # Initialize embedder
    embedder = OllamaEmbedder(host=ollama_host, model=embedder_model)
    test_emb = embedder.embed("test")
    if not test_emb:
        raise RuntimeError("Embedder not responding")

    # Initialize vector store
    vector_store = VectorStore(
        collection_name=f"drift_test_{strategy}",
        persist_directory="./chroma_drift_temp",
    )
    vector_store.clear_collection()

    # Embed and store chunks
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_batch(chunk_texts)
    valid = [(c, e) for c, e in zip(chunks, embeddings) if e]
    if not valid:
        raise RuntimeError("No valid embeddings")
    valid_chunks, valid_embeddings = zip(*valid)
    vector_store.add_chunks(list(valid_chunks), list(valid_embeddings))

    # Run queries
    page_match_count = 0
    page_in_top_k_count = 0
    distance_sum = 0.0
    distance_count = 0

    for q in questions:
        question = q["question"]
        expected_page = q.get("source_page", None)

        query_emb = embedder.embed(question)
        if not query_emb:
            continue

        # Time the search
        start_time = time.time()
        hits = vector_store.search(
            query_embedding=query_emb,
            top_k=5,
            hybrid=hybrid,
            query_text=question if hybrid else None,
        )
        latency_ms = int((time.time() - start_time) * 1000)

        if not hits:
            continue

        retrieved_pages = [meta.get("page", 0) for _, meta, _ in hits]
        distances = [dist for _, _, dist in hits]

        exact_match = False
        in_top_k = False

        if expected_page is not None:
            exact_match = expected_page in retrieved_pages
            in_top_k = any(abs(expected_page - p) <= 2 for p in retrieved_pages)

            if exact_match:
                page_match_count += 1
            elif in_top_k:
                page_in_top_k_count += 1

        # Log to metrics collector
        if metrics_collector:
            metrics_collector.log_retrieval_metrics(
                query=question,
                strategy=strategy,
                hybrid=hybrid,
                exact_match=exact_match,
                in_top_k=in_top_k,
                distance=distances[0] if distances else 0.0,
                latency_ms=latency_ms,
                num_results=len(hits),
                expected_page=expected_page,
                retrieved_pages=retrieved_pages,
            )

        for dist in distances:
            distance_sum += dist
            distance_count += 1

    total = len(questions)
    return {
        "total_questions": total,
        "exact_matches": page_match_count,
        "tolerance_matches": page_in_top_k_count,
        "exact_accuracy": page_match_count / total if total > 0 else 0,
        "tolerance_accuracy": (page_match_count + page_in_top_k_count) / total if total > 0 else 0,
        "avg_distance": distance_sum / distance_count if distance_count > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="NGS RAG Drift Monitor")
    parser.add_argument("--pdf", type=str, default="data/TruSight-Oncology-500-v2.pdf")
    parser.add_argument(
        "--questions",
        type=str,
        default="validation/questions/TruSight-Oncology-500-v2_questions.json",
    )
    parser.add_argument("--strategy", type=str, default="basic",
                        choices=["basic", "table_aware", "semantic", "keyword"])
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--output", type=str, default="drift_metrics.json")
    parser.add_argument("--reference-texts", type=str, default=None,
                        help="JSON file with reference texts for embedding drift")
    parser.add_argument("--model", type=str, default="haybu/mxbai-embed-large-latest:latest")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434")

    args = parser.parse_args()

    # Initialize metrics collector
    metrics_collector = MetricsCollector()

    # Load reference texts
    if args.reference_texts and os.path.exists(args.reference_texts):
        with open(args.reference_texts, 'r') as f:
            reference_texts = json.load(f)
    else:
        reference_texts = DEFAULT_REFERENCE_TEXTS

    # Load previous metrics
    previous_metrics = load_previous_metrics(args.output)
    previous_embeddings = previous_metrics.get("embedding_drift", {}).get("current_embeddings", {})

    # Initialize embedder
    embedder = OllamaEmbedder(host=args.ollama_host, model=args.model)

    # Compute embedding drift + log metrics
    print("Computing embedding drift for reference texts...")
    for text in reference_texts:
        start_time = time.time()
        emb = embedder.embed(text)
        latency_ms = int((time.time() - start_time) * 1000)
        if emb:
            metrics_collector.log_embedding_metrics(
                model=args.model,
                text_length=len(text),
                embedding_dim=len(emb),
                latency_ms=latency_ms,
            )

    embedding_drift = compute_embedding_drift(embedder, reference_texts, previous_embeddings)

    # Run retrieval test if PDF exists
    retrieval_metrics = None
    if os.path.exists(args.pdf):
        print("\nRunning retrieval accuracy test...")
        try:
            retrieval_metrics = run_retrieval_test(
                pdf_path=args.pdf,
                questions_path=args.questions,
                strategy=args.strategy,
                hybrid=args.hybrid,
                embedder_model=args.model,
                ollama_host=args.ollama_host,
                metrics_collector=metrics_collector,
            )
        except Exception as e:
            print(f"⚠️ Retrieval test failed: {e}")
    else:
        print(f"\n⚠️ PDF not found at {args.pdf}, skipping retrieval test")
        retrieval_metrics = {
            "total_questions": 0,
            "exact_matches": 0,
            "tolerance_matches": 0,
            "exact_accuracy": 0,
            "tolerance_accuracy": 0,
            "avg_distance": 0,
        }

    # Combine metrics
    timestamp = datetime.now().isoformat()
    current_entry = {
        "timestamp": timestamp,
        "embedding_model": args.model,
        "chunking_strategy": args.strategy,
        "hybrid_search": args.hybrid,
        "retrieval_metrics": retrieval_metrics,
        "embedding_drift": embedding_drift,
    }

    # Append to output file
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            all_metrics = json.load(f)
            if not isinstance(all_metrics, list):
                all_metrics = [all_metrics]
    else:
        all_metrics = []

    all_metrics.append(current_entry)

    with open(args.output, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"DRIFT MONITOR SUMMARY ({timestamp})")
    print(f"{'='*60}")
    print(f"\nRetrieval Metrics:")
    print(f"  Exact Accuracy: {retrieval_metrics['exact_accuracy']*100:.1f}%")
    print(f"  Tolerance Accuracy: {retrieval_metrics['tolerance_accuracy']*100:.1f}%")
    print(f"  Avg Distance: {retrieval_metrics['avg_distance']:.3f}")

    print(f"\nEmbedding Drift (avg similarity): {embedding_drift['avg_similarity']:.3f}")
    if embedding_drift['drift_results']:
        print("  Drift details:")
        for text, result in embedding_drift['drift_results'].items():
            status = "⚠️ DRIFT" if result['drifted'] else "✓ Stable"
            print(f"    {status} ({result['similarity']:.3f}): {text[:50]}...")

    print(f"\nMetrics appended to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
