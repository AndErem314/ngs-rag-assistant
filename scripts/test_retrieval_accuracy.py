"""
Retrieval Accuracy Test for NGS RAG Pipeline.

Tests the RAG pipeline's retrieval accuracy using the validation question set.
Measures how often the expected page is within the retrieved chunks' pages.

Usage:
    # With default settings (basic chunking, vector search only)
    python scripts/test_retrieval_accuracy.py

    # With table-aware chunking
    python scripts/test_retrieval_accuracy.py --strategy table_aware

    # With hybrid search
    python scripts/test_retrieval_accuracy.py --hybrid

    # Specify custom PDF and questions file
    python scripts/test_retrieval_accuracy.py --pdf data/manual.pdf --questions validation/questions/custom.json
"""

import argparse
import json
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion import chunk_document_with_strategy, ChunkingStrategy
from src.ingestion.pdf_parser import extract_pages
from src.embeddings.embedder import OllamaEmbedder
from src.retrieval.vector_store import VectorStore
from src.observability import MetricsCollector


def test_retrieval_accuracy(
    pdf_path: str,
    questions_path: str,
    strategy: str = "basic",
    hybrid: bool = False,
    top_k: int = 5,
    max_distance: float = 0.5,
    ollama_host: str = "http://localhost:11434",
    embedder_model: str = "qwen3-embedding:0.6b",
    metrics_collector: object = None,
):
    """
    Run retrieval accuracy test against validation question set.

    Returns:
        Dict with accuracy metrics.
    """
    print(f"\n{'='*60}")
    print(f"NGS RAG Retrieval Accuracy Test")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"Questions: {questions_path}")
    print(f"Strategy: {strategy}")
    print(f"Hybrid search: {hybrid}")
    print(f"{'='*60}\n")

    # 1. Load questions
    print(f"Loading questions from {questions_path}...")
    try:
        with open(questions_path, 'r') as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions.")
    except Exception as e:
        print(f"Error loading questions: {e}")
        return None

    # 2. Extract pages from PDF
    print(f"\nExtracting pages from {pdf_path}...")
    try:
        pages = extract_pages(pdf_path)
        print(f"Extracted {len(pages)} pages.")
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None

    # 3. Chunk the document
    print(f"\nChunking with strategy: {strategy}...")
    try:
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
        print(f"Created {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error chunking: {e}")
        return None

    # 4. Initialize embedder and vector store
    print(f"\nInitializing Ollama embedder ({embedder_model})...")
    try:
        embedder = OllamaEmbedder(host=ollama_host, model=embedder_model)
        # Test embedder
        test_emb = embedder.embed("test")
        if not test_emb:
            print("Error: Embedder returned empty result. Is Ollama running?")
            return None
        print(f"Embedder ready (dimension: {len(test_emb)})")
    except Exception as e:
        print(f"Error initializing embedder: {e}")
        return None

    print(f"\nInitializing vector store...")
    try:
        vector_store = VectorStore(
            collection_name=f"test_accuracy_{strategy}",
            persist_directory="./chroma_test_temp"
        )
        vector_store.clear_collection()
        print(f"Vector store ready (cleared previous data).")
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None

    # 5. Embed and store chunks
    print(f"\nEmbedding and storing {len(chunks)} chunks...")
    try:
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed_batch(chunk_texts)
        # Filter out empty embeddings
        valid = [(c, e) for c, e in zip(chunks, embeddings) if e]
        if not valid:
            print("Error: No valid embeddings generated.")
            return None
        valid_chunks, valid_embeddings = zip(*valid)
        vector_store.add_chunks(list(valid_chunks), list(valid_embeddings))
        print(f"Stored {len(valid_chunks)} chunks with valid embeddings.")
    except Exception as e:
        print(f"Error embedding/storing chunks: {e}")
        return None

    # 6. Run queries and check accuracy
    print(f"\n{'='*60}")
    print(f"Running retrieval accuracy test...")
    print(f"{'='*60}\n")

    results = []
    page_match_count = 0
    page_in_top_k_count = 0
    distance_sum = 0.0
    distance_count = 0

    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected_page = q.get("source_page", None)
        expected_answer = q.get("expected_answer", "")

        print(f"\n[{i}/{len(questions)}] Q: {question[:80]}...")
        if expected_page:
            print(f"    Expected page: {expected_page}")

        # Embed query
        try:
            query_emb = embedder.embed(question)
            if not query_emb:
                print(f"    ✗ No embedding for query")
                results.append({
                    "question": question,
                    "expected_page": expected_page,
                    "retrieved_pages": [],
                    "page_match": False,
                    "in_top_k": False,
                })
                continue
        except Exception as e:
            print(f"    ✗ Error embedding query: {e}")
            continue

        # Search with timing
        start_time = time.time()
        try:
            hits = vector_store.search(
                query_embedding=query_emb,
                top_k=top_k,
                max_distance=max_distance,
                hybrid=hybrid,
                query_text=question if hybrid else None,
            )
        except Exception as e:
            print(f"    ✗ Error searching: {e}")
            continue
        latency_ms = int((time.time() - start_time) * 1000)

        if not hits:
            print(f"    ✗ No results found")
            results.append({
                "question": question,
                "expected_page": expected_page,
                "retrieved_pages": [],
                "page_match": False,
                "in_top_k": False,
            })
            continue

        # Extract retrieved pages
        retrieved_pages = [meta.get("page", 0) for _, meta, _ in hits]
        distances = [dist for _, _, dist in hits]

        print(f"    Retrieved pages: {retrieved_pages}")
        if distances:
            print(f"    Distances: {[f'{d:.3f}' for d in distances]}")

        # Check accuracy
        page_match = False
        in_top_k = False

        if expected_page is not None:
            # Exact page match
            page_match = expected_page in retrieved_pages
            # Within ±2 pages tolerance
            in_top_k = any(
                abs(expected_page - p) <= 2 for p in retrieved_pages
            )

        if page_match:
            print(f"    ✓ Page match!")
            page_match_count += 1
        elif in_top_k:
            print(f"    ~ Within tolerance (±2 pages)")
            page_in_top_k_count += 1
        else:
            print(f"    ✗ No page match")

        # Track distances
        for dist in distances:
            distance_sum += dist
            distance_count += 1

        # Log to metrics collector
        if metrics_collector:
            metrics_collector.log_retrieval_metrics(
                query=question,
                strategy=strategy,
                hybrid=hybrid,
                exact_match=page_match,
                in_top_k=in_top_k,
                distance=distances[0] if distances else 0.0,
                latency_ms=latency_ms,
                num_results=len(hits),
                expected_page=expected_page,
                retrieved_pages=retrieved_pages,
            )

        results.append({
            "question": question,
            "expected_page": expected_page,
            "retrieved_pages": retrieved_pages,
            "distances": distances,
            "page_match": page_match,
            "in_top_k": in_top_k,
        })

    # 7. Calculate metrics
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}\n")

    total = len(questions)
    exact_accuracy = page_match_count / total if total > 0 else 0
    tolerance_accuracy = (page_match_count + page_in_top_k_count) / total if total > 0 else 0
    avg_distance = distance_sum / distance_count if distance_count > 0 else 0

    metrics = {
        "total_questions": total,
        "exact_page_matches": page_match_count,
        "tolerance_matches": page_in_top_k_count,
        "exact_accuracy": exact_accuracy,
        "tolerance_accuracy": tolerance_accuracy,
        "avg_distance": avg_distance,
    }

    print(f"Total questions:        {total}")
    print(f"Exact page matches:     {page_match_count} ({exact_accuracy*100:.1f}%)")
    print(f"Within ±2 pages:        {page_match_count + page_in_top_k_count} ({tolerance_accuracy*100:.1f}%)")
    print(f"Average distance:        {avg_distance:.3f}")
    print(f"\nStrategy: {strategy}")
    print(f"Hybrid search: {hybrid}")
    print(f"{'='*60}\n")

    # Save results
    results_file = f"retrieval_accuracy_results_{strategy}_{'hybrid' if hybrid else 'vector'}.json"
    output = {
        "metrics": metrics,
        "results": results,
    }
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {results_file}\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NGS RAG Retrieval Accuracy Test")
    parser.add_argument("--pdf", type=str, default="data/TruSight-Oncology-500-v2.pdf",
                        help="Path to the PDF manual")
    parser.add_argument("--questions", type=str,
                        default="validation/questions/TruSight-Oncology-500-v2_questions.json",
                        help="Path to validation questions JSON")
    parser.add_argument("--strategy", type=str, default="basic",
                        choices=["basic", "table_aware", "semantic", "keyword"],
                        help="Chunking strategy to use")
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid search (vector + BM25)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to retrieve")
    parser.add_argument("--max-distance", type=float, default=0.5,
                        help="Maximum cosine distance for results")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434",
                        help="Ollama server URL")
    parser.add_argument("--model", type=str, default="qwen3-embedding:0.6b",
                        help="Ollama embedding model")

    args = parser.parse_args()

    # Initialize metrics collector
    metrics_collector = MetricsCollector()

    # Check if PDF exists
    if not os.path.exists(args.pdf):
        print(f"Error: PDF not found: {args.pdf}")
        print("Please place your PDF in the data/ folder or specify the correct path.")
        sys.exit(1)

    # Check if questions file exists
    if not os.path.exists(args.questions):
        print(f"Error: Questions file not found: {args.questions}")
        sys.exit(1)

    metrics = test_retrieval_accuracy(
        pdf_path=args.pdf,
        questions_path=args.questions,
        strategy=args.strategy,
        hybrid=args.hybrid,
        top_k=args.top_k,
        max_distance=args.max_distance,
        ollama_host=args.ollama_host,
        embedder_model=args.model,
        metrics_collector=metrics_collector,
    )

    if metrics:
        print("Test completed successfully.")
    else:
        print("Test failed.")
        sys.exit(1)
