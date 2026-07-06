#!/usr/bin/env python3
"""
Quality Streak Evaluation Harness for NGS RAG Assistant.

Runs the full RAG pipeline against validation questions + AI-generated
scenarios, scores each answer with a 4-dimension rubric, and outputs
a JSON report used by the streak-loop cron job.

Usage:
    python scripts/streak_eval.py                          # default settings
    python scripts/streak_eval.py --limit 10               # only 10 questions
    python scripts/streak_eval.py --no-ai-questions         # validation only
    python scripts/streak_eval.py --output /tmp/eval.json  # custom output path

Model: llama3.1:8b (local Ollama) for both answer generation and scoring.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — run from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedder import OllamaEmbedder
from src.retrieval.vector_store import VectorStore
from src.generation.llm_client import OllamaGenerator
from src.retrieval.query_processor import retrieve_context

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
GEN_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "mistral-nemo:12b")  # Separate model for scoring
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", str(PROJECT_ROOT / "chroma_db"))
VALIDATION_DIR = PROJECT_ROOT / "validation" / "questions"
STREAK_LOG = PROJECT_ROOT / ".hermes" / "streak-log.md"
DEFAULT_MAX_RUNTIME = 600  # 10 minutes — hard safety limit

# Scoring weights (must sum to 100)
WEIGHT_FACTUAL = 40      # Does the answer match ground truth?
WEIGHT_HALLUCINATION = 30 # Does the answer avoid fabricating facts?
WEIGHT_COMPLETENESS = 20  # Does the answer cover all key points?
WEIGHT_SOURCE = 10        # Does the answer cite correct sources?

PASS_THRESHOLD = 60  # Overall score >= this = pass

# Tier 3 files — changes here halt the loop
TIER_3_FILES = {
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
}

# Tier 2 patterns — changes here need review
TIER_2_PATTERNS = [
    "src/ingestion/chunker.py",
    "src/retrieval/query_processor.py",
    "src/retrieval/vector_store.py",
    "src/retrieval/pgvector_store.py",
    "src/embeddings/embedder.py",
    "scripts/drift_monitor.py",
]

# Tier 1 patterns — safe to auto-merge
TIER_1_PATTERNS = [
    "src/generation/llm_client.py",
    "src/report/report_builder.py",
    "src/ui/streamlit_app.py",
    "validation/questions/",
    "README.md",
    ".hermes/",
]


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------
def load_validation_questions() -> List[Dict]:
    """Load ground-truth questions from validation JSON files."""
    all_questions = []
    if not VALIDATION_DIR.exists():
        print(f"⚠️  Validation directory not found: {VALIDATION_DIR}")
        return all_questions

    for json_file in sorted(VALIDATION_DIR.glob("*.json")):
        try:
            with open(json_file) as f:
                questions = json.load(f)
                for q in questions:
                    q["_source_file"] = json_file.name
                all_questions.extend(questions)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Skipping {json_file.name}: {e}")

    return all_questions


def generate_ai_questions(generator: OllamaGenerator, count: int = 10) -> List[Dict]:
    """Use the LLM to generate additional scenario questions for broader coverage."""
    system_prompt = (
        "You are an NGS protocol testing expert. Generate diverse questions that "
        "test a RAG system's ability to answer from Illumina protocol PDFs. "
        "Cover these categories: DNA/RNA input amounts, library preparation steps, "
        "PCR cycles, cleanup procedures, reagent storage, quality control, "
        "troubleshooting, and instrument settings. "
        "Output ONLY a JSON array of objects with 'question' and 'expected_answer' keys. "
        "Make expected_answer a concise 1-2 sentence factual answer. "
        "Generate exactly the requested number of questions."
    )

    user_prompt = f"Generate {count} diverse NGS protocol questions. Return ONLY valid JSON array."

    try:
        response = generator.client.chat(
            model=generator.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response["message"]["content"]
        # Extract JSON from possible markdown code block
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        questions = json.loads(text.strip())
        for q in questions:
            q["_source_file"] = "ai-generated"
        return questions
    except Exception as e:
        print(f"⚠️  Failed to generate AI questions: {e}")
        return []


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------
def run_rag_pipeline(
    question: str,
    embedder: OllamaEmbedder,
    vector_store: VectorStore,
    generator: OllamaGenerator,
    max_distance: float = 0.65,
hybrid: bool = True,
) -> Tuple[str, List[Dict]]:
    """Run a question through the full RAG pipeline and return answer + metadata."""
    context, metadata = retrieve_context(
        question=question,
        embedder=embedder,
        vector_store=vector_store,
        top_k=10,
        max_distance=max_distance,
        hybrid=hybrid,
    )

    if not context:
        return "No relevant information found.", []

    answer = generator.answer_question(question, context, metadata)
    return answer, metadata


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _fallback_score(answer: str, expected: str) -> Dict:
    """Rule-based fallback when LLM scoring fails."""
    # Simple keyword overlap as rough accuracy measure
    expected_keywords = set(expected.lower().split())
    answer_keywords = set(answer.lower().split())
    if not expected_keywords:
        return {"factual_accuracy": 0, "hallucination": 10, "completeness": 0,
                "source_fidelity": 0, "explanation": "Fallback: empty expected"}
    overlap = len(expected_keywords & answer_keywords) / len(expected_keywords)
    factual = min(int(overlap * WEIGHT_FACTUAL), WEIGHT_FACTUAL)
    return {
        "factual_accuracy": factual,
        "hallucination": max(WEIGHT_HALLUCINATION - 10, 10),  # conservative
        "completeness": min(int(overlap * WEIGHT_COMPLETENESS), WEIGHT_COMPLETENESS),
        "source_fidelity": 0,
        "explanation": "Fallback rule-based score (LLM unavailable)",
    }


def score_answer(
    question: str,
    answer: str,
    expected: str,
    metadata: List[Dict],
    judge_model: str,
    ollama_host: str = OLLAMA_HOST,
) -> Dict:
    """
    Score an answer using a dedicated judge LLM across 4 dimensions.

    Uses a separate model (default: mistral-nemo:12b) from the answer generator
    for more nuanced and consistent evaluation.

    Returns a dict with per-dimension scores, overall score, and explanation.
    """
    sources_text = "\n".join(
        f"- {m.get('source', 'unknown')}, page {m.get('page', '?')} (distance {m.get('distance', '?')})"
        for m in metadata
    ) if metadata else "No sources available"

    system_prompt = (
        "You are an expert evaluator scoring RAG system answers against ground truth. "
        "Score on 4 dimensions and return ONLY a JSON object. No other text.\n\n"
        f"Dimensions (weighted):\n"
        f"- factual_accuracy (0-{WEIGHT_FACTUAL}): How well does the answer match the expected answer?\n"
        f"- hallucination (0-{WEIGHT_HALLUCINATION}): Does the answer avoid fabricating facts not in context?\n"
        f"- completeness (0-{WEIGHT_COMPLETENESS}): Does it cover all key points from the expected answer?\n"
        f"- source_fidelity (0-{WEIGHT_SOURCE}): Does it cite correct sources/pages?\n\n"
        "JSON format: "
        '{"factual_accuracy": N, "hallucination": N, "completeness": N, '
        '"source_fidelity": N, "explanation": "brief reason"}'
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"RAG answer: {answer}\n\n"
        f"Retrieved sources:\n{sources_text}"
    )

    try:
        import ollama as ollama_lib
        client = ollama_lib.Client(host=ollama_host)
        response = client.chat(
            model=judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.1},  # Low temp for consistent scoring
        )
        text = response["message"]["content"]
        # Extract JSON — handle code blocks, trailing text, and multiple JSON objects
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        # Strip any text before the first { and after the matching }
        text = text.strip()
        if "{" in text:
            start = text.index("{")
            depth = 0
            end = start
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            text = text[start:end]
        result = json.loads(text.strip())
    except Exception as e:
        print(f"  ⚠️  Scoring failed with {judge_model}, falling back to rule-based: {e}")
        result = _fallback_score(answer, expected)

    # Compute total
    total = sum([
        result.get("factual_accuracy", 0),
        result.get("hallucination", 0),
        result.get("completeness", 0),
        result.get("source_fidelity", 0),
    ])

    return {
        "question": question,
        "expected_answer": expected,
        "rag_answer": answer,
        "sources": [
            {
                "source": m.get("source", "unknown"),
                "page": m.get("page", 0),
                "distance": m.get("distance", None),
            }
            for m in metadata
        ],
        "factual_accuracy": result.get("factual_accuracy", 0),
        "hallucination": result.get("hallucination", 0),
        "completeness": result.get("completeness", 0),
        "source_fidelity": result.get("source_fidelity", 0),
        "total_score": total,
        "passed": total >= PASS_THRESHOLD,
        "explanation": result.get("explanation", ""),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(
    questions: List[Dict],
    embedder: OllamaEmbedder,
    vector_store: VectorStore,
    generator: OllamaGenerator,
    judge_model: str = JUDGE_MODEL,
    limit: Optional[int] = None,
    max_distance: float = 0.65,
    hybrid: bool = True,
) -> Dict:
    """Run the full evaluation across all (or limited) questions."""
    if limit:
        questions = questions[:limit]

    results = []
    passed = 0
    failed = 0
    total_score = 0

    print(f"\n🔬 Evaluating {len(questions)} questions... (judge: {judge_model})\n")

    for i, q in enumerate(questions):
        question_text = q.get("question", "")
        expected = q.get("expected_answer", "")
        source_file = q.get("_source_file", "unknown")

        print(f"  [{i+1}/{len(questions)}] {question_text[:80]}...")
        sys.stdout.flush()

        answer, metadata = run_rag_pipeline(
            question_text, embedder, vector_store, generator,
            max_distance=max_distance,
            hybrid=hybrid,
        )

        score = score_answer(question_text, answer, expected, metadata, judge_model)
        score["_source_file"] = source_file
        results.append(score)

        if score["passed"]:
            passed += 1
            print(f"      ✅ {score['total_score']}/100")
        else:
            failed += 1
            print(f"      ❌ {score['total_score']}/100 — {score['explanation'][:60]}")

        total_score += score["total_score"]
        sys.stdout.flush()

    avg_score = total_score / len(results) if results else 0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(results) * 100, 1) if results else 0,
        "average_score": round(avg_score, 1),
        "pass_threshold": PASS_THRESHOLD,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Streak log update
# ---------------------------------------------------------------------------
def update_streak_log(report: Dict, previous_streak: int, new_streak: int):
    """Append a new row to the streak log file."""
    STREAK_LOG.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    date_str = now.strftime("%d %b %Y")
    run_id = now.strftime("%Y%m%d-%H%M")

    # Summarize changes from results
    failed_questions = [r for r in report["results"] if not r["passed"]]
    changes = ", ".join(
        r["question"][:60] + ("..." if len(r["question"]) > 60 else "")
        for r in failed_questions[:3]
    ) if failed_questions else "none needed"

    new_row = (
        f"| {run_id} | {date_str} | {report['passed']}/{report['total_questions']} "
        f"| {new_streak} | {changes} | — | {report['average_score']}% |\n"
    )

    if STREAK_LOG.exists():
        with open(STREAK_LOG, "r") as f:
            content = f.read()

        # Insert after the table header separator
        insert_after = "|------|------|--------|--------|---------|----------|------------|"
        if insert_after in content:
            parts = content.split(insert_after)
            new_content = parts[0] + insert_after + "\n" + new_row + parts[1]
        else:
            new_content = content + new_row
    else:
        # Create fresh log file
        header = (
            "# NGS RAG — Quality Streak Log\n\n"
            "Automated daily evaluation. Each run generates questions, scores answers, "
            "and applies Tier 1 fixes automatically.\n\n"
            "| Run ID | Date | Passed | Streak | Changes | Decision | Avg Score |\n"
            "|------|------|--------|--------|---------|----------|------------|\n"
        )
        new_content = header + new_row

    with open(STREAK_LOG, "w") as f:
        f.write(new_content)

    print(f"\n📋 Streak log updated: {STREAK_LOG}")


def get_previous_streak() -> int:
    """Read the streak from the last row of the streak log."""
    if not STREAK_LOG.exists():
        return 0

    try:
        with open(STREAK_LOG) as f:
            lines = f.readlines()
        # Find the last data row (starts with |)
        for line in reversed(lines):
            if line.startswith("|") and "Run ID" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5:
                    return int(parts[4])  # Streak column
    except (ValueError, IndexError):
        pass
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Quality Streak Evaluation for NGS RAG Assistant"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: stdout + streak-log update)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions to evaluate",
    )
    parser.add_argument(
        "--no-ai-questions", action="store_true",
        help="Skip AI-generated scenario questions, use only validation",
    )
    parser.add_argument(
        "--ai-count", type=int, default=10,
        help="Number of AI-generated questions (default: 10)",
    )
    parser.add_argument(
        "--max-distance", type=float, default=0.65,
        help="Max cosine distance for retrieval (default: 0.45)",
    )
    parser.add_argument(
        "--judge-model", type=str, default=JUDGE_MODEL,
        help=f"Ollama model for scoring (default: {JUDGE_MODEL})",
    )
    parser.add_argument(
        "--max-runtime", type=int, default=DEFAULT_MAX_RUNTIME,
        help=f"Hard limit in seconds before aborting (default: {DEFAULT_MAX_RUNTIME}s)",
    )
    parser.add_argument(
        "--no-update-log", action="store_true",
        help="Skip updating the streak log",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Initialize components
    # -----------------------------------------------------------------------
    print("🔧 Initializing...")
    embedder = OllamaEmbedder(host=OLLAMA_HOST, model=EMBED_MODEL)
    vector_store = VectorStore(persist_directory=CHROMA_DIR)
    generator = OllamaGenerator(host=OLLAMA_HOST, model=GEN_MODEL)

    # Quick connectivity check
    try:
        _ = embedder.embed("test")
        print("   ✅ Embedder connected")
    except Exception as e:
        print(f"   ❌ Embedder failed: {e}")
        return 1

    # Verify judge model availability
    try:
        import ollama as ollama_lib
        judge_client = ollama_lib.Client(host=OLLAMA_HOST)
        judge_client.chat(model=args.judge_model, messages=[
            {"role": "user", "content": "ping"}
        ])
        print(f"   ✅ Judge model ready: {args.judge_model}")
    except Exception as e:
        print(f"   ⚠️  Judge model {args.judge_model} unavailable: {e}")
        print(f"   ⚠️  Falling back to {GEN_MODEL} for scoring")
        args.judge_model = GEN_MODEL

    # -----------------------------------------------------------------------
    # Load questions
    # -----------------------------------------------------------------------
    questions = load_validation_questions()
    print(f"   📋 Loaded {len(questions)} validation questions")

    if not args.no_ai_questions:
        print(f"   🤖 Generating {args.ai_count} AI scenario questions...")
        ai_questions = generate_ai_questions(generator, args.ai_count)
        questions.extend(ai_questions)
        print(f"   📋 Total: {len(questions)} questions")

    if not questions:
        print("❌ No questions to evaluate. Exiting.")
        return 1

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    start_time = time.time()
    report = run_evaluation(
        questions=questions,
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        judge_model=args.judge_model,
        limit=args.limit,
        max_distance=args.max_distance,
    )
    elapsed = time.time() - start_time

    # Check hard runtime limit
    if elapsed > args.max_runtime:
        print(f"⏰ RUNTIME LIMIT EXCEEDED ({elapsed:.0f}s > {args.max_runtime}s) — results may be incomplete")
        # Don't exit — still log what we have

    # -----------------------------------------------------------------------
    # Compute streak
    # -----------------------------------------------------------------------
    previous_streak = get_previous_streak()
    new_streak = previous_streak + 1 if report["passed"] == report["total_questions"] else 0

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION COMPLETE — {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Questions: {report['total_questions']}")
    print(f"  Passed:    {report['passed']} ✅")
    print(f"  Failed:    {report['failed']} ❌")
    print(f"  Pass rate: {report['pass_rate']}%")
    print(f"  Avg score: {report['average_score']}/100")
    print(f"  Streak:    {previous_streak} → {new_streak} 🔥")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Update streak log
    # -----------------------------------------------------------------------
    if not args.no_update_log:
        update_streak_log(report, previous_streak, new_streak)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    output_data = {
        **report,
        "streak_before": previous_streak,
        "streak_after": new_streak,
        "elapsed_seconds": round(elapsed, 1),
        "model_used": GEN_MODEL,
        "judge_model_used": args.judge_model,
        "embed_model_used": EMBED_MODEL,
        "runtime_exceeded": elapsed > args.max_runtime,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"📄 Full report: {output_path}")
    else:
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

    return 0 if report["passed"] == report["total_questions"] else 1


if __name__ == "__main__":
    sys.exit(main())
