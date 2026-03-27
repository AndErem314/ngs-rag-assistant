from typing import List, Optional
from src.retrieval.query_processor import retrieve_context
from src.embeddings.embedder import OllamaEmbedder
from src.retrieval.vector_store import VectorStore
from src.generation.llm_client import OllamaGenerator

# Predefined questions for the report (customise as needed)
REPORT_QUESTIONS = [
    "What is the minimum DNA/RNA input amount?",
    "What are the recommended shearing settings for the Covaris instruments?",
    "List all reagents and their storage temperatures from the kit boxes.",
    "How many cycles are used in the index PCR?",
    "What are the steps in the library preparation workflow?",
    "What are the quality control criteria for DNA and RNA samples?",
    "What are the important safety precautions or handling notes?"
]

def generate_report(
    selected_sources: List[str],
    embedder: OllamaEmbedder,
    vector_store: VectorStore,
    generator: OllamaGenerator
) -> str:
    """
    Generate a structured Markdown report by asking predefined questions.

    Args:
        selected_sources: List of source filenames to include.
        embedder: OllamaEmbedder instance.
        vector_store: VectorStore instance.
        generator: OllamaGenerator instance.

    Returns:
        Markdown string with the report.
    """
    sections = []
    for q in REPORT_QUESTIONS:
        # Retrieve context
        context, metadata = retrieve_context(q, embedder, vector_store, source_filter=selected_sources, top_k=5)
        if not context:
            answer = "No relevant information found."
        else:
            answer = generator.answer_question(q, context, metadata)
        sections.append(f"## {q}\n\n{answer}\n")

    return "\n".join(sections)