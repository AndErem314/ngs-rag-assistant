# src/ui/streamlit_app.py

import os
import streamlit as st
from dotenv import load_dotenv

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_document
from src.embeddings.embedder import OllamaEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.query_processor import retrieve_context
from src.generation.llm_client import OllamaGenerator
from src.report.report_builder import generate_report

load_dotenv()

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_HOST  = os.getenv("OLLAMA_HOST",       "http://localhost:11434")
DEFAULT_EMBED_MODEL  = os.getenv("EMBEDDING_MODEL",   "nomic-embed-text-v2-moe")
DEFAULT_LLM_MODEL    = os.getenv("LLM_MODEL",         "llama3.1:8b")
MAX_PDFS             = 5
DEFAULT_TOP_K        = 5
DEFAULT_MAX_DISTANCE = 1.0   # sensible starting point for cosine distance


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialise all session-state keys exactly once per session."""
    defaults = {
        "embedder":         None,   # created after host is confirmed
        "generator":        None,
        "vector_store":     None,
        "sources":          [],     # list of ingested source filenames
        "ingested":         False,
        "selected_sources": [],
        "ollama_ok":        None,   # None = unchecked, True/False = result
        "max_distance":     DEFAULT_MAX_DISTANCE,
        "top_k":            DEFAULT_TOP_K,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_clients(host: str) -> None:
    """(Re-)instantiate Ollama clients bound to the given host."""
    st.session_state.embedder = OllamaEmbedder(
        host=host, model=DEFAULT_EMBED_MODEL
    )
    st.session_state.generator = OllamaGenerator(
        host=host, model=DEFAULT_LLM_MODEL
    )
    st.session_state.vector_store = VectorStore(
        collection_name="ngs_docs", persist_directory="./chroma_db"
    )


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def check_ollama(host: str) -> bool:
    """
    Ping Ollama by listing available models.
    Returns True if reachable, False otherwise.
    Sets st.session_state.ollama_ok.
    """
    try:
        import ollama
        client = ollama.Client(host=host)
        client.list()
        st.session_state.ollama_ok = True
        return True
    except Exception:
        st.session_state.ollama_ok = False
        return False


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def process_pdfs(uploaded_files) -> None:
    """
    Parse → chunk → embed → upsert for each uploaded PDF.

    Tracks and reports per-file chunk counts and a session-wide total.
    Skips files that fail at any stage rather than aborting the whole batch.
    """
    new_sources: list[str] = []
    total_chunks = 0

    progress = st.progress(0, text="Starting ingestion…")

    for idx, uploaded_file in enumerate(uploaded_files):
        name = uploaded_file.name
        progress.progress(idx / len(uploaded_files), text=f"Processing {name}…")

        # 1. Parse
        pages = extract_pages(uploaded_file)
        if not pages:
            st.error(f"❌ Could not extract text from **{name}**. Skipping.")
            continue

        # 2. Chunk
        chunks = chunk_document(pages, name, chunk_size=500, overlap=50)
        if not chunks:
            st.warning(f"⚠️ No chunks generated for **{name}**. Skipping.")
            continue

        # 3. Embed
        texts      = [c["text"] for c in chunks]
        embeddings = st.session_state.embedder.embed_batch(texts)

        # Guard: filter out any failed (empty) embeddings before upserting
        valid_pairs = [
            (c, e) for c, e in zip(chunks, embeddings) if e
        ]
        if not valid_pairs:
            st.error(f"❌ All embeddings failed for **{name}**. Is Ollama running?")
            continue

        if len(valid_pairs) < len(chunks):
            st.warning(
                f"⚠️ {len(chunks) - len(valid_pairs)} chunk(s) from **{name}** "
                f"could not be embedded and were skipped."
            )

        valid_chunks, valid_embeddings = zip(*valid_pairs)

        # 4. Upsert (safe on re-ingestion thanks to content-hash IDs)
        st.session_state.vector_store.add_chunks(
            list(valid_chunks), list(valid_embeddings)
        )

        n = len(valid_chunks)
        total_chunks += n
        new_sources.append(name)
        st.info(f"✅ **{name}** — {n} chunk(s) ingested.")

    progress.progress(1.0, text="Done.")

    # Merge new sources without duplicates, preserving order
    existing = set(st.session_state.sources)
    for src in new_sources:
        if src not in existing:
            st.session_state.sources.append(src)
            existing.add(src)

    st.session_state.ingested = bool(st.session_state.sources)

    if new_sources:
        st.success(
            f"🎉 Ingested **{len(new_sources)}** PDF(s) producing "
            f"**{total_chunks}** chunk(s) in total."
        )
    else:
        st.error("No PDFs were successfully ingested.")


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def clear_collection() -> None:
    with st.spinner("Clearing collection…"):
        st.session_state.vector_store.clear_collection()
    st.session_state.sources          = []
    st.session_state.selected_sources = []
    st.session_state.ingested         = False
    st.success("Collection cleared. You can now ingest new PDFs.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="NGS‑RAG Assistant", page_icon="🧬", layout="wide")
    st.title("🧬 NGS‑RAG Assistant")
    st.caption(
        "Local RAG assistant for NGS sample preparation protocols. "
        "Upload PDF manuals, ask questions, or generate structured reports. "
        "All processing runs locally via Ollama — no data leaves your machine."
    )

    init_session_state()

    # ------------------------------------------------------------------
    # Sidebar — configuration + ingestion
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        ollama_host = st.text_input(
            "Ollama host",
            value=DEFAULT_OLLAMA_HOST,
            help="URL of the Ollama server. Change if running on a remote machine.",
        )

        if st.button("🔌 Check connection"):
            with st.spinner("Pinging Ollama…"):
                ok = check_ollama(ollama_host)
            if ok:
                st.success("Ollama is reachable.")
                _build_clients(ollama_host)
            else:
                st.error(
                    f"Cannot reach Ollama at **{ollama_host}**. "
                    "Make sure `ollama serve` is running."
                )

        # Auto-build clients on first load if we haven't yet
        if st.session_state.embedder is None:
            _build_clients(ollama_host)

        # Show live status indicator
        if st.session_state.ollama_ok is True:
            st.success("● Ollama connected", icon=None)
        elif st.session_state.ollama_ok is False:
            st.error("● Ollama unreachable", icon=None)

        st.divider()

        # Retrieval settings
        st.header("🔧 Retrieval settings")
        st.session_state.top_k = st.slider(
            "Top-K chunks",
            min_value=1, max_value=20,
            value=st.session_state.top_k,
            help="Maximum number of chunks retrieved per query before distance filtering.",
        )
        st.session_state.max_distance = st.slider(
            "Max distance threshold",
            min_value=0.1, max_value=2.0,
            value=st.session_state.max_distance,
            step=0.05,
            help=(
                "ChromaDB cosine distance upper bound. "
                "Lower = stricter (only very close chunks pass). "
                "0.5–0.8 is a good starting range; 1.0 is permissive."
            ),
        )

        st.divider()

        # Ingestion
        st.header("📄 Document ingestion")
        uploaded_files = st.file_uploader(
            f"Upload PDFs (up to {MAX_PDFS})",
            type="pdf",
            accept_multiple_files=True,
            key="uploader",
        )

        if uploaded_files and len(uploaded_files) > MAX_PDFS:
            st.warning(f"Maximum {MAX_PDFS} files. Only the first {MAX_PDFS} will be processed.")
            uploaded_files = uploaded_files[:MAX_PDFS]

        if st.button("📥 Ingest PDFs", type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one PDF first.")
            else:
                process_pdfs(uploaded_files)

        # Source selector (only shown after at least one successful ingest)
        if st.session_state.ingested:
            st.divider()
            st.subheader("🔍 Active protocols")
            st.caption("Uncheck a protocol to exclude it from queries and reports.")

            selected: list[str] = []
            for src in st.session_state.sources:
                if st.checkbox(src, value=True, key=f"src_{src}"):
                    selected.append(src)
            st.session_state.selected_sources = selected

            if not selected:
                st.warning("No protocols selected — queries will return no results.")

            st.divider()
            if st.button("🗑️ Clear collection"):
                clear_collection()

    # ------------------------------------------------------------------
    # Main area
    # ------------------------------------------------------------------
    if not st.session_state.ingested:
        st.info("👈 Start by uploading one or more PDFs and clicking **Ingest PDFs**.")
        return

    # Resolve selected sources; fall back to all sources if somehow empty
    selected_sources = st.session_state.get("selected_sources") or st.session_state.sources

    # Guard: if the user deselected everything, warn and bail early
    if not selected_sources:
        st.warning(
            "No protocols are selected. "
            "Tick at least one protocol in the sidebar before querying."
        )
        return

    # ------------------------------------------------------------------
    # Q&A section
    # ------------------------------------------------------------------
    st.header("💬 Ask a question")
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g. What is the minimum DNA input for TruSight Oncology 500?",
    )

    if st.button("🔍 Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context and generating answer…"):
                context, metadata = retrieve_context(
                    question=question,
                    embedder=st.session_state.embedder,
                    vector_store=st.session_state.vector_store,
                    source_filter=selected_sources,
                    top_k=st.session_state.top_k,
                    max_distance=st.session_state.max_distance,
                )

            if not context:
                st.warning(
                    "No relevant chunks found. Try raising the **Max distance** "
                    "threshold in the sidebar, or check that the correct protocols "
                    "are selected."
                )
            else:
                answer = st.session_state.generator.answer_question(
                    question, context, metadata
                )
                st.subheader("Answer")
                st.markdown(answer)

                if metadata:
                    with st.expander("📚 Sources & relevance scores", expanded=False):
                        for m in metadata:
                            dist_label = f"distance {m['distance']:.4f}"
                            st.markdown(f"- **{m['source']}**, page {m['page']} — *{dist_label}*")

    st.divider()

    # ------------------------------------------------------------------
    # Report section
    # ------------------------------------------------------------------
    st.header("📝 Generate report")
    st.caption(
        "Runs a set of pre-defined protocol questions against the selected sources "
        "and compiles the answers into a Markdown report."
    )

    if st.button("📊 Generate report"):
        with st.spinner("Generating report — this may take a minute…"):
            report = generate_report(
                selected_sources=selected_sources,
                embedder=st.session_state.embedder,
                vector_store=st.session_state.vector_store,
                generator=st.session_state.generator,
            )

        if not report.strip():
            st.warning("Report is empty. Check that PDFs were ingested correctly.")
        else:
            st.subheader("Report")
            st.markdown(report)
            st.download_button(
                label="📥 Download as Markdown",
                data=report,
                file_name="ngs_report.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()