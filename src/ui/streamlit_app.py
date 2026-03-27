# src/ui/streamlit_app.py

import streamlit as st
import os
from typing import List, Dict, Optional
from pathlib import Path

# Import our modules
from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_document
from src.embeddings.embedder import OllamaEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.query_processor import retrieve_context
from src.generation.llm_client import OllamaGenerator
from src.report.report_builder import generate_report

# ----------------------------------------------------------------------
# Helper functions for session state management
def init_session_state():
    """Initialize session state variables."""
    if "embedder" not in st.session_state:
        st.session_state.embedder = OllamaEmbedder()
    if "generator" not in st.session_state:
        st.session_state.generator = OllamaGenerator()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(collection_name="ngs_docs", persist_directory="./chroma_db")
    if "sources" not in st.session_state:
        st.session_state.sources = []  # list of source filenames
    if "ingested" not in st.session_state:
        st.session_state.ingested = False  # flag to indicate if any PDFs have been ingested

def clear_collection():
    """Clear the vector store and reset session state sources."""
    with st.spinner("Clearing collection..."):
        st.session_state.vector_store.clear_collection()
        st.session_state.sources = []
        st.session_state.ingested = False
        st.success("Collection cleared. You can now ingest new PDFs.")

def process_pdfs(uploaded_files):
    all_chunks = []
    all_embeddings = []
    all_sources = []

    for uploaded_file in uploaded_files:
        # Directly use the BytesIO object
        pages = extract_pages(uploaded_file)
        if not pages:
            st.error(f"Could not extract text from {uploaded_file.name}. Skipping.")
            continue

        chunks = chunk_document(pages, uploaded_file.name, chunk_size=500, overlap=50)
        if not chunks:
            st.warning(f"No chunks generated for {uploaded_file.name}.")
            continue

        texts = [chunk["text"] for chunk in chunks]
        embeddings = st.session_state.embedder.embed_batch(texts)
        if not embeddings or len(embeddings) != len(chunks):
            st.error(f"Embedding failed for {uploaded_file.name}. Skipping.")
            continue

        st.session_state.vector_store.add_chunks(chunks, embeddings)
        all_sources.append(uploaded_file.name)

    st.session_state.sources.extend(all_sources)
    st.session_state.ingested = True
    st.success(f"Ingested {len(all_sources)} PDF(s) with {len(all_chunks)} chunks.")

# ----------------------------------------------------------------------
# Main Streamlit app
def main():
    st.set_page_config(page_title="NGS‑RAG Assistant", page_icon="🧬")
    st.title("🧬 NGS‑RAG Assistant")
    st.markdown("""
    **Local RAG assistant for NGS sample preparation protocols.**  
    Upload PDF user manuals (max 5), then ask questions or generate structured reports.  
    All processing runs locally using Ollama models.
    """)

    init_session_state()

    # Sidebar
    with st.sidebar:
        st.header("📄 Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload PDFs (up to 5)",
            type="pdf",
            accept_multiple_files=True,
            key="uploader"
        )

        if uploaded_files and len(uploaded_files) > 5:
            st.warning("Maximum 5 files allowed. Only the first 5 will be processed.")
            uploaded_files = uploaded_files[:5]

        if st.button("📥 Ingest PDFs", type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    process_pdfs(uploaded_files)

        if st.session_state.ingested:
            st.divider()
            st.subheader("🔍 Select Protocols")
            selected_sources = []
            for src in st.session_state.sources:
                if st.checkbox(src, value=True, key=f"src_{src}"):
                    selected_sources.append(src)
            st.session_state.selected_sources = selected_sources

            if st.button("🗑️ Clear Collection", on_click=clear_collection):
                pass  # handled by on_click

    # Main area
    if not st.session_state.ingested:
        st.info("👈 Start by uploading PDFs and clicking 'Ingest PDFs'.")
        return

    # Query section
    st.header("💬 Ask a Question")
    question = st.text_input("Enter your question:")
    if st.button("🔍 Ask", type="primary") and question:
        with st.spinner("Searching and generating answer..."):
            context, metadata = retrieve_context(
                question=question,
                embedder=st.session_state.embedder,
                vector_store=st.session_state.vector_store,
                source_filter=st.session_state.selected_sources,
                top_k=5
            )
            if not context:
                st.warning("No relevant information found for the selected protocols.")
            else:
                answer = st.session_state.generator.answer_question(question, context, metadata)
                st.subheader("Answer")
                st.markdown(answer)
                if metadata:
                    st.subheader("Sources")
                    sources_md = "\n".join([f"- {m['source']}, page {m['page']}" for m in metadata])
                    st.markdown(sources_md)

    # Report generation section
    st.header("📝 Generate Report")
    if st.button("📊 Generate Report"):
        with st.spinner("Generating report..."):
            report = generate_report(
                selected_sources=st.session_state.selected_sources,
                embedder=st.session_state.embedder,
                vector_store=st.session_state.vector_store,
                generator=st.session_state.generator
            )
        st.subheader("Report")
        st.markdown(report)

        # Download button
        st.download_button(
            label="📥 Download as Markdown",
            data=report,
            file_name="ngs_report.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()