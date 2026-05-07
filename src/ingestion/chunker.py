from typing import List, Tuple, Dict
from enum import Enum
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber

# ---------------------------------------------------------------------------
# Chunking Strategy
# ---------------------------------------------------------------------------

class ChunkingStrategy(Enum):
    """Available chunking strategies for NGS RAG pipeline."""
    BASIC = "basic"
    TABLE_AWARE = "table_aware"
    SEMANTIC = "semantic"
    KEYWORD_ANCHORED = "keyword"

# Common NGS keywords for keyword-anchored chunking
NGS_KEYWORDS = [
    "DNA", "RNA", "PCR", "QC", "input", "ng", "µL", "°C", "cycles",
    "TruSight", "Illumina", "sequencing", "library", "hybridization",
    "exon", "gene", "variant", "mutation", "Q30", "Q20"
]

# ---------------------------------------------------------------------------
# Existing chunk_document (unchanged, with syntax fixes)
# ---------------------------------------------------------------------------

def chunk_document(
    pages: List[Tuple[int, str]],
    source_filename: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict]:
    """
    Split a document's pages into overlapping chunks, attaching metadata.

    Args:
        pages: List of (page_num, text) for each page. Page numbers are 1-indexed.
        source_filename: The name of the source PDF file.
        chunk_size: Approximate number of tokens per chunk (character-based).
        overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of dicts, each with keys:
            - "text": the chunk text
            - "metadata": {"source": source_filename, "page": page_num}
        The page number is the page where the chunk starts (approximated).
    """
    if not pages:
        return []

    page_separator = "\n\n--- PAGE BREAK ---\n\n"
    full_text = ""
    page_boundaries = []  # list of (start_char_index, page_num)
    current_pos = 0

    for page_num, text in pages:
        if full_text:
            full_text += page_separator
            current_pos += len(page_separator)
        full_text += text
        page_boundaries.append((current_pos, page_num))
        current_pos += len(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(full_text)

    def get_page_for_pos(pos: int) -> int:
        for i in range(len(page_boundaries) - 1, -1, -1):
            if pos >= page_boundaries[i][0]:
                return page_boundaries[i][1]
        return 1

    last_pos = 0
    results = []
    for chunk in chunks:
        pos = full_text.find(chunk, last_pos)
        if pos == -1:
            pos = last_pos
        page_num = get_page_for_pos(pos)
        results.append({
            "text": chunk,
            "metadata": {"source": source_filename, "page": page_num, "type": "basic"}
        })
        last_pos = pos + len(chunk)

    return results

# ---------------------------------------------------------------------------
# New: Table-aware chunking (uses pdfplumber)
# ---------------------------------------------------------------------------

def chunk_document_table_aware(
    pdf_path: str,
    source_filename: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict]:
    """
    Chunk PDF with table-aware logic: extract tables separately, preserve structure.

    Args:
        pdf_path: Path to the PDF file (for pdfplumber table extraction).
        source_filename: Name of the source PDF (for metadata).
        chunk_size: Non-table text chunk size.
        overlap: Non-table text overlap.

    Returns:
        List of chunks with metadata, including table chunks (type: "table") and text chunks.
    """
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract tables from the page
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table:
                    continue
                table_text = f"Table {table_idx+1}:\n"
                if table[0]:
                    table_text += "Header: " + " | ".join(str(cell) for cell in table[0]) + "\n"
                for row in table[1:]:
                    table_text += "Row: " + " | ".join(str(cell) for cell in row) + "\n"
                results.append({
                    "text": table_text,
                    "metadata": {
                        "source": source_filename,
                        "page": page_num,
                        "type": "table",
                        "table_idx": table_idx
                    }
                })

            # Extract non-table text (fallback to regular chunking for page text)
            page_text = page.extract_text()
            if page_text:
                text_chunks = chunk_document(
                    pages=[(page_num, page_text)],
                    source_filename=source_filename,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                for chunk in text_chunks:
                    chunk["metadata"]["type"] = "text"
                results.extend(text_chunks)

    return results

# ---------------------------------------------------------------------------
# New: Semantic chunking (uses LangChain SemanticChunker)
# ---------------------------------------------------------------------------

def chunk_document_semantic(
    pages: List[Tuple[int, str]],
    source_filename: str,
    embedder_model: str = "haybu/mxbai-embed-large:latest",
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict]:
    """
    Chunk text semantically using LangChain's SemanticChunker (needs embeddings).

    Args:
        pages: List of (page_num, text) tuples.
        source_filename: Source filename for metadata.
        embedder_model: Ollama embedder model to use for semantic boundaries.
        chunk_size: Fallback chunk size for non-semantic splitting.
        overlap: Fallback overlap.

    Returns:
        List of semantically chunked dicts with metadata.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from src.embeddings.embedder import OllamaEmbedder

        embedder = OllamaEmbedder(model=embedder_model)

        # Wrapper to make OllamaEmbedder compatible with SemanticChunker
        class OllamaEmbeddingsWrapper:
            def __init__(self, embedder: OllamaEmbedder):
                self.embedder = embedder
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.embedder.embed_batch(texts)
            def embed_query(self, text: str) -> List[float]:
                return self.embedder.embed(text)

        embeddings_wrapper = OllamaEmbeddingsWrapper(embedder)
        semantic_splitter = SemanticChunker(
            embeddings=embeddings_wrapper,
            breakpoint_threshold_type="percentile"
        )

        # Combine pages into single text with page markers
        full_text = ""
        page_map = []
        for page_num, text in pages:
            if full_text:
                full_text += "\n\n--- PAGE BREAK ---\n\n"
                page_map.append((len(full_text) - len("\n\n--- PAGE BREAK ---\n\n"), page_num))
            full_text += text
            page_map.append((len(full_text) - len(text), page_num))

        # Split using semantic chunker
        semantic_chunks = semantic_splitter.split_text(full_text)

        # Assign page numbers to chunks
        results = []
        for chunk_text in semantic_chunks:
            chunk_start = full_text.find(chunk_text)
            if chunk_start == -1:
                chunk_start = 0
            # Find closest page boundary
            page_num = 1
            for char_start, p_num in sorted(page_map, key=lambda x: x[0], reverse=True):
                if char_start <= chunk_start:
                    page_num = p_num
                    break
            results.append({
                "text": chunk_text,
                "metadata": {"source": source_filename, "page": page_num, "type": "semantic"}
            })

        return results
    except ImportError:
        print("langchain_experimental not installed. Falling back to basic chunking.")
        return chunk_document(pages, source_filename, chunk_size, overlap)
    except Exception as e:
        print(f"Semantic chunking failed: {e}. Falling back to basic chunking.")
        return chunk_document(pages, source_filename, chunk_size, overlap)

# ---------------------------------------------------------------------------
# New: Keyword-anchored chunking (NGS-specific terms)
# ---------------------------------------------------------------------------

def chunk_document_keyword_anchored(
    pages: List[Tuple[int, str]],
    source_filename: str,
    keywords: List[str] = None,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict]:
    """
    Chunk text anchored on NGS-specific keywords to preserve domain context.

    Args:
        pages: List of (page_num, text) tuples.
        source_filename: Source filename for metadata.
        keywords: Custom keyword list (defaults to NGS_KEYWORDS).
        chunk_size: Maximum chunk size between keywords.
        overlap: Overlap between chunks.

    Returns:
        List of keyword-anchored chunks with metadata.
    """
    if keywords is None:
        keywords = NGS_KEYWORDS

    # Build a single text with page markers
    page_separator = "\n\n--- PAGE BREAK ---\n\n"
    full_text = ""
    page_boundaries = []
    current_pos = 0
    for page_num, text in pages:
        if full_text:
            full_text += page_separator
            current_pos += len(page_separator)
        full_text += text
        page_boundaries.append((current_pos, page_num))
        current_pos += len(text)

    # Find all keyword positions
    keyword_positions = []
    for keyword in keywords:
        start = 0
        while True:
            pos = full_text.lower().find(keyword.lower(), start)
            if pos == -1:
                break
            keyword_positions.append(pos)
            start = pos + 1
    keyword_positions = sorted(set(keyword_positions))

    # Split text at keyword positions
    chunks = []
    last_pos = 0
    for kw_pos in keyword_positions:
        if kw_pos - last_pos > chunk_size:
            gap_text = full_text[last_pos:kw_pos]
            if gap_text.strip():
                gap_pages = [(1, gap_text)]
                gap_chunks = chunk_document(
                    gap_pages, source_filename, chunk_size=chunk_size, overlap=overlap
                )
                for chunk in gap_chunks:
                    chunk_start_in_gap = gap_text.find(chunk["text"])
                    if chunk_start_in_gap == -1:
                        chunk_start_in_gap = 0
                    abs_pos = last_pos + chunk_start_in_gap
                    page_num = 1
                    for pb_pos, pb_page in sorted(page_boundaries, key=lambda x: x[0], reverse=True):
                        if pb_pos <= abs_pos:
                            page_num = pb_page
                            break
                    chunk["metadata"]["page"] = page_num
                    chunk["metadata"]["type"] = "keyword_anchored"
                chunks.extend(gap_chunks)
        last_pos = kw_pos

    # Add remaining text after last keyword
    if last_pos < len(full_text):
        remaining_text = full_text[last_pos:]
        if remaining_text.strip():
            remaining_pages = [(1, remaining_text)]
            remaining_chunks = chunk_document(
                remaining_pages, source_filename, chunk_size=chunk_size, overlap=overlap
            )
            for chunk in remaining_chunks:
                chunk["metadata"]["type"] = "keyword_anchored"
            chunks.extend(remaining_chunks)

    # Fallback to regular chunking if no keywords found
    if not chunks:
        return chunk_document(
            pages, source_filename, chunk_size=chunk_size, overlap=overlap
        )

    return chunks

# ---------------------------------------------------------------------------
# Unified function: chunk with strategy
# ---------------------------------------------------------------------------

def chunk_document_with_strategy(
    pages: List[Tuple[int, str]],
    source_filename: str,
    pdf_path: str = None,
    strategy: ChunkingStrategy = ChunkingStrategy.BASIC,
    chunk_size: int = 500,
    overlap: int = 50,
    **kwargs
) -> List[Dict]:
    """
    Unified chunking function that delegates to the selected strategy.

    Args:
        pages: List of (page_num, text) tuples (for non-table-aware strategies).
        source_filename: Source filename for metadata.
        pdf_path: Path to PDF file (required for TABLE_AWARE strategy).
        strategy: ChunkingStrategy enum value.
        chunk_size: Chunk size.
        overlap: Overlap size.
        **kwargs: Additional args (e.g., embedder_model for SEMANTIC).

    Returns:
        List of chunk dicts with metadata.
    """
    if strategy == ChunkingStrategy.TABLE_AWARE:
        if pdf_path is None:
            raise ValueError("pdf_path is required for TABLE_AWARE strategy")
        return chunk_document_table_aware(pdf_path, source_filename, chunk_size, overlap)
    elif strategy == ChunkingStrategy.SEMANTIC:
        embedder_model = kwargs.get("embedder_model", "haybu/mxbai-embed-large:latest")
        return chunk_document_semantic(pages, source_filename, embedder_model, chunk_size, overlap)
    elif strategy == ChunkingStrategy.KEYWORD_ANCHORED:
        keywords = kwargs.get("keywords", None)
        return chunk_document_keyword_anchored(pages, source_filename, keywords, chunk_size, overlap)
    else:  # BASIC
        return chunk_document(pages, source_filename, chunk_size, overlap)
