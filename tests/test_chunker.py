"""
Unit tests for the ingestion chunker module.

Covers:
  - chunk_document() with empty/single/multiple pages
  - Chunk size and overlap behavior
  - Page number assignment accuracy
  - Metadata correctness (source, page)
  - Edge cases (empty text, special characters)
"""

import pytest
from src.ingestion.chunker import chunk_document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pages(page_texts):
    """Convert list of text strings to (page_num, text) tuples (1-indexed)."""
    return [(i+1, text) for i, text in enumerate(page_texts)]


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestChunkDocument:
    def test_empty_pages_returns_empty_list(self):
        chunks = chunk_document(pages=[], source_filename="test.pdf")
        assert chunks == []

    def test_single_page_chunking(self):
        pages = _make_pages(["This is a single page of NGS protocol text."])
        chunks = chunk_document(pages, source_filename="manual.pdf", chunk_size=50, overlap=10)

        assert len(chunks) >= 1
        assert all(chunk["metadata"]["source"] == "manual.pdf" for chunk in chunks)
        assert all(chunk["metadata"]["page"] == 1 for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)

    def test_multiple_pages_chunking(self):
        pages = _make_pages([
            "Page 1 NGS DNA input requirements.",
            "Page 2 PCR cycling parameters.",
            "Page 3 QC criteria for sequencing."
        ])
        chunks = chunk_document(pages, source_filename="tso500.pdf", chunk_size=100, overlap=20)

        # Should have chunks from multiple pages
        page_numbers = {chunk["metadata"]["page"] for chunk in chunks}
        assert len(page_numbers) >= 2  # At least 2 pages represented

    def test_chunk_size_approximate(self):
        pages = _make_pages(["A" * 200])  # 200 char page
        chunks = chunk_document(pages, source_filename="test.pdf", chunk_size=50, overlap=10)

        # Each chunk should be ~50 chars (± overlap)
        for chunk in chunks:
            assert 40 <= len(chunk["text"]) <= 60  # Approximate due to separators

    def test_overlap_between_chunks(self):
        pages = _make_pages(["ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10])  # Long text
        chunks = chunk_document(pages, source_filename="test.pdf", chunk_size=20, overlap=5)

        # Check overlapping content between consecutive chunks
        for i in range(len(chunks)-1):
            current_end = chunks[i]["text"][-5:]
            next_start = chunks[i+1]["text"][:5]
            assert current_end in chunks[i+1]["text"] or next_start in chunks[i]["text"]

    def test_page_number_assignment_accuracy(self):
        pages = _make_pages([
            "Unique page 1 content: DNA input 100ng. " * 10,
            "Unique page 2 content: PCR 30 cycles. " * 10,
            "Unique page 3 content: QC > Q30. " * 10
        ])
        chunks = chunk_document(pages, source_filename="tso.pdf", chunk_size=100, overlap=20)

        # Chunks from page 1 should have page=1, etc.
        page1_chunks = [c for c in chunks if c["metadata"]["page"] == 1]
        page2_chunks = [c for c in chunks if c["metadata"]["page"] == 2]
        page3_chunks = [c for c in chunks if c["metadata"]["page"] == 3]

        assert len(page1_chunks) >= 1
        assert len(page2_chunks) >= 1
        assert len(page3_chunks) >= 1

        # Content should match page
        assert "DNA input" in page1_chunks[0]["text"]
        # Page 2 chunks should contain PCR (check all since separator chunks may lack it)
        assert any("PCR" in c["text"] for c in page2_chunks)
        assert any("QC" in c["text"] for c in page3_chunks)

    def test_metadata_contains_source_and_page(self):
        pages = _make_pages(["Test content"])
        chunks = chunk_document(pages, source_filename="protocol.pdf")

        for chunk in chunks:
            assert "source" in chunk["metadata"]
            assert "page" in chunk["metadata"]
            assert chunk["metadata"]["source"] == "protocol.pdf"
            assert isinstance(chunk["metadata"]["page"], int)

    def test_large_text_splitting(self):
        pages = _make_pages(["X" * 1000])  # 1000 char page
        chunks = chunk_document(pages, source_filename="big.pdf", chunk_size=200, overlap=50)

        assert len(chunks) >= 4  # 1000 chars / 200 chunk size = ~5 chunks
        assert all(len(chunk["text"]) <= 250 for chunk in chunks)  # ~200 + overlap

    def test_special_characters_handling(self):
        pages = _make_pages(["NGS protocol: 37°C, 30 min, 100 µL, 2.5 ng/µL"])
        chunks = chunk_document(pages, source_filename="special.pdf")

        assert len(chunks) >= 1
        assert "°C" in chunks[0]["text"]
        assert "µL" in chunks[0]["text"]
