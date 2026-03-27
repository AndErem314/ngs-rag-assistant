"""
Smoke tests for the ingestion pipeline.

Covers:
  - VectorStore.add_chunks() — happy path
  - VectorStore._make_chunk_id() — stability and collision resistance
  - Re-ingestion safety (upsert semantics: same ID → update, no duplicate)
  - Mismatched chunks/embeddings raises ValueError
  - VectorStore.clear_collection()

All tests use an in-memory ChromaDB client (EphemeralClient) so no files
are written to disk and no Ollama server is required.
"""

import pytest
import chromadb
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(collection_name: str = "test"):
    """Return a VectorStore backed by an in-memory ChromaDB client."""
    # We patch chromadb.PersistentClient so VectorStore never touches disk.
    ephemeral = chromadb.EphemeralClient()

    with patch("src.retrieval.vector_store.chromadb.PersistentClient", return_value=ephemeral):
        from src.retrieval.vector_store import VectorStore
        store = VectorStore(collection_name=collection_name, persist_directory="/tmp/unused")

    return store


def _sample_chunks():
    return [
        {
            "text": "The minimum DNA input is 100 ng.",
            "metadata": {"source": "manual_a.pdf", "page": 5},
        },
        {
            "text": "Incubate at 37 °C for 30 minutes.",
            "metadata": {"source": "manual_a.pdf", "page": 8},
        },
    ]


def _dummy_embeddings(n: int = 2, dim: int = 4) -> list:
    """Tiny deterministic embeddings — sufficient for ChromaDB smoke tests."""
    return [[float(i + j * 0.1) for j in range(dim)] for i in range(n)]


# ---------------------------------------------------------------------------
# _make_chunk_id
# ---------------------------------------------------------------------------

class TestMakeChunkId:
    def setup_method(self):
        from src.retrieval.vector_store import VectorStore
        self.make_id = VectorStore._make_chunk_id

    def test_same_inputs_produce_same_id(self):
        id1 = self.make_id("doc.pdf", 1, "Some text here")
        id2 = self.make_id("doc.pdf", 1, "Some text here")
        assert id1 == id2, "ID must be deterministic for identical inputs"

    def test_different_source_produces_different_id(self):
        id1 = self.make_id("doc_a.pdf", 1, "Same text")
        id2 = self.make_id("doc_b.pdf", 1, "Same text")
        assert id1 != id2

    def test_different_page_produces_different_id(self):
        id1 = self.make_id("doc.pdf", 1, "Same text")
        id2 = self.make_id("doc.pdf", 2, "Same text")
        assert id1 != id2

    def test_different_text_produces_different_id(self):
        id1 = self.make_id("doc.pdf", 1, "Text A")
        id2 = self.make_id("doc.pdf", 1, "Text B")
        assert id1 != id2

    def test_id_is_16_hex_chars(self):
        chunk_id = self.make_id("doc.pdf", 3, "Hello world")
        assert len(chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk_id)


# ---------------------------------------------------------------------------
# add_chunks
# ---------------------------------------------------------------------------

class TestAddChunks:
    def setup_method(self):
        self.store = _make_store()

    def test_add_chunks_happy_path(self):
        chunks = _sample_chunks()
        embeddings = _dummy_embeddings(len(chunks))
        # Should not raise
        self.store.add_chunks(chunks, embeddings)
        result = self.store.collection.get()
        assert len(result["ids"]) == len(chunks)

    def test_add_chunks_ids_are_deterministic(self):
        from src.retrieval.vector_store import VectorStore
        chunks = _sample_chunks()
        embeddings = _dummy_embeddings(len(chunks))
        self.store.add_chunks(chunks, embeddings)
        stored_ids = set(self.store.collection.get()["ids"])

        expected_ids = {
            VectorStore._make_chunk_id(c["metadata"]["source"], c["metadata"]["page"], c["text"])
            for c in chunks
        }
        assert stored_ids == expected_ids

    def test_mismatched_lengths_raise_value_error(self):
        chunks = _sample_chunks()
        bad_embeddings = _dummy_embeddings(1)           # one fewer than chunks
        with pytest.raises(ValueError, match="must match"):
            self.store.add_chunks(chunks, bad_embeddings)

    def test_reingest_same_chunks_does_not_duplicate(self):
        """Re-ingesting identical chunks must upsert, not create duplicates."""
        chunks = _sample_chunks()
        embeddings = _dummy_embeddings(len(chunks))

        self.store.add_chunks(chunks, embeddings)
        self.store.add_chunks(chunks, embeddings)   # second ingest

        result = self.store.collection.get()
        assert len(result["ids"]) == len(chunks), (
            "Re-ingesting the same chunks must not create duplicate records"
        )

    def test_reingest_different_chunks_same_source_adds_new_records(self):
        """New chunks from the same source get new IDs and are added alongside old ones."""
        chunk_a = [{"text": "DNA input: 100 ng.", "metadata": {"source": "doc.pdf", "page": 1}}]
        chunk_b = [{"text": "RNA input: 200 ng.", "metadata": {"source": "doc.pdf", "page": 2}}]

        self.store.add_chunks(chunk_a, _dummy_embeddings(1))
        self.store.add_chunks(chunk_b, _dummy_embeddings(1))

        result = self.store.collection.get()
        assert len(result["ids"]) == 2


# ---------------------------------------------------------------------------
# clear_collection
# ---------------------------------------------------------------------------

class TestClearCollection:
    def setup_method(self):
        self.store = _make_store()

    def test_clear_removes_all_records(self):
        chunks = _sample_chunks()
        self.store.add_chunks(chunks, _dummy_embeddings(len(chunks)))
        assert len(self.store.collection.get()["ids"]) == len(chunks)

        self.store.clear_collection()
        assert self.store.collection.get()["ids"] == []

    def test_clear_on_empty_collection_does_not_raise(self):
        # Should be a no-op, not an error
        self.store.clear_collection()
        assert self.store.collection.get()["ids"] == []