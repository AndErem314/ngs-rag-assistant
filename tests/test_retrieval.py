"""
Smoke tests for the retrieval pipeline.

Covers:
  - VectorStore.search() — happy path, source_filter, max_distance
  - retrieve_context() — happy path, empty embedding, no results,
    max_distance filtering, metadata enrichment with distance field

No Ollama server is required. OllamaEmbedder is mocked so embed() returns
a controlled vector, and ChromaDB runs in-memory via EphemeralClient.
"""

import pytest
import chromadb
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_store(collection_name: str = "test_retrieval"):
    """VectorStore backed by an in-memory ChromaDB — no disk I/O."""
    ephemeral = chromadb.EphemeralClient()
    with patch("src.retrieval.vector_store.chromadb.PersistentClient", return_value=ephemeral):
        from src.retrieval.vector_store import VectorStore
        store = VectorStore(collection_name=collection_name, persist_directory="/tmp/unused")
    return store


def _populated_store():
    """Return a store pre-loaded with three chunks across two sources."""
    store = _make_store()

    chunks = [
        {"text": "Minimum DNA input is 100 ng.",        "metadata": {"source": "a.pdf", "page": 1}},
        {"text": "Incubate at 37 °C for 30 minutes.",   "metadata": {"source": "a.pdf", "page": 3}},
        {"text": "Store reagents at -20 °C.",            "metadata": {"source": "b.pdf", "page": 2}},
    ]
    # Use orthogonal unit vectors so distances between them are meaningful
    # and predictable for assertions.
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    store.add_chunks(chunks, embeddings)
    return store


def _mock_embedder(vector: list) -> MagicMock:
    """Return a mock OllamaEmbedder whose embed() always returns ``vector``."""
    embedder = MagicMock()
    embedder.embed.return_value = vector
    return embedder


# ---------------------------------------------------------------------------
# VectorStore.search
# ---------------------------------------------------------------------------

class TestVectorStoreSearch:
    def setup_method(self):
        self.store = _populated_store()

    def test_search_returns_top_k_results(self):
        query = [1.0, 0.0, 0.0, 0.0]   # closest to chunk 0
        results = self.store.search(query_embedding=query, top_k=2)
        assert len(results) == 2

    def test_search_result_structure(self):
        query = [1.0, 0.0, 0.0, 0.0]
        results = self.store.search(query_embedding=query, top_k=1)
        doc, meta, distance = results[0]
        assert isinstance(doc, str)
        assert "source" in meta and "page" in meta
        assert isinstance(distance, float)

    def test_search_closest_chunk_is_first(self):
        """Querying with chunk 0's vector should return chunk 0 first."""
        query = [1.0, 0.0, 0.0, 0.0]
        results = self.store.search(query_embedding=query, top_k=3)
        assert "100 ng" in results[0][0]

    def test_source_filter_restricts_results(self):
        query = [1.0, 0.0, 0.0, 0.0]
        results = self.store.search(query_embedding=query, top_k=3, source_filter=["b.pdf"])
        assert all(meta["source"] == "b.pdf" for _, meta, _ in results)

    def test_max_distance_filters_low_quality_chunks(self):
        """A very tight distance threshold should drop all but the nearest chunk."""
        query = [1.0, 0.0, 0.0, 0.0]
        # Orthogonal vectors have cosine distance = 1.0; use 0.5 to keep only
        # the identical-direction chunk (distance ≈ 0).
        results = self.store.search(query_embedding=query, top_k=3, max_distance=0.5)
        assert len(results) == 1
        assert "100 ng" in results[0][0]

    def test_max_distance_none_returns_all_top_k(self):
        query = [1.0, 0.0, 0.0, 0.0]
        results = self.store.search(query_embedding=query, top_k=3, max_distance=None)
        assert len(results) == 3

    def test_search_empty_collection_returns_empty_list(self):
        empty_store = _make_store("empty")
        results = empty_store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# retrieve_context
# ---------------------------------------------------------------------------

class TestRetrieveContext:
    def setup_method(self):
        self.store = _populated_store()

    def _call(self, vector, **kwargs):
        from src.retrieval.query_processor import retrieve_context
        embedder = _mock_embedder(vector)
        return retrieve_context(
            question="test question",
            embedder=embedder,
            vector_store=self.store,
            **kwargs,
        )

    def test_happy_path_returns_non_empty_context(self):
        context, metadata = self._call([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert isinstance(context, str) and len(context) > 0
        assert isinstance(metadata, list) and len(metadata) == 2

    def test_metadata_contains_required_keys(self):
        _, metadata = self._call([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert "source" in metadata[0]
        assert "page" in metadata[0]
        assert "distance" in metadata[0]

    def test_metadata_distance_is_rounded_float(self):
        _, metadata = self._call([1.0, 0.0, 0.0, 0.0], top_k=1)
        d = metadata[0]["distance"]
        assert isinstance(d, float)
        # rounded to 4 decimal places → string representation ≤ 6 chars
        assert len(str(d).split(".")[-1]) <= 4

    def test_context_chunks_joined_by_double_newline(self):
        context, _ = self._call([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert "\n\n" in context

    def test_empty_embedding_returns_empty_tuple(self):
        from src.retrieval.query_processor import retrieve_context
        embedder = _mock_embedder([])           # simulate Ollama failure
        context, metadata = retrieve_context(
            question="anything",
            embedder=embedder,
            vector_store=self.store,
        )
        assert context == ""
        assert metadata == []

    def test_max_distance_filters_poor_results(self):
        """A very tight threshold should reduce the returned chunks."""
        # All chunks at top_k=3 but only the nearest should survive dist < 0.5
        context_all, meta_all = self._call([1.0, 0.0, 0.0, 0.0], top_k=3, max_distance=None)
        context_filt, meta_filt = self._call([1.0, 0.0, 0.0, 0.0], top_k=3, max_distance=0.5)
        assert len(meta_filt) < len(meta_all)

    def test_source_filter_restricts_context(self):
        context, metadata = self._call(
            [1.0, 0.0, 0.0, 0.0],
            top_k=3,
            source_filter=["b.pdf"],
        )
        assert all(m["source"] == "b.pdf" for m in metadata)

    def test_no_results_after_filter_returns_empty_tuple(self):
        """Querying with max_distance=0.0 should drop everything."""
        context, metadata = self._call([1.0, 0.0, 0.0, 0.0], top_k=3, max_distance=0.0)
        assert context == ""
        assert metadata == []