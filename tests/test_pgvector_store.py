"""
Integration tests for PgvectorStore (secondary pgvector backend).

Note: Requires pgvector running (use `docker-compose up -d` to start).
Tests assume default connection: postgresql://ngs_user:***@localhost:5432/ngs_rag
These tests are automatically skipped if PostgreSQL is unavailable.
"""

import pytest
import json
from src.retrieval.pgvector_store import PgvectorStore


def _pgvector_available() -> bool:
    """Check if PostgreSQL with pgvector is reachable."""
    import psycopg2
    try:
        conn = psycopg2.connect("postgresql://ngs_user:***@localhost:5432/ngs_rag")
        conn.close()
        return True
    except Exception:
        return False


pgvector_available = pytest.mark.skipif(
    not _pgvector_available(),
    reason="PostgreSQL with pgvector not available (start with: docker-compose up -d)"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store():
    """Initialize PgvectorStore with test collection (skipped if PostgreSQL unavailable)."""
    if not _pgvector_available():
        pytest.skip("PostgreSQL with pgvector not available (start with: docker-compose up -d)")
    test_store = PgvectorStore(
        db_url="postgresql://ngs_user:ngs_secure_password@localhost:5432/ngs_rag",
        collection_name="test_pgvector",
        embedding_dim=1024
    )
    # Clear any existing data
    test_store.clear_collection()
    yield test_store
    # Cleanup after tests
    test_store.clear_collection()


@pytest.fixture(scope="module")
def sample_chunks():
    return [
        {"text": "DNA input: 100 ng.", "metadata": {"source": "manual.pdf", "page": 5}},
        {"text": "PCR cycles: 30.", "metadata": {"source": "manual.pdf", "page": 8}},
        {"text": "QC criteria: >80% Q30.", "metadata": {"source": "tso500.pdf", "page": 12}}
    ]


@pytest.fixture(scope="module")
def sample_embeddings():
    # Dummy 1024-dimensional embeddings (matching mxbai-embed-large-latest)
    return [
        [0.1] * 1024,
        [0.2] * 1024,
        [0.3] * 1024
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pgvector_available
class TestPgvectorStoreInit:
    def test_initialization(self):
        store = PgvectorStore(collection_name="init_test")
        assert store.collection_name == "init_test"
        assert store.embedding_dim == 1024
        store.clear_collection()


class TestMakeChunkId:
    def test_same_inputs_produce_same_id(self):
        id1 = PgvectorStore._make_chunk_id("doc.pdf", 1, "Some text")
        id2 = PgvectorStore._make_chunk_id("doc.pdf", 1, "Some text")
        assert id1 == id2
        assert len(id1) == 16

    def test_different_source_produces_different_id(self):
        id1 = PgvectorStore._make_chunk_id("doc_a.pdf", 1, "Text")
        id2 = PgvectorStore._make_chunk_id("doc_b.pdf", 1, "Text")
        assert id1 != id2

    def test_different_page_produces_different_id(self):
        id1 = PgvectorStore._make_chunk_id("doc.pdf", 1, "Text")
        id2 = PgvectorStore._make_chunk_id("doc.pdf", 2, "Text")
        assert id1 != id2


@pgvector_available
class TestAddChunks:
    def test_add_chunks_success(self, store, sample_chunks, sample_embeddings):
        store.add_chunks(sample_chunks, sample_embeddings)
        # If no error, consider it success
        assert True

    def test_add_chunks_mismatched_lengths(self, store):
        chunks = [{"text": "test", "metadata": {"source": "a.pdf", "page": 1}}]
        embeddings = [[0.1]*1024, [0.2]*1024]  # 2 embeddings for 1 chunk
        with pytest.raises(ValueError, match="must match"):
            store.add_chunks(chunks, embeddings)

    def test_upsert_deduplication(self, store, sample_chunks, sample_embeddings):
        # Add same chunks twice
        store.add_chunks(sample_chunks, sample_embeddings)
        store.add_chunks(sample_chunks, sample_embeddings)
        # Search should return only 3 unique chunks
        results = store.search(query_embedding=[0.1]*1024, top_k=10)
        assert len(results) == 3


@pgvector_available
class TestSearch:
    def test_search_returns_results(self, store, sample_chunks, sample_embeddings):
        store.add_chunks(sample_chunks, sample_embeddings)
        results = store.search(query_embedding=[0.1]*1024, top_k=2)
        assert len(results) <= 2
        assert all(len(item) == 3 for item in results)  # (content, metadata, distance)

    def test_search_with_source_filter(self, store, sample_chunks, sample_embeddings):
        store.add_chunks(sample_chunks, sample_embeddings)
        # Filter for manual.pdf only
        results = store.search(
            query_embedding=[0.1]*1024,
            top_k=10,
            source_filter=["manual.pdf"]
        )
        assert all(meta["source"] == "manual.pdf" for _, meta, _ in results)

    def test_search_with_max_distance(self, store, sample_chunks, sample_embeddings):
        store.add_chunks(sample_chunks, sample_embeddings)
        # Distance should be <= max_distance
        results = store.search(
            query_embedding=[0.1]*1024,
            top_k=10,
            max_distance=0.5
        )
        assert all(dist <= 0.5 for _, _, dist in results)

    def test_search_ngs_content(self, store):
        ngs_chunks = [
            {"text": "TruSight Oncology 500: 500 genes.", "metadata": {"source": "tso.pdf", "page": 1}}
        ]
        ngs_embeddings = [[0.15]*1024]
        store.add_chunks(ngs_chunks, ngs_embeddings)
        results = store.search(query_embedding=[0.15]*1024, top_k=1)
        assert len(results) == 1
        assert "TruSight Oncology 500" in results[0][0]


@pgvector_available
class TestClearCollection:
    def test_clear_removes_all_records(self, store, sample_chunks, sample_embeddings):
        store.add_chunks(sample_chunks, sample_embeddings)
        assert len(store.search(query_embedding=[0.1]*1024, top_k=10)) == 3
        store.clear_collection()
        assert len(store.search(query_embedding=[0.1]*1024, top_k=10)) == 0
