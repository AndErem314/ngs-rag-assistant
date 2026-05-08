"""
Adversarial Testing for NGS RAG Pipeline

Tests system robustness against:
1. Malformed NGS queries (incomplete gene names, rare variants)
2. Edge cases (empty queries, extremely long queries, special characters)
3. Conflicting table data queries
4. SQL injection attempts (security)

Usage:
    pytest tests/test_adversarial.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embeddings.embedder import OllamaEmbedder
from src.retrieval.vector_store import VectorStore


class TestAdversarialQueries:
    """Test RAG pipeline robustness against adversarial inputs."""

    @pytest.fixture
    def embedder(self):
        """Initialize Ollama embedder."""
        return OllamaEmbedder(
            host="http://localhost:11434",
            model="haybu/mxbai-embed-large-latest:latest"
        )

    @pytest.fixture
    def vector_store(self):
        """Initialize a temporary vector store."""
        store = VectorStore(
            collection_name="test_adversarial",
            persist_directory="./chroma_test_adversarial"
        )
        store.clear_collection()
        yield store
        store.clear_collection()

    # --- Malformed NGS Queries ---

    @pytest.mark.parametrize("malformed_query", [
        "",  # Empty query
        "   ",  # Whitespace only
        "DN",  # Incomplete gene prefix
        "PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR PCR",  # Repeated term
        "What is the protocol for preparing DNA???!!!",  # Excessive punctuation
        "DNA\x00injection",  # Null byte injection
        "' OR '1'='1",  # SQL injection attempt
        "<script>alert('xss')</script>",  # XSS attempt
        "a" * 10000,  # Extremely long query (10k chars)
        "PCR\x01\x02\x03injection",  # Control characters
    ])
    def test_malformed_queries_handle_gracefully(self, embedder, vector_store, malformed_query):
        """Test that malformed queries don't crash the system."""
        # Should not raise an exception
        embedding = embedder.embed(malformed_query)
        
        # Empty/whitespace queries may return empty embedding (graceful handling)
        if malformed_query.strip() == "":
            assert embedding == [] or embedding is not None
        else:
            # Other queries should either return valid embedding or empty list (not crash)
            assert embedding is not None
            if embedding:  # If we got an embedding
                assert isinstance(embedding, list)
                assert len(embedding) > 0

    def test_unicode_special_characters(self, embedder):
        """Test queries with unicode/emoji/special chars."""
        unicode_queries = [
            "DNA 🧬 extraction protocol",
            "PCR — dash test",
            "RNA 'quote' test",
            "Gene name: α-actin",
            "Protocol for 测序 (sequencing)",
        ]
        for query in unicode_queries:
            embedding = embedder.embed(query)
            # Should not crash; may return empty or valid embedding
            assert embedding is not None

    def test_ngs_edge_cases(self, embedder):
        """Test NGS-specific edge case queries."""
        edge_cases = [
            "TruSight Oncology 500",  # Exact product name
            "TSO500",  # Abbreviation
            "Illumina TruSight-Oncology-500-v2",  # Full name with version
            "v2.0.1",  # Version number only
            "Figure 3-1",  # Figure reference
            "Table 5.2",  # Table reference
            "Page 42",  # Page reference
            "Supplementary Table S1",  # Supplementary reference
        ]
        for query in edge_cases:
            embedding = embedder.embed(query)
            assert embedding is not None

    def test_conflicting_terms(self, embedder):
        """Test queries with conflicting or ambiguous terms."""
        conflicting = [
            "DNA but not DNA",
            "PCR vs PCR",
            "TruSight and not TruSight",
            "protocol NOT protocol",
        ]
        for query in conflicting:
            embedding = embedder.embed(query)
            assert embedding is not None

    def test_vector_store_with_empty_results(self, embedder, vector_store):
        """Test vector store search with no matching results."""
        # Don't add any documents
        query_emb = embedder.embed("DNA extraction")
        if not query_emb:
            pytest.skip("Embedder not available")

        results = vector_store.search(query_embedding=query_emb, top_k=5)
        assert results == []  # Should return empty list, not crash

    def test_vector_store_with_malformed_embedding(self, vector_store):
        """Test vector store with invalid embedding dimensions."""
        # Try searching with wrong-dimension embedding
        wrong_emb = [0.1, 0.2, 0.3]  # 3-dim instead of 1024
        results = vector_store.search(query_embedding=wrong_emb, top_k=5)
        # Should handle gracefully (return empty or raise clear error)
        assert isinstance(results, list)


class TestRetrievalRobustness:
    """Test retrieval pipeline robustness with adversarial inputs."""

    @pytest.fixture
    def setup_pipeline(self):
        """Set up a minimal pipeline with test data."""
        embedder = OllamaEmbedder(
            host="http://localhost:11434",
            model="haybu/mxbai-embed-large-latest:latest"
        )
        store = VectorStore(
            collection_name="test_robustness",
            persist_directory="./chroma_test_robustness"
        )
        store.clear_collection()

        # Add minimal test chunks
        test_chunks = [
            {"text": "DNA extraction protocol", "metadata": {"page": 1, "source": "test.pdf"}},
            {"text": "PCR amplification steps", "metadata": {"page": 2, "source": "test.pdf"}},
        ]
        embeddings = embedder.embed_batch([c["text"] for c in test_chunks])
        valid = [(c, e) for c, e in zip(test_chunks, embeddings) if e]
        if valid:
            chunks, embs = zip(*valid)
            store.add_chunks(list(chunks), list(embs))

        yield {"embedder": embedder, "store": store}
        store.clear_collection()

    def test_hybrid_search_with_special_chars(self, setup_pipeline):
        """Test hybrid search handles special characters in query text."""
        pipeline = setup_pipeline
        embedder = pipeline["embedder"]
        store = pipeline["store"]

        query = "DNA & RNA (extraction) [protocol]: how?"
        query_emb = embedder.embed(query)
        if not query_emb:
            pytest.skip("Embedder not available")

        # Should not crash with special chars in query_text
        results = store.search(
            query_embedding=query_emb,
            top_k=5,
            hybrid=True,
            query_text=query
        )
        assert isinstance(results, list)

    def test_max_distance_filter(self, setup_pipeline):
        """Test that max_distance parameter filters results correctly."""
        pipeline = setup_pipeline
        embedder = pipeline["embedder"]
        store = pipeline["store"]

        query_emb = embedder.embed("DNA")
        if not query_emb:
            pytest.skip("Embedder not available")

        # With very strict distance (should return few/none)
        results_strict = store.search(query_embedding=query_emb, top_k=5, max_distance=0.01)
        # With loose distance (should return more)
        results_loose = store.search(query_embedding=query_emb, top_k=5, max_distance=1.0)

        assert isinstance(results_strict, list)
        assert isinstance(results_loose, list)
        # Loose should return at least as many as strict
        assert len(results_loose) >= len(results_strict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
