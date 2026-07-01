"""
Unit tests for the embeddings embedder module.

Covers:
  - OllamaEmbedder initialization (host, model, client setup)
  - embed() single text embedding
  - embed_batch() multiple texts (native batching)
  - Error handling (Ollama unavailable, invalid response)
  - Edge cases (empty text, long text)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.embeddings.embedder import OllamaEmbedder


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestOllamaEmbedderInit:
    def test_default_initialization(self):
        with patch('src.embeddings.embedder.ollama.Client') as mock_client:
            embedder = OllamaEmbedder()
            assert embedder.model == "qwen3-embedding:0.6b"
            mock_client.assert_called_once_with(host="http://localhost:11434")

    def test_custom_host_initialization(self):
        with patch('src.embeddings.embedder.ollama.Client') as mock_client:
            embedder = OllamaEmbedder(host="http://192.168.1.100:11434")
            mock_client.assert_called_once_with(host="http://192.168.1.100:11434")

    def test_custom_model_initialization(self):
        with patch('src.embeddings.embedder.ollama.Client'):
            embedder = OllamaEmbedder(model="qwen3-embedding:0.6b")
            assert embedder.model == "qwen3-embedding:0.6b"

    def test_client_initialization(self):
        mock_client = Mock()
        with patch('src.embeddings.embedder.ollama.Client', return_value=mock_client):
            embedder = OllamaEmbedder()
            assert embedder.client == mock_client


class TestEmbedMethod:
    def test_embed_returns_correct_vector(self):
        mock_client = Mock()
        mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]]}
        
        with patch('src.embeddings.embedder.ollama.Client', return_value=mock_client):
            embedder = OllamaEmbedder()
            result = embedder.embed("test text")
            
            assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_client.embed.assert_called_once_with(
                model="qwen3-embedding:0.6b",
                input="test text"
            )

    def test_embed_empty_text(self):
        mock_client = Mock()
        mock_client.embed.return_value = {"embeddings": [[]]}
        
        with patch('src.embeddings.embedder.ollama.Client', return_value=mock_client):
            embedder = OllamaEmbedder()
            result = embedder.embed("")
            assert result == []

    def test_embed_ngs_specific_text(self):
        """Test embedding NGS protocol text."""
        mock_client = Mock()
        mock_client.embed.return_value = {"embeddings": [[0.9] * 1024]}
        
        with patch('src.embeddings.embedder.ollama.Client', return_value=mock_client):
            embedder = OllamaEmbedder()
            ngs_text = "DNA input: 100 ng, PCR cycles: 30, QC: >80% Q30"
            result = embedder.embed(ngs_text)
            
            assert len(result) == 1024
            mock_client.embed.assert_called_once_with(
                model="qwen3-embedding:0.6b",
                input=ngs_text
            )

    def test_embed_handles_exception(self):
        mock_client = Mock()
        mock_client.embed.side_effect = Exception("Ollama connection failed")
        
        with patch('src.embeddings.embedder.ollama.Client', return_value=mock_client):
            embedder = OllamaEmbedder()
            result = embedder.embed("test text")
            assert result == []

    def test_embed_uses_correct_model(self):
        mock_client = Mock()
        mock_client.embed.return_value = {"embeddings": [[0.1, 0.2]]}
        
        with patch('src.embeddings.embedder.ollama.Client', return_value=mock_client):
            embedder = OllamaEmbedder(model="qwen3-embedding:0.6b")
            embedder.embed("test")
            mock_client.embed.assert_called_once_with(
                model="qwen3-embedding:0.6b",
                input="test"
            )


class TestEmbedBatchMethod:
    def test_embed_batch_returns_correct_vectors(self):
        mock_client = Mock()
        mock_client.embed.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        }
        
        with patch('src.embeddings.embedder.ollama.Client', return_value=mock_client):
            embedder = OllamaEmbedder()
            result = embedder.embed_batch(["text1", "text2", "text3"])
            
            assert len(result) == 3
            assert result[0] == [0.1, 0.2]
            assert result[1] == [0.3, 0.4]
            assert result[2] == [0.5, 0.6]
            # Native batching: one call with array input
            mock_client.embed.assert_called_once_with(
                model="qwen3-embedding:0.6b",
                input=["text1", "text2", "text3"]
            )

    def test_embed_batch_empty_list(self):
        with patch('src.embeddings.embedder.ollama.Client'):
            embedder = OllamaEmbedder()
            result = embedder.embed_batch([])
            assert result == []

    def test_embed_batch_with_api_error(self):
        mock_client = Mock()
        mock_client.embed.side_effect = Exception("Batch embedding failed")
        
        with patch('src.embeddings.embedder.ollama.Client', return_value=mock_client):
            embedder = OllamaEmbedder()
            result = embedder.embed_batch(["text1", "text2", "text3"])
            
            assert len(result) == 3
            assert result == [[], [], []]  # All return empty on batch failure

    def test_embed_batch_single_text(self):
        mock_client = Mock()
        mock_client.embed.return_value = {"embeddings": [[0.7, 0.8, 0.9]]}
        
        with patch('src.embeddings.embedder.ollama.Client', return_value=mock_client):
            embedder = OllamaEmbedder()
            result = embedder.embed_batch(["single text"])
            
            assert len(result) == 1
            assert result[0] == [0.7, 0.8, 0.9]
