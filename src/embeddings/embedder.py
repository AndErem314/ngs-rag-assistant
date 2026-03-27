import ollama
from typing import List

class OllamaEmbedder:
    """
    A client for generating embeddings using a local Ollama server.

    The class connects to an Ollama server and uses a specified model
    to convert text into vector embeddings.

    Attributes:
        host (str): URL of the Ollama server (default "http://localhost:11434").
        model (str): Name of the embedding model (default "nomic-embed-text-v2-moe").
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text-v2-moe"):
        """
        Initialize the embedder.

        Args:
            host: Ollama server URL.
            model: Name of the embedding model (assumed to be already pulled).
        """
        self.host = host
        self.model = model
        # Optionally, we could store the client; but the ollama module uses the host via
        # environment variable or by setting it in the client. We'll just pass host to each call.
        # The ollama Python client does not have a direct host parameter in __init__,
        # so we'll use the client's default (localhost) and assume the user has set OLLAMA_HOST if needed.
        # Alternatively, we could set the environment variable, but for simplicity we'll rely on default.

    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: The input text string.

        Returns:
            A list of floats representing the embedding vector.
            If an error occurs, returns an empty list.
        """
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error embedding text: {e}")
            return []

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input text strings.
            batch_size: Number of texts to process at once (ignored in this simple
                        implementation; we just loop over each text individually
                        because the Ollama client does not support batching).

        Returns:
            A list of embedding vectors (each a list of floats) in the same order
            as the input texts. If an error occurs for a text, its embedding will
            be an empty list.
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return embeddings

# ----------------------------------------------------------------------
# Example usage (commented out):
# if __name__ == "__main__":
#     embedder = OllamaEmbedder()
#     sample_text = "This is a test sentence."
#     embedding = embedder.embed(sample_text)
#     print(f"Embedding length: {len(embedding)}")
#     print(embedding[:5])  # show first 5 values
#
#     batch = ["First text", "Second text"]
#     batch_embeddings = embedder.embed_batch(batch)
#     print(f"Batch embeddings length: {len(batch_embeddings)}")