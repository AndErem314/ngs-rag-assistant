import ollama
from typing import List


class OllamaEmbedder:
    """
    A client for generating embeddings using a local Ollama server.

    The class connects to an Ollama server and uses a specified model
    to convert text into vector embeddings.

    Attributes:
        model (str): Name of the embedding model (default "bge-m3:latest").
        client (ollama.Client): Ollama client bound to the specified host.
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "bge-m3:latest"):
        """
        Initialize the embedder.

        Args:
            host: Ollama server URL. Passed directly to ollama.Client so it
                  actually takes effect (unlike relying on the OLLAMA_HOST env var).
            model: Name of the embedding model (assumed to be already pulled).
        """
        self.model = model
        self.client = ollama.Client(host=host)

    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: The input text string.

        Returns:
            A list of floats representing the embedding vector.
            Returns an empty list if an error occurs.
        """
        try:
            response = self.client.embed(model=self.model, input=text)
            return response["embeddings"][0]
        except Exception as e:
            print(f"Error embedding text: {e}")
            return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using native Ollama batching.

        The Ollama /api/embed endpoint supports batched input natively,
        so all texts are sent in a single request for efficiency.

        Args:
            texts: List of input text strings.

        Returns:
            A list of embedding vectors in the same order as the input texts.
        """
        if not texts:
            return []
        try:
            response = self.client.embed(model=self.model, input=texts)
            return response["embeddings"]
        except Exception as e:
            print(f"Error embedding batch: {e}")
            return [[] for _ in texts]


# ----------------------------------------------------------------------
# Example usage (commented out):
# if __name__ == "__main__":
#     embedder = OllamaEmbedder(host="http://localhost:11434")
#
#     sample_text = "This is a test sentence."
#     embedding = embedder.embed(sample_text)
#     print(f"Embedding length: {len(embedding)}")
#     print(embedding[:5])
#
#     batch = ["First text", "Second text", "Third text"]
#     batch_embeddings = embedder.embed_batch(batch)
#     print(f"Batch size: {len(batch_embeddings)}")