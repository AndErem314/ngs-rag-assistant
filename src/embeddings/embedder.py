import ollama
from typing import List


class OllamaEmbedder:
    """
    A client for generating embeddings using a local Ollama server.

    The class connects to an Ollama server and uses a specified model
    to convert text into vector embeddings.

    Attributes:
        model (str): Name of the embedding model (default "nomic-embed-text-v2-moe").
        client (ollama.Client): Ollama client bound to the specified host.
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text-v2-moe"):
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
            response = self.client.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error embedding text: {e}")
            return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts sequentially.

        The Ollama client does not support native batching, so texts are
        processed one at a time. If an embedding fails for a given text,
        an empty list is stored at that position.

        Args:
            texts: List of input text strings.

        Returns:
            A list of embedding vectors in the same order as the input texts.
        """
        return [self.embed(text) for text in texts]


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