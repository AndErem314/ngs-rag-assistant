import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple, Optional

class VectorStore:
    """
    A persistent vector store using ChromaDB. Stores chunks with embeddings and metadata.
    Supports filtering by source file.
    """

    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the Chroma collection.
            persist_directory: Directory where ChromaDB persists data.
        """
        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """
        Add chunks and their embeddings to the vector store.

        Args:
            chunks: List of dicts, each containing "text" and "metadata" (with at least "source" and "page").
            embeddings: List of embedding vectors, corresponding to each chunk.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        ids = []
        metadatas = []
        documents = []
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"]["source"]
            page = chunk["metadata"]["page"]
            # Generate unique ID
            ids.append(f"{source}_{i}")
            metadatas.append({"source": source, "page": page})
            documents.append(chunk["text"])

        try:
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
        except Exception as e:
            print(f"Error adding chunks: {e}")

    def search(self, query_embedding: List[float], top_k: int = 5, source_filter: Optional[List[str]] = None) -> List[Tuple[str, Dict, float]]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            source_filter: Optional list of source filenames to restrict search to.

        Returns:
            List of tuples: (document_text, metadata, similarity_score).
            Similarity score is distance (lower is better). Chroma returns distances.
        """
        where = None
        if source_filter:
            where = {"source": {"$in": source_filter}}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            # results is a dict: keys: 'ids', 'documents', 'metadatas', 'distances', each a list of lists.
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            return list(zip(docs, metas, distances))
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def clear_collection(self) -> None:
        """Delete all documents from the collection."""
        try:
            # Get all IDs and delete them
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
        except Exception as e:
            print(f"Error clearing collection: {e}")

# ----------------------------------------------------------------------
# Example usage (commented out):
# if __name__ == "__main__":
#     vs = VectorStore("test_collection")
#
#     # Add dummy data
#     chunks = [
#         {"text": "This is a test document about DNA extraction.", "metadata": {"source": "test1.pdf", "page": 1}},
#         {"text": "Another document about shearing conditions.", "metadata": {"source": "test2.pdf", "page": 2}}
#     ]
#     # Dummy embeddings (e.g., 3‑dim for demonstration)
#     embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
#     vs.add_chunks(chunks, embeddings)
#
#     # Search with filter
#     query_emb = [0.1, 0.2, 0.3]
#     results = vs.search(query_emb, top_k=2, source_filter=["test1.pdf"])
#     for doc, meta, score in results:
#         print(f"Doc: {doc[:50]}... Meta: {meta}, Score: {score}")
#
#     # Clear
#     vs.clear_collection()