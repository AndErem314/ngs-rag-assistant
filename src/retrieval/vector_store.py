import hashlib
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple, Optional


class VectorStore:
    """
    A persistent vector store using ChromaDB. Stores chunks with embeddings
    and metadata. Supports filtering by source file.

    Chunk IDs are derived from a SHA-256 hash of the source filename, page
    number, and the first 128 characters of the chunk text. This makes IDs
    stable and deterministic: re-ingesting the same PDF will produce the same
    IDs and update existing entries in place, rather than silently creating
    duplicates or overwriting unrelated chunks.
    """

    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the Chroma collection.
            persist_directory: Directory where ChromaDB persists data.
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk_id(source: str, page: int, text: str) -> str:
        """
        Derive a stable, collision-resistant ID for a chunk.

        The ID is a 16-character hex prefix of the SHA-256 digest computed
        from the source filename, page number, and the first 128 characters
        of the chunk text. This prevents:
          - Silent overwrites when re-ingesting the same document (same hash
            → ChromaDB upserts the existing record in place).
          - Accidental collisions between chunks from different sources that
            happen to share a sequential index.

        Args:
            source: Source filename (e.g. "TruSight_500_Manual.pdf").
            page:   Page number within that source.
            text:   Full chunk text (only the first 128 chars are hashed to
                    keep computation cheap while still being discriminating).

        Returns:
            A 16-character lowercase hex string.
        """
        fingerprint = f"{source}|{page}|{text[:128]}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """
        Add (or update) chunks and their embeddings in the vector store.

        ChromaDB's ``upsert`` is used instead of ``add`` so that re-ingesting
        the same document is safe: existing records with the same ID are
        updated rather than raising a duplicate-key error.

        Args:
            chunks: List of dicts, each containing:
                      - "text"     (str)  – the chunk text
                      - "metadata" (dict) – must include "source" (str) and
                                            "page" (int)
            embeddings: List of embedding vectors, one per chunk.

        Raises:
            ValueError: If the number of chunks and embeddings do not match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) and embeddings "
                f"({len(embeddings)}) must match."
            )

        ids = []
        metadatas = []
        documents = []

        for chunk, embedding in zip(chunks, embeddings):
            source = chunk["metadata"]["source"]
            page   = chunk["metadata"]["page"]

            ids.append(self._make_chunk_id(source, page, chunk["text"]))
            metadatas.append({"source": source, "page": page})
            documents.append(chunk["text"])

        try:
            # upsert: insert new records, update existing ones with the same ID
            self.collection.upsert(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids,
            )
        except Exception as e:
            print(f"Error upserting chunks: {e}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        source_filter: Optional[List[str]] = None,
        max_distance: Optional[float] = None,
    ) -> List[Tuple[str, Dict, float]]:
        """
        Search for the most similar chunks to a query embedding.

        Args:
            query_embedding: Query embedding vector.
            top_k:           Maximum number of results to return.
            source_filter:   Optional list of source filenames to restrict
                             the search to.
            max_distance:    Optional upper bound on ChromaDB distance
                             (lower = more similar). Results with a distance
                             above this threshold are discarded. A typical
                             useful value for cosine distance is 0.5–1.0;
                             leave as None to return all top_k results
                             regardless of quality.

        Returns:
            List of (document_text, metadata, distance) tuples, sorted from
            most to least similar. Distance is ChromaDB's raw distance metric
            (lower is better).
        """
        where = None
        if source_filter:
            where = {"source": {"$in": source_filter}}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            docs      = results["documents"][0]
            metas     = results["metadatas"][0]
            distances = results["distances"][0]

            hits = list(zip(docs, metas, distances))

            if max_distance is not None:
                hits = [(doc, meta, dist) for doc, meta, dist in hits if dist <= max_distance]

            return hits

        except Exception as e:
            print(f"Error searching collection: {e}")
            return []

    def clear_collection(self) -> None:
        """Delete all documents from the collection."""
        try:
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
#     chunks = [
#         {"text": "DNA extraction requires at least 100 ng input.",
#          "metadata": {"source": "manual.pdf", "page": 5}},
#         {"text": "Covaris E220: 140W PIP, 10% DC, 200 cycles/burst, 80s.",
#          "metadata": {"source": "manual.pdf", "page": 12}},
#     ]
#     embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
#     vs.add_chunks(chunks, embeddings)
#
#     # Re-ingesting the same chunks is safe — upsert updates in place.
#     vs.add_chunks(chunks, embeddings)
#
#     query_emb = [0.1, 0.2, 0.3]
#     results = vs.search(query_emb, top_k=2, max_distance=1.0)
#     for doc, meta, dist in results:
#         print(f"[{dist:.4f}] {meta} — {doc[:60]}")
#
#     vs.clear_collection()