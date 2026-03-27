import hashlib
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple, Optional


class VectorStore:
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        # Use cosine distance — produces values in [0, 2] and is the correct
        # metric for text embeddings. ChromaDB 1.x defaults to l2 (squared
        # Euclidean) which yields huge values (100–200+) for high-dimensional
        # vectors like nomic-embed-text-v2-moe, making distance thresholds
        # meaningless. Must clear+recreate the collection if it was created
        # without this setting.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _make_chunk_id(source: str, page: int, text: str) -> str:
        fingerprint = f"{source}|{page}|{text}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) and embeddings "
                f"({len(embeddings)}) must match."
            )

        seen_ids: set = set()
        ids: List[str] = []
        metadatas: List[Dict] = []
        documents: List[str] = []
        deduped_embeddings: List[List[float]] = []

        for chunk, embedding in zip(chunks, embeddings):
            source = chunk["metadata"]["source"]
            page   = chunk["metadata"]["page"]
            chunk_id = self._make_chunk_id(source, page, chunk["text"])

            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            ids.append(chunk_id)
            metadatas.append({"source": source, "page": page})
            documents.append(chunk["text"])
            deduped_embeddings.append(embedding)

        if not ids:
            return

        try:
            self.collection.upsert(
                embeddings=deduped_embeddings,
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
        where = None
        if source_filter:
            if len(source_filter) == 1:
                where = {"source": {"$eq": source_filter[0]}}
            else:
                where = {"$or": [{"source": {"$eq": s}} for s in source_filter]}

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
        try:
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
        except Exception as e:
            print(f"Error clearing collection: {e}")