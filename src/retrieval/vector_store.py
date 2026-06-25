import hashlib
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple, Optional

# Try to import rank_bm25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

class VectorStore:
    def __init__(self, collection_name: str = "ngs_docs", persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        # Use cosine distance — produces values in [0, 2] and is the correct
        # metric for text embeddings.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # BM25 index for hybrid search
        self._bm25_index = None
        self._bm25_documents = []
        self._bm25_ids = []

    @staticmethod
    def _make_chunk_id(source: str, page: int, text: str) -> str:
        fingerprint = f"{source}|{page}|{text}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Basic tokenization: lowercase, split on whitespace and punctuation
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def _build_bm25_index(self):
        """Build BM25 index from documents in ChromaDB."""
        if not BM25_AVAILABLE:
            return
        try:
            results = self.collection.get(include=["documents", "metadatas"])
            documents = results.get("documents", [])
            ids = results.get("ids", [])
            if documents:
                tokenized_docs = [self._tokenize(doc) for doc in documents]
                self._bm25_index = BM25Okapi(tokenized_docs)
                self._bm25_documents = documents
                self._bm25_ids = ids
        except Exception as e:
            print(f"Error building BM25 index: {e}")

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
            # Rebuild BM25 index after adding new documents
            if BM25_AVAILABLE:
                self._build_bm25_index()
        except Exception as e:
            print(f"Error upserting chunks: {e}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        source_filter: Optional[List[str]] = None,
        max_distance: Optional[float] = None,
        hybrid: bool = False,
        query_text: Optional[str] = None,
    ) -> List[Tuple[str, Dict, float]]:
        """
        Search the vector store.

        Args:
            query_embedding: The embedding vector for the query.
            top_k: Number of results to return.
            source_filter: Optional list of source filenames to filter by.
            max_distance: Maximum cosine distance for results.
            hybrid: If True, combine vector search with BM25 keyword search.
            query_text: Required for hybrid search - the original query text.

        Returns:
            List of (document_text, metadata_dict, distance) tuples.
        """
        where = None
        if source_filter:
            if len(source_filter) == 1:
                where = {"source": {"$eq": source_filter[0]}}
            else:
                where = {"$or": [{"source": {"$eq": s}} for s in source_filter]}

        try:
            # Vector search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2 if hybrid else top_k,  # Get more for reranking
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            docs      = results["documents"][0]
            metas     = results["metadatas"][0]
            distances = results["distances"][0]

            hits = list(zip(docs, metas, distances))

            # Hybrid search: combine with BM25
            if hybrid and query_text and BM25_AVAILABLE and self._bm25_index:
                # Get BM25 scores
                query_tokens = self._tokenize(query_text)
                bm25_scores = self._bm25_index.get_scores(query_tokens)

                # Normalize BM25 scores to [0, 1]
                if len(bm25_scores) > 0:
                    bm25_max = max(bm25_scores)
                    bm25_min = min(bm25_scores)
                    if bm25_max > bm25_min:
                        bm25_norm = [(s - bm25_min) / (bm25_max - bm25_min) for s in bm25_scores]
                    else:
                        bm25_norm = [0.0] * len(bm25_scores)
                else:
                    bm25_norm = []

                # Normalize vector distances to [0, 1] (cosine distance is in [0, 2])
                if distances:
                    dist_norm = [d / 2.0 for d in distances]
                else:
                    dist_norm = []

                # Combine: vector_score = 1 - normalized_distance, then weighted sum
                combined_scores = []
                for i, (doc, meta, dist) in enumerate(hits):
                    vector_score = 1.0 - (dist / 2.0)  # Convert distance to similarity
                    # Find BM25 score for this document
                    doc_idx = -1
                    for idx, d in enumerate(self._bm25_documents):
                        if d == doc:
                            doc_idx = idx
                            break
                    bm25_score = bm25_norm[doc_idx] if doc_idx >= 0 and doc_idx < len(bm25_norm) else 0.0

                    # Weighted combination: 60% vector, 40% keyword
                    combined = 0.6 * vector_score + 0.4 * bm25_score
                    combined_scores.append((doc, meta, dist, combined))

                # Sort by combined score descending
                combined_scores.sort(key=lambda x: x[3], reverse=True)
                hits = [(doc, meta, dist) for doc, meta, dist, _ in combined_scores[:top_k]]

            if max_distance is not None:
                hits = [(doc, meta, dist) for doc, meta, dist in hits if dist <= max_distance]

            return hits[:top_k]

        except Exception as e:
            print(f"Error searching collection: {e}")
            return []

    def clear_collection(self) -> None:
        try:
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
            # Reset BM25 index
            self._bm25_index = None
            self._bm25_documents = []
            self._bm25_ids = []
        except Exception as e:
            print(f"Error clearing collection: {e}")
