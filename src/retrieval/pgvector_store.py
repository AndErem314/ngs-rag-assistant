import json
import hashlib
import psycopg2
from psycopg2.extras import DictCursor
from typing import List, Dict, Tuple, Optional


class PgvectorStore:
    """
    Secondary vector store using PostgreSQL + pgvector extension.
    Mirrors the interface of the existing ChromaDB-based VectorStore.
    """

    def __init__(
        self,
        db_url: str = "postgresql://ngs_user:ngs_secure_password@localhost:5432/ngs_rag",
        collection_name: str = "documents",
        embedding_dim: int = 1024  # Default for mxbai-embed-large-latest
    ):
        self.db_url = db_url
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._init_db()

    def _init_db(self):
        """Initialize PostgreSQL with pgvector extension and create table if not exists."""
        with psycopg2.connect(self.db_url) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Create pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create table with id, content, metadata (JSONB), embedding (vector)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.collection_name} (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB,
                        embedding VECTOR({self.embedding_dim})
                    );
                """)

                # Create index for fast cosine similarity search
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.collection_name}_embedding_idx 
                    ON {self.collection_name} 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)

    @staticmethod
    def _make_chunk_id(source: str, page: int, text: str) -> str:
        """Generate content-hash ID matching the existing VectorStore logic."""
        fingerprint = f"{source}|{page}|{text}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """
        Add or update chunks with their embeddings (upsert logic matching ChromaDB).
        Deduplicates by chunk ID.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match."
            )

        seen_ids = set()
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    source = chunk["metadata"]["source"]
                    page = chunk["metadata"]["page"]
                    chunk_id = self._make_chunk_id(source, page, chunk["text"])

                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)

                    # Upsert: insert or update on conflict
                    cur.execute(f"""
                        INSERT INTO {self.collection_name} (id, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding;
                    """, (
                        chunk_id,
                        chunk["text"],
                        json.dumps(chunk["metadata"]),
                        embedding
                    ))
            conn.commit()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        source_filter: Optional[List[str]] = None,
        max_distance: Optional[float] = None
    ) -> List[Tuple[str, Dict, float]]:
        """
        Search for similar chunks using cosine distance (pgvector's <=> operator).
        Returns list of (content, metadata_dict, distance).
        """
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                # Base query
                query = f"""
                    SELECT content, metadata, (embedding <=> %s::vector) AS distance
                    FROM {self.collection_name}
                """
                params = [query_embedding]

                # Add source filter
                if source_filter:
                    if len(source_filter) == 1:
                        query += " WHERE metadata->>'source' = %s"
                        params.append(source_filter[0])
                    else:
                        query += " WHERE metadata->>'source' IN %s"
                        params.append(tuple(source_filter))

                # Add max distance filter
                if max_distance is not None:
                    clause = " AND (embedding <=> %s::vector) <= %s"
                    if "WHERE" not in query:
                        clause = " WHERE (embedding <=> %s::vector) <= %s"
                    query += clause
                    params.extend([query_embedding, max_distance])

                # Add ordering and limit
                query += " ORDER BY distance ASC LIMIT %s;"
                params.append(top_k)

                cur.execute(query, params)
                results = cur.fetchall()

                # Process results
                output = []
                for row in results:
                    content = row["content"]
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                    distance = row["distance"]
                    output.append((content, metadata, distance))

                return output

    def clear_collection(self) -> None:
        """Delete all records from the collection table."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.collection_name};")
            conn.commit()
