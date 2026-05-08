"""
Observability Module for NGS RAG Pipeline

Tracks metrics for retrieval quality, latency, and system health.
Metrics are logged to SQLite database for historical analysis.

Usage:
    from src.observability.metrics import MetricsCollector
    
    collector = MetricsCollector()
    collector.log_retrieval_metrics(
        query="DNA extraction",
        strategy="basic",
        hybrid=True,
        exact_match=True,
        distance=0.25,
        latency_ms=150
    )
"""

import sqlite3
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class MetricsCollector:
    """Collects and stores RAG pipeline metrics in SQLite."""

    def __init__(self, db_path: str = "observability/metrics.db"):
        """Initialize metrics collector with SQLite backend."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create metrics tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS retrieval_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT,
                    strategy TEXT,
                    hybrid BOOLEAN,
                    exact_match BOOLEAN,
                    in_top_k BOOLEAN,
                    distance REAL,
                    latency_ms INTEGER,
                    num_results INTEGER,
                    expected_page INTEGER,
                    retrieved_pages TEXT
                );
                
                CREATE TABLE IF NOT EXISTS embedding_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT,
                    text_length INTEGER,
                    embedding_dim INTEGER,
                    latency_ms INTEGER
                );
                
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT,
                    metric_value REAL,
                    tags TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_retrieval_timestamp 
                ON retrieval_metrics(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_embedding_timestamp 
                ON embedding_metrics(timestamp);
            """)

    def log_retrieval_metrics(
        self,
        query: str,
        strategy: str,
        hybrid: bool,
        exact_match: bool,
        in_top_k: bool,
        distance: float,
        latency_ms: int,
        num_results: int,
        expected_page: Optional[int] = None,
        retrieved_pages: Optional[list] = None,
    ):
        """Log retrieval accuracy metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO retrieval_metrics
                (timestamp, query, strategy, hybrid, exact_match, in_top_k,
                 distance, latency_ms, num_results, expected_page, retrieved_pages)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                query[:500],  # Truncate long queries
                strategy,
                hybrid,
                exact_match,
                in_top_k,
                distance,
                latency_ms,
                num_results,
                expected_page,
                str(retrieved_pages) if retrieved_pages else None,
            ))

    def log_embedding_metrics(
        self,
        model: str,
        text_length: int,
        embedding_dim: int,
        latency_ms: int,
    ):
        """Log embedding generation metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO embedding_metrics
                (timestamp, model, text_length, embedding_dim, latency_ms)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                model,
                text_length,
                embedding_dim,
                latency_ms,
            ))

    def log_system_metric(
        self,
        metric_name: str,
        metric_value: float,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Log system-level metrics (e.g., chunk quality, cache hit rate)."""
        import json
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_metrics
                (timestamp, metric_name, metric_value, tags)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                metric_name,
                metric_value,
                json.dumps(tags) if tags else None,
            ))

    def get_retrieval_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of retrieval metrics for the last N days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN exact_match THEN 1 ELSE 0 END) as exact_matches,
                    SUM(CASE WHEN in_top_k THEN 1 ELSE 0 END) as top_k_matches,
                    AVG(distance) as avg_distance,
                    AVG(latency_ms) as avg_latency
                FROM retrieval_metrics
                WHERE timestamp > datetime('now', ? || ' days')
            """, (f"-{days}",))

            row = cursor.fetchone()
            if row:
                total = row["total_queries"] or 1
                return {
                    "total_queries": total,
                    "exact_accuracy": (row["exact_matches"] or 0) / total,
                    "tolerance_accuracy": (row["top_k_matches"] or 0) / total,
                    "avg_distance": row["avg_distance"] or 0,
                    "avg_latency_ms": row["avg_latency"] or 0,
                }
            return {}

    def get_strategy_comparison(self) -> Dict[str, Any]:
        """Compare performance across chunking strategies."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    strategy,
                    hybrid,
                    COUNT(*) as total,
                    SUM(CASE WHEN exact_match THEN 1 ELSE 0 END) as exact,
                    AVG(distance) as avg_dist,
                    AVG(latency_ms) as avg_lat
                FROM retrieval_metrics
                GROUP BY strategy, hybrid
                ORDER BY strategy, hybrid
            """)

            results = []
            for row in cursor.fetchall():
                total = row["total"] or 1
                results.append({
                    "strategy": row["strategy"],
                    "hybrid": bool(row["hybrid"]),
                    "total_queries": row["total"],
                    "exact_accuracy": (row["exact"] or 0) / total,
                    "avg_distance": row["avg_dist"] or 0,
                    "avg_latency_ms": row["avg_lat"] or 0,
                })
            return results


class LatencyTimer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, metric_name: str, tags: Optional[Dict] = None):
        self.collector = collector
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        self.collector.log_system_metric(
            metric_name=self.metric_name,
            metric_value=elapsed_ms,
            tags=self.tags,
        )


if __name__ == "__main__":
    # Quick test
    collector = MetricsCollector()
    collector.log_retrieval_metrics(
        query="DNA extraction protocol",
        strategy="basic",
        hybrid=False,
        exact_match=True,
        in_top_k=True,
        distance=0.25,
        latency_ms=120,
        num_results=5,
        expected_page=1,
        retrieved_pages=[1, 2, 3, 4, 5],
    )
    print("Metrics logged successfully!")
    print("\nRetrieval Summary (7 days):")
    print(collector.get_retrieval_summary())
    print("\nStrategy Comparison:")
    print(collector.get_strategy_comparison())
