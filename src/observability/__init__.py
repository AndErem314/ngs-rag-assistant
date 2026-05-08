"""
Observability module for NGS RAG pipeline.

Provides metrics collection, drift detection, and dashboard components.
"""

from .metrics import MetricsCollector, LatencyTimer

__all__ = ["MetricsCollector", "LatencyTimer"]
