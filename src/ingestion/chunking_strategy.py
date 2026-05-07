from enum import Enum
from typing import List, Tuple, Dict, Optional

class ChunkingStrategy(Enum):
    """Available chunking strategies for NGS RAG pipeline."""
    BASIC = "basic"                    # Original RecursiveCharacterTextSplitter
    TABLE_AWARE = "table_aware"        # Table-aware + text chunking
    SEMANTIC = "semantic"              # Embedding-based semantic chunking
    KEYWORD_ANCHORED = "keyword"       # NGS keyword-anchored chunking
