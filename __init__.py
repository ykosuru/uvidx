"""
Unified Indexer - Cross-reference code, documents, and logs with domain-aware parsing

This package provides a unified indexing system that can:
1. Parse TAL/COBOL code, PDFs, and transaction logs
2. Extract domain concepts using a payment systems vocabulary
3. Index into a hybrid vector + concept store
4. Enable cross-type retrieval (find code that handles errors from logs)

Architecture:
- ContentParser: Abstract base with shared concept matching
- TalCodeParser: Leverages existing tal_enhanced_parser
- DocumentParser: PDF/DOCX parsing with section awareness  
- LogParser: JSON/structured log parsing with trace correlation
- HybridIndex: Dual vector + concept indexing for retrieval

Local Embeddings (no external APIs):
- HybridEmbedder: Domain concepts + text features
- TFIDFEmbedder: TF-IDF with domain boosting
- HashEmbedder: Feature hashing (no fitting needed)
- DomainConceptEmbedder: Pure domain concept matching
- BM25Embedder: BM25 ranking-based embeddings

LLM Integration (optional):
- LLMInterface: Abstract interface for LLM invocation (implement with your provider)
- LLMEnhancedPipeline: Pipeline with LLM-powered enhancements
"""

from .models import (
    SourceType,
    DomainMatch,
    IndexableChunk,
    SearchResult,
    VocabularyEntry
)

from .vocabulary import DomainVocabulary
from .parsers.base import ContentParser
from .parsers.tal_parser import TalCodeParser
from .parsers.document_parser import DocumentParser
from .parsers.log_parser import LogParser
from .index import HybridIndex

from .embeddings import (
    create_embedder,
    HybridEmbedder,
    TFIDFEmbedder,
    HashEmbedder,
    DomainConceptEmbedder,
    BM25Embedder,
    TextPreprocessor,
    cosine_similarity,
    batch_cosine_similarity
)

from .pipeline import (
    IndexingPipeline,
    LLMInterface,
    LLMEnhancedPipeline,
    ExampleLLMImplementation
)

__version__ = "1.0.0"
__all__ = [
    # Models
    "SourceType",
    "DomainMatch", 
    "IndexableChunk",
    "SearchResult",
    "VocabularyEntry",
    # Vocabulary
    "DomainVocabulary",
    # Parsers
    "ContentParser",
    "TalCodeParser",
    "DocumentParser",
    "LogParser",
    # Index
    "HybridIndex",
    # Local Embeddings
    "create_embedder",
    "HybridEmbedder",
    "TFIDFEmbedder",
    "HashEmbedder",
    "DomainConceptEmbedder",
    "BM25Embedder",
    "TextPreprocessor",
    "cosine_similarity",
    "batch_cosine_similarity",
    # Pipeline
    "IndexingPipeline",
    # LLM Integration
    "LLMInterface",
    "LLMEnhancedPipeline",
    "ExampleLLMImplementation"
]
