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
from .pipeline import IndexingPipeline

__version__ = "1.0.0"
__all__ = [
    "SourceType",
    "DomainMatch", 
    "IndexableChunk",
    "SearchResult",
    "VocabularyEntry",
    "DomainVocabulary",
    "ContentParser",
    "TalCodeParser",
    "DocumentParser",
    "LogParser",
    "HybridIndex",
    "IndexingPipeline"
]
