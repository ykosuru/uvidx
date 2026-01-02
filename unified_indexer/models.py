"""
Core data models for the Unified Indexer

These models define the common data structures used across all parsers
and the indexing system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json


class SourceType(Enum):
    """Type of source content being indexed"""
    CODE = "code"
    DOCUMENT = "document"
    LOG = "log"
    

class SemanticType(Enum):
    """Semantic classification of chunks"""
    # Code types
    PROCEDURE = "procedure"
    SUBPROC = "subproc"
    FUNCTION = "function"
    VARIABLE_DECL = "variable_declaration"
    STRUCT_DEF = "structure_definition"
    COMMENT = "comment"
    
    # Document types
    SECTION = "section"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    
    # Log types
    ERROR_TRACE = "error_trace"
    TRANSACTION = "transaction"
    AUDIT_ENTRY = "audit_entry"
    DEBUG_LOG = "debug_log"
    
    # Generic
    UNKNOWN = "unknown"


@dataclass
class VocabularyEntry:
    """
    A single entry from the domain vocabulary
    
    Maps keywords to business capabilities and provides
    synonym expansion for matching.
    """
    canonical_term: str
    keywords: List[str]
    related_keywords: List[str]
    description: str
    metadata_category: str
    business_capabilities: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VocabularyEntry':
        """Create VocabularyEntry from vocabulary JSON entry"""
        # Handle keywords - can be string (comma-separated) or list
        keywords_raw = data.get('keywords', '')
        if isinstance(keywords_raw, list):
            keywords = [k.strip() for k in keywords_raw if k]
        elif isinstance(keywords_raw, str):
            keywords = [k.strip() for k in keywords_raw.split(',') if k.strip()]
        else:
            keywords = []
        
        # Handle related_keywords - can be string (comma-separated) or list
        related_raw = data.get('related_keywords', '')
        if isinstance(related_raw, list):
            related = [k.strip() for k in related_raw if k]
        elif isinstance(related_raw, str):
            related = [k.strip() for k in related_raw.split(',') if k.strip()]
        else:
            related = []
        
        # Handle business_capability - can be string or list
        capabilities_raw = data.get('business_capability', [])
        if isinstance(capabilities_raw, str):
            capabilities = [c.strip() for c in capabilities_raw.split(',') if c.strip()]
        elif isinstance(capabilities_raw, list):
            capabilities = capabilities_raw
        else:
            capabilities = []
        
        return cls(
            canonical_term=keywords[0] if keywords else '',
            keywords=keywords,
            related_keywords=related,
            description=data.get('description', ''),
            metadata_category=data.get('metadata', ''),
            business_capabilities=capabilities
        )
    
    def all_terms(self) -> List[str]:
        """Get all searchable terms (keywords + related)"""
        return self.keywords + self.related_keywords


@dataclass
class DomainMatch:
    """
    Represents a matched domain concept in source content
    
    Tracks which vocabulary term was matched, where it occurred,
    and what business capabilities it maps to.
    """
    matched_term: str           # The actual term that matched
    canonical_term: str         # Primary/canonical form from vocabulary
    capabilities: List[str]     # Business capabilities this maps to
    category: str               # Metadata category (e.g., 'payment-networks')
    span: Optional[Tuple[int, int]] = None  # Character offsets in source
    confidence: float = 1.0     # Match confidence (1.0 for exact, lower for fuzzy)
    context: Optional[str] = None  # Surrounding text for context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'matched_term': self.matched_term,
            'canonical_term': self.canonical_term,
            'capabilities': self.capabilities,
            'category': self.category,
            'span': self.span,
            'confidence': self.confidence,
            'context': self.context
        }


@dataclass
class SourceReference:
    """
    Reference to the source location of a chunk
    
    Provides precise location information for navigation
    back to the original source.
    """
    file_path: str
    # For code
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    procedure_name: Optional[str] = None
    
    # For documents
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    
    # For logs
    timestamp: Optional[str] = None
    trace_id: Optional[str] = None
    transaction_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {k: v for k, v in {
            'file_path': self.file_path,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'procedure_name': self.procedure_name,
            'page_number': self.page_number,
            'section_title': self.section_title,
            'timestamp': self.timestamp,
            'trace_id': self.trace_id,
            'transaction_id': self.transaction_id
        }.items() if v is not None}
    
    def __str__(self) -> str:
        """Human-readable source reference"""
        parts = [self.file_path]
        if self.line_start:
            if self.line_end and self.line_end != self.line_start:
                parts.append(f"lines {self.line_start}-{self.line_end}")
            else:
                parts.append(f"line {self.line_start}")
        if self.procedure_name:
            parts.append(f"proc {self.procedure_name}")
        if self.page_number:
            parts.append(f"page {self.page_number}")
        if self.timestamp:
            parts.append(f"@ {self.timestamp}")
        return ":".join(parts)


@dataclass
class IndexableChunk:
    """
    A chunk of content ready for indexing
    
    This is the universal unit of indexing across all source types.
    Contains the raw text, domain matches, and metadata needed
    for both vector and concept-based retrieval.
    """
    # Core content
    chunk_id: str
    text: str                           # Raw text content
    embedding_text: str                 # Optimized text for embedding (may differ from raw)
    
    # Classification
    source_type: SourceType
    semantic_type: SemanticType
    source_ref: SourceReference
    
    # Domain analysis
    domain_matches: List[DomainMatch] = field(default_factory=list)
    
    # Additional context
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed fields (set during indexing)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Generate chunk ID if not provided"""
        if not self.chunk_id:
            content_hash = hashlib.md5(
                f"{self.source_ref.file_path}:{self.text[:100]}".encode()
            ).hexdigest()[:12]
            self.chunk_id = f"{self.source_type.value}_{content_hash}"
    
    @property
    def capability_set(self) -> set:
        """Get all unique business capabilities from domain matches"""
        capabilities = set()
        for match in self.domain_matches:
            capabilities.update(match.capabilities)
        return capabilities
    
    @property
    def canonical_terms(self) -> List[str]:
        """Get list of canonical terms matched"""
        return list(set(m.canonical_term for m in self.domain_matches))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'embedding_text': self.embedding_text,
            'source_type': self.source_type.value,
            'semantic_type': self.semantic_type.value,
            'source_ref': self.source_ref.to_dict(),
            'domain_matches': [m.to_dict() for m in self.domain_matches],
            'context_before': self.context_before,
            'context_after': self.context_after,
            'metadata': self.metadata,
            'capabilities': list(self.capability_set),
            'canonical_terms': self.canonical_terms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexableChunk':
        """Reconstruct from dictionary"""
        return cls(
            chunk_id=data['chunk_id'],
            text=data['text'],
            embedding_text=data['embedding_text'],
            source_type=SourceType(data['source_type']),
            semantic_type=SemanticType(data['semantic_type']),
            source_ref=SourceReference(**data['source_ref']),
            domain_matches=[DomainMatch(**m) for m in data.get('domain_matches', [])],
            context_before=data.get('context_before'),
            context_after=data.get('context_after'),
            metadata=data.get('metadata', {})
        )


@dataclass
class SearchResult:
    """
    A single search result from the hybrid index
    
    Combines vector similarity with concept matching scores.
    """
    chunk: IndexableChunk
    
    # Scoring
    vector_score: float = 0.0           # Cosine similarity from vector search
    concept_score: float = 0.0          # Score from concept/keyword matching
    combined_score: float = 0.0         # Fused score
    
    # Match details
    matched_concepts: List[str] = field(default_factory=list)
    matched_capabilities: List[str] = field(default_factory=list)
    
    # Ranking metadata
    rank: int = 0
    retrieval_method: str = "hybrid"    # "vector", "concept", or "hybrid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display/serialization"""
        return {
            'chunk_id': self.chunk.chunk_id,
            'text_preview': self.chunk.text[:200] + '...' if len(self.chunk.text) > 200 else self.chunk.text,
            'source_type': self.chunk.source_type.value,
            'source_ref': str(self.chunk.source_ref),
            'vector_score': round(self.vector_score, 4),
            'concept_score': round(self.concept_score, 4),
            'combined_score': round(self.combined_score, 4),
            'matched_concepts': self.matched_concepts,
            'matched_capabilities': self.matched_capabilities,
            'rank': self.rank,
            'retrieval_method': self.retrieval_method
        }
