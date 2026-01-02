"""
Base Content Parser - Abstract interface for all content parsers

Provides the common contract that all parsers must implement,
plus shared functionality for domain concept matching.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Generator
from pathlib import Path
import hashlib

from ..models import (
    IndexableChunk, 
    SourceType, 
    SemanticType,
    SourceReference,
    DomainMatch
)
from ..vocabulary import DomainVocabulary


class ContentParser(ABC):
    """
    Abstract base class for content parsers
    
    All content parsers (code, documents, logs) must implement this
    interface. The base class provides shared concept matching and
    chunk ID generation.
    """
    
    # Must be set by subclasses
    SOURCE_TYPE: SourceType = None
    SUPPORTED_EXTENSIONS: List[str] = []
    
    def __init__(self, vocabulary: DomainVocabulary):
        """
        Initialize parser with domain vocabulary
        
        Args:
            vocabulary: Loaded domain vocabulary for concept matching
        """
        self.vocabulary = vocabulary
    
    @abstractmethod
    def parse(self, content: bytes, source_path: str) -> List[IndexableChunk]:
        """
        Parse content and extract indexable chunks
        
        This is the main method that subclasses must implement.
        
        Args:
            content: Raw file content as bytes
            source_path: Path to the source file
            
        Returns:
            List of IndexableChunk objects ready for indexing
        """
        pass
    
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given file
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if this parser can process the file
        """
        pass
    
    def parse_file(self, file_path: str) -> List[IndexableChunk]:
        """
        Parse a file from disk
        
        Convenience method that reads the file and calls parse().
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            List of IndexableChunk objects
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = path.read_bytes()
        return self.parse(content, str(path))
    
    def parse_directory(self, 
                        directory_path: str,
                        recursive: bool = True) -> Generator[IndexableChunk, None, None]:
        """
        Parse all supported files in a directory
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to recurse into subdirectories
            
        Yields:
            IndexableChunk objects from all parsed files
        """
        path = Path(directory_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and self.can_parse(str(file_path)):
                try:
                    chunks = self.parse_file(str(file_path))
                    for chunk in chunks:
                        yield chunk
                except Exception as e:
                    print(f"Warning: Failed to parse {file_path}: {e}")
    
    # ========== Shared Helper Methods ==========
    
    def match_domain_concepts(self, 
                              text: str,
                              context_window: int = 50) -> List[DomainMatch]:
        """
        Match domain concepts in text using vocabulary
        
        This is the shared concept matching logic used by all parsers.
        
        Args:
            text: Text to scan for domain concepts
            context_window: Characters of context around matches
            
        Returns:
            List of DomainMatch objects
        """
        return self.vocabulary.match_text(text, context_window=context_window)
    
    def generate_chunk_id(self, 
                          source_path: str, 
                          content: str,
                          additional: str = "") -> str:
        """
        Generate a unique, deterministic chunk ID
        
        IDs are based on content hash for deduplication.
        
        Args:
            source_path: Source file path
            content: Chunk content
            additional: Additional identifying info (e.g., line number)
            
        Returns:
            Unique chunk ID string
        """
        hash_input = f"{source_path}:{content[:200]}:{additional}"
        content_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        return f"{self.SOURCE_TYPE.value}_{content_hash}"
    
    def create_embedding_text(self,
                              text: str,
                              semantic_type: SemanticType,
                              domain_matches: List[DomainMatch],
                              metadata: Dict[str, Any] = None) -> str:
        """
        Create optimized text for embedding
        
        Transforms raw text into a form that's better for semantic
        search by adding context and domain terminology.
        
        Args:
            text: Raw chunk text
            semantic_type: Type of content
            domain_matches: Matched domain concepts
            metadata: Additional metadata to include
            
        Returns:
            Embedding-optimized text
        """
        parts = []
        
        # Add semantic type context
        type_descriptions = {
            SemanticType.PROCEDURE: "TAL procedure that",
            SemanticType.SUBPROC: "TAL subprocedure that",
            SemanticType.VARIABLE_DECL: "Variable declaration for",
            SemanticType.STRUCT_DEF: "Data structure definition for",
            SemanticType.SECTION: "Document section about",
            SemanticType.PARAGRAPH: "Document content about",
            SemanticType.TABLE: "Data table showing",
            SemanticType.ERROR_TRACE: "Error trace for",
            SemanticType.TRANSACTION: "Transaction record for",
            SemanticType.AUDIT_ENTRY: "Audit log entry for",
        }
        
        type_prefix = type_descriptions.get(semantic_type, "Content related to")
        
        # Add domain concepts
        if domain_matches:
            concepts = list(set(m.canonical_term for m in domain_matches[:5]))
            capabilities = list(set(
                cap for m in domain_matches for cap in m.capabilities[:2]
            ))[:5]
            
            parts.append(f"{type_prefix} {', '.join(concepts)}.")
            if capabilities:
                parts.append(f"Business capabilities: {', '.join(capabilities)}.")
        
        # Add metadata context
        if metadata:
            if 'procedure_name' in metadata:
                parts.append(f"Procedure: {metadata['procedure_name']}")
            if 'section_title' in metadata:
                parts.append(f"Section: {metadata['section_title']}")
            if 'error_code' in metadata:
                parts.append(f"Error: {metadata['error_code']}")
        
        # Add the actual content (truncated if needed)
        max_content_length = 1500 - sum(len(p) for p in parts)
        content_text = text[:max_content_length] if len(text) > max_content_length else text
        parts.append(content_text)
        
        return " ".join(parts)
    
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """
        Extract potential keywords from text
        
        Simple keyword extraction for additional matching.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        import re
        
        # Extract words that look like identifiers or domain terms
        # Handles camelCase, snake_case, UPPERCASE, and hyphenated-terms
        patterns = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
            r'\b[a-z]+(?:_[a-z]+)+\b',            # snake_case
            r'\b[A-Z]+(?:_[A-Z]+)+\b',            # UPPER_CASE
            r'\b[A-Z]{2,}\b',                      # ACRONYMS
            r'\b[a-z]+(?:-[a-z]+)+\b',            # hyphenated-words
        ]
        
        keywords = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)
        
        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def chunk_text(self, 
                   text: str, 
                   max_chunk_size: int = 1000,
                   overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks
        
        Used for long documents that need to be split for embedding.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near the chunk boundary
                for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                    last_sep = text.rfind(sep, start + max_chunk_size // 2, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about this parser"""
        return {
            'source_type': self.SOURCE_TYPE.value if self.SOURCE_TYPE else None,
            'supported_extensions': self.SUPPORTED_EXTENSIONS,
            'parser_class': self.__class__.__name__,
            'vocabulary_loaded': self.vocabulary is not None,
            'vocabulary_terms': len(self.vocabulary.term_to_entry) if self.vocabulary else 0
        }
