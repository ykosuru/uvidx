"""
Indexing Pipeline - Orchestrates parsing and indexing of multiple content types

Provides a unified interface for indexing code, documents, and logs
together with a shared domain vocabulary.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import IndexableChunk, SourceType, SearchResult
from .vocabulary import DomainVocabulary
from .parsers.base import ContentParser
from .parsers.tal_parser import TalCodeParser
from .parsers.document_parser import DocumentParser
from .parsers.log_parser import LogParser
from .index import HybridIndex


@dataclass
class IndexingResult:
    """Result of indexing a single file"""
    file_path: str
    source_type: str
    success: bool
    chunks_created: int
    error: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class PipelineStatistics:
    """Statistics from a pipeline run"""
    files_processed: int = 0
    files_failed: int = 0
    total_chunks: int = 0
    by_source_type: Dict[str, int] = field(default_factory=dict)
    by_capability: Dict[str, int] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class IndexingPipeline:
    """
    Unified pipeline for indexing code, documents, and logs
    
    Orchestrates:
    1. Loading domain vocabulary
    2. Selecting appropriate parsers
    3. Parsing content into chunks
    4. Indexing chunks in hybrid store
    5. Providing search interface
    """
    
    def __init__(self, 
                 vocabulary_path: Optional[str] = None,
                 vocabulary_data: Optional[List[Dict]] = None,
                 embedding_fn: Optional[Callable[[str], List[float]]] = None,
                 tal_parser_path: Optional[str] = None):
        """
        Initialize the indexing pipeline
        
        Args:
            vocabulary_path: Path to vocabulary JSON file
            vocabulary_data: Vocabulary as list of dicts (alternative to path)
            embedding_fn: Function to generate embeddings
            tal_parser_path: Path to TAL parser modules
        """
        # Load vocabulary
        self.vocabulary = DomainVocabulary()
        if vocabulary_path:
            self.vocabulary.load(vocabulary_path)
        elif vocabulary_data:
            self.vocabulary.load_from_data(vocabulary_data)
        
        # Initialize parsers
        self.parsers: Dict[SourceType, ContentParser] = {}
        self._init_parsers(tal_parser_path)
        
        # Initialize index
        self.index = HybridIndex(self.vocabulary, embedding_fn)
        
        # Statistics
        self.stats = PipelineStatistics()
    
    def _init_parsers(self, tal_parser_path: Optional[str] = None):
        """Initialize all content parsers"""
        self.parsers[SourceType.CODE] = TalCodeParser(
            self.vocabulary, 
            tal_parser_path=tal_parser_path
        )
        self.parsers[SourceType.DOCUMENT] = DocumentParser(self.vocabulary)
        self.parsers[SourceType.LOG] = LogParser(self.vocabulary)
    
    def set_embedding_function(self, fn: Callable[[str], List[float]]):
        """Set the embedding function for vector search"""
        self.index.set_embedding_function(fn)
    
    def get_parser_for_file(self, file_path: str) -> Optional[ContentParser]:
        """
        Determine the appropriate parser for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Appropriate parser or None if no parser matches
        """
        for parser in self.parsers.values():
            if parser.can_parse(file_path):
                return parser
        return None
    
    def index_file(self, file_path: str) -> IndexingResult:
        """
        Index a single file
        
        Args:
            file_path: Path to the file
            
        Returns:
            IndexingResult with details
        """
        import time
        start_time = time.time()
        
        # Find appropriate parser
        parser = self.get_parser_for_file(file_path)
        
        if not parser:
            return IndexingResult(
                file_path=file_path,
                source_type="unknown",
                success=False,
                chunks_created=0,
                error="No parser found for file type"
            )
        
        try:
            # Parse file
            chunks = parser.parse_file(file_path)
            
            # Index chunks
            for chunk in chunks:
                self.index.index_chunk(chunk)
            
            processing_time = (time.time() - start_time) * 1000
            
            return IndexingResult(
                file_path=file_path,
                source_type=parser.SOURCE_TYPE.value,
                success=True,
                chunks_created=len(chunks),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return IndexingResult(
                file_path=file_path,
                source_type=parser.SOURCE_TYPE.value if parser else "unknown",
                success=False,
                chunks_created=0,
                error=str(e),
                processing_time_ms=processing_time
            )
    
    def index_directory(self, 
                        directory: str,
                        recursive: bool = True,
                        extensions: Optional[List[str]] = None,
                        exclude_patterns: Optional[List[str]] = None,
                        max_workers: int = 4,
                        progress_callback: Optional[Callable[[str, int, int], None]] = None
                        ) -> PipelineStatistics:
        """
        Index all supported files in a directory
        
        Args:
            directory: Path to directory
            recursive: Whether to recurse into subdirectories
            extensions: Optional list of extensions to include
            exclude_patterns: Patterns to exclude (e.g., ['*.bak', 'test_*'])
            max_workers: Number of parallel workers
            progress_callback: Called with (file_path, current, total)
            
        Returns:
            PipelineStatistics with run results
        """
        import time
        import fnmatch
        
        start_time = time.time()
        path = Path(directory)
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        # Collect files to process
        files_to_process = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue
            
            # Check extension filter
            if extensions:
                if file_path.suffix.lower() not in extensions:
                    continue
            
            # Check exclude patterns
            if exclude_patterns:
                excluded = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(file_path.name, pattern):
                        excluded = True
                        break
                if excluded:
                    continue
            
            # Check if we have a parser
            if self.get_parser_for_file(str(file_path)):
                files_to_process.append(str(file_path))
        
        total_files = len(files_to_process)
        print(f"Found {total_files} files to index")
        
        # Process files
        stats = PipelineStatistics()
        
        if max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.index_file, fp): fp 
                    for fp in files_to_process
                }
                
                for i, future in enumerate(as_completed(futures)):
                    file_path = futures[future]
                    
                    if progress_callback:
                        progress_callback(file_path, i + 1, total_files)
                    
                    try:
                        result = future.result()
                        self._update_stats(stats, result)
                    except Exception as e:
                        stats.files_failed += 1
                        stats.errors.append(f"{file_path}: {e}")
        else:
            # Sequential processing
            for i, file_path in enumerate(files_to_process):
                if progress_callback:
                    progress_callback(file_path, i + 1, total_files)
                
                result = self.index_file(file_path)
                self._update_stats(stats, result)
        
        stats.processing_time_seconds = time.time() - start_time
        self.stats = stats
        
        return stats
    
    def _update_stats(self, stats: PipelineStatistics, result: IndexingResult):
        """Update statistics from an indexing result"""
        if result.success:
            stats.files_processed += 1
            stats.total_chunks += result.chunks_created
            
            source_type = result.source_type
            if source_type not in stats.by_source_type:
                stats.by_source_type[source_type] = 0
            stats.by_source_type[source_type] += result.chunks_created
        else:
            stats.files_failed += 1
            if result.error:
                stats.errors.append(f"{result.file_path}: {result.error}")
    
    def index_content(self, 
                      content: bytes,
                      source_path: str,
                      source_type: Optional[SourceType] = None) -> List[IndexableChunk]:
        """
        Index content directly (not from file)
        
        Args:
            content: Raw content bytes
            source_path: Virtual path for the content
            source_type: Explicit source type (auto-detected if None)
            
        Returns:
            List of created chunks
        """
        # Determine parser
        if source_type:
            parser = self.parsers.get(source_type)
        else:
            parser = self.get_parser_for_file(source_path)
        
        if not parser:
            raise ValueError(f"No parser available for: {source_path}")
        
        # Parse and index
        chunks = parser.parse(content, source_path)
        
        for chunk in chunks:
            self.index.index_chunk(chunk)
        
        return chunks
    
    def search(self, 
               query: str,
               top_k: int = 10,
               source_types: Optional[List[SourceType]] = None,
               capabilities: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Search the index
        
        Args:
            query: Search query
            top_k: Number of results
            source_types: Filter by source types
            capabilities: Filter by business capabilities
            
        Returns:
            List of SearchResult objects
        """
        return self.index.search(
            query,
            top_k=top_k,
            source_types=source_types,
            capabilities=capabilities
        )
    
    def search_cross_reference(self,
                               query: str,
                               from_type: SourceType,
                               to_types: List[SourceType],
                               top_k: int = 5) -> Dict[str, List[SearchResult]]:
        """
        Cross-reference search across content types
        
        Example: Find code that handles errors from logs
        
        Args:
            query: Search query
            from_type: Primary source type
            to_types: Source types to cross-reference
            top_k: Results per source type
            
        Returns:
            Dict mapping source type to results
        """
        return self.index.search_cross_reference(
            query,
            source_type=from_type,
            reference_types=to_types,
            top_k=top_k
        )
    
    def get_related_code(self, 
                         error_message: str,
                         top_k: int = 5) -> List[SearchResult]:
        """
        Find code that might handle a specific error
        
        Convenience method for a common cross-reference pattern.
        
        Args:
            error_message: Error message or code
            top_k: Number of results
            
        Returns:
            List of code chunks that might be relevant
        """
        return self.search(
            error_message,
            top_k=top_k,
            source_types=[SourceType.CODE]
        )
    
    def get_documentation(self, 
                          topic: str,
                          top_k: int = 5) -> List[SearchResult]:
        """
        Find documentation about a topic
        
        Args:
            topic: Topic to search for
            top_k: Number of results
            
        Returns:
            List of document chunks
        """
        return self.search(
            topic,
            top_k=top_k,
            source_types=[SourceType.DOCUMENT]
        )
    
    def get_by_capability(self, 
                          capability: str,
                          top_k: int = 10) -> List[SearchResult]:
        """
        Find all content related to a business capability
        
        Args:
            capability: Business capability name
            top_k: Number of results
            
        Returns:
            List of results across all source types
        """
        return self.index.search_by_capability(capability, top_k)
    
    def save(self, directory: str):
        """
        Save the index to disk
        
        Args:
            directory: Directory to save index
        """
        self.index.save(directory)
        
        # Also save vocabulary and metadata
        path = Path(directory)
        
        with open(path / 'vocabulary.json', 'w') as f:
            json.dump(self.vocabulary.to_dict(), f, indent=2)
        
        with open(path / 'pipeline_stats.json', 'w') as f:
            json.dump({
                'files_processed': self.stats.files_processed,
                'files_failed': self.stats.files_failed,
                'total_chunks': self.stats.total_chunks,
                'by_source_type': self.stats.by_source_type,
                'processing_time_seconds': self.stats.processing_time_seconds,
                'saved_at': datetime.now().isoformat()
            }, f, indent=2)
    
    def load(self, directory: str):
        """
        Load the index from disk
        
        Args:
            directory: Directory containing saved index
        """
        self.index.load(directory)
        
        # Load vocabulary if saved
        vocab_path = Path(directory) / 'vocabulary.json'
        if vocab_path.exists():
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            self.vocabulary.load_from_data(vocab_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            'pipeline': {
                'files_processed': self.stats.files_processed,
                'files_failed': self.stats.files_failed,
                'total_chunks': self.stats.total_chunks,
                'by_source_type': self.stats.by_source_type
            },
            'index': self.index.get_statistics(),
            'vocabulary': self.vocabulary.get_statistics()
        }
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("INDEXING PIPELINE STATISTICS")
        print("=" * 60)
        
        print("\nPipeline:")
        print(f"  Files processed: {stats['pipeline']['files_processed']}")
        print(f"  Files failed: {stats['pipeline']['files_failed']}")
        print(f"  Total chunks: {stats['pipeline']['total_chunks']}")
        
        print("\n  By source type:")
        for st, count in stats['pipeline']['by_source_type'].items():
            print(f"    {st}: {count} chunks")
        
        print("\nIndex:")
        idx_stats = stats['index']
        print(f"  Vector store: {idx_stats.get('vector_store_size', 0)} chunks")
        
        if 'concept_index' in idx_stats:
            ci = idx_stats['concept_index']
            print(f"  Unique concepts: {ci.get('unique_concepts', 0)}")
            print(f"  Unique capabilities: {ci.get('unique_capabilities', 0)}")
        
        print("\nVocabulary:")
        v_stats = stats['vocabulary']
        print(f"  Total entries: {v_stats.get('total_entries', 0)}")
        print(f"  Searchable terms: {v_stats.get('total_terms', 0)}")
        
        print("\n" + "=" * 60)


def create_pipeline_with_openai(vocabulary_path: str,
                                 openai_api_key: Optional[str] = None,
                                 model: str = "text-embedding-3-small") -> IndexingPipeline:
    """
    Create a pipeline with OpenAI embeddings
    
    Args:
        vocabulary_path: Path to vocabulary JSON
        openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model: Embedding model to use
        
    Returns:
        Configured IndexingPipeline
    """
    try:
        import openai
    except ImportError:
        raise ImportError("Install openai package: pip install openai")
    
    api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key required")
    
    client = openai.OpenAI(api_key=api_key)
    
    def embed_fn(text: str) -> List[float]:
        response = client.embeddings.create(
            input=text[:8000],  # Truncate to avoid token limits
            model=model
        )
        return response.data[0].embedding
    
    pipeline = IndexingPipeline(vocabulary_path=vocabulary_path)
    pipeline.set_embedding_function(embed_fn)
    
    return pipeline


def create_pipeline_with_sentence_transformers(
    vocabulary_path: str,
    model_name: str = "all-MiniLM-L6-v2"
) -> IndexingPipeline:
    """
    Create a pipeline with local sentence-transformer embeddings
    
    Args:
        vocabulary_path: Path to vocabulary JSON
        model_name: Sentence transformer model name
        
    Returns:
        Configured IndexingPipeline
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    
    model = SentenceTransformer(model_name)
    
    def embed_fn(text: str) -> List[float]:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    pipeline = IndexingPipeline(vocabulary_path=vocabulary_path)
    pipeline.set_embedding_function(embed_fn)
    
    return pipeline
