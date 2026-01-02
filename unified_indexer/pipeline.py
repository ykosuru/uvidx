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
from .embeddings import (
    create_embedder,
    HybridEmbedder,
    TFIDFEmbedder,
    HashEmbedder,
    DomainConceptEmbedder,
    BM25Embedder
)


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
    
    Supports local embeddings (no external APIs needed):
    - HybridEmbedder: Domain concepts + text features
    - TFIDFEmbedder: TF-IDF with domain boosting
    - HashEmbedder: Feature hashing (no fitting needed)
    - DomainConceptEmbedder: Pure domain concept matching
    - BM25Embedder: BM25 ranking-based embeddings
    """
    
    def __init__(self, 
                 vocabulary_path: Optional[str] = None,
                 vocabulary_data: Optional[List[Dict]] = None,
                 embedding_fn: Optional[Callable[[str], List[float]]] = None,
                 embedder_type: Optional[str] = "hash",
                 tal_parser_path: Optional[str] = None):
        """
        Initialize the indexing pipeline
        
        Args:
            vocabulary_path: Path to vocabulary JSON file
            vocabulary_data: Vocabulary as list of dicts (alternative to path)
            embedding_fn: Custom function to generate embeddings (overrides embedder_type)
            embedder_type: Type of local embedder to use:
                - "hash": Feature hashing (default, no fitting needed)
                - "hybrid": Domain + text hybrid
                - "tfidf": TF-IDF (requires fitting)
                - "domain": Pure domain concepts
                - "bm25": BM25 ranking (requires fitting)
                - None: No embeddings (concept-only search)
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
        
        # Initialize local embedder if no custom function provided
        self.embedder = None
        self.embedder_type = embedder_type
        
        if embedding_fn:
            # Use custom embedding function
            embed_fn = embedding_fn
        elif embedder_type:
            # Create local embedder
            self.embedder = create_embedder(
                embedder_type=embedder_type,
                domain_vocabulary=self.vocabulary
            )
            embed_fn = self.embedder.get_embedding
        else:
            embed_fn = None
        
        # Initialize index
        self.index = HybridIndex(self.vocabulary, embed_fn)
        
        # Statistics
        self.stats = PipelineStatistics()
        
        # Track if embedder needs fitting
        self._embedder_fitted = embedder_type in ["hash", "domain", None]
    
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
    
    def set_embedder(self, 
                     embedder_type: str,
                     fit_documents: Optional[List[str]] = None,
                     **kwargs):
        """
        Set a local embedder.
        
        Args:
            embedder_type: Type of embedder ("hash", "hybrid", "tfidf", "domain", "bm25")
            fit_documents: Documents to fit the embedder on (for tfidf/bm25)
            **kwargs: Additional embedder arguments
        """
        self.embedder = create_embedder(
            embedder_type=embedder_type,
            domain_vocabulary=self.vocabulary,
            **kwargs
        )
        self.embedder_type = embedder_type
        
        # Fit if needed
        if fit_documents and hasattr(self.embedder, 'fit'):
            self.embedder.fit(fit_documents)
            self._embedder_fitted = True
        else:
            self._embedder_fitted = embedder_type in ["hash", "domain"]
        
        # Update index
        self.index.set_embedding_function(self.embedder.get_embedding)
    
    def fit_embedder(self, documents: List[str]):
        """
        Fit the embedder on a corpus of documents.
        
        Required for TF-IDF and BM25 embedders before indexing.
        
        Args:
            documents: List of document texts to fit on
        """
        if self.embedder and hasattr(self.embedder, 'fit'):
            self.embedder.fit(documents)
            self._embedder_fitted = True
    
    def fit_embedder_from_chunks(self, chunks: List[IndexableChunk]):
        """
        Fit the embedder using already parsed chunks.
        
        Args:
            chunks: List of IndexableChunk objects
        """
        documents = [chunk.embedding_text or chunk.text for chunk in chunks]
        self.fit_embedder(documents)
    
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
        
        # Get embedder dimensions if available
        embedder_dims = None
        if self.embedder:
            if hasattr(self.embedder, 'n_features'):
                embedder_dims = self.embedder.n_features
            elif hasattr(self.embedder, 'output_dim'):
                embedder_dims = self.embedder.output_dim
        
        with open(path / 'pipeline_stats.json', 'w') as f:
            json.dump({
                'files_processed': self.stats.files_processed,
                'files_failed': self.stats.files_failed,
                'total_chunks': self.stats.total_chunks,
                'by_source_type': self.stats.by_source_type,
                'processing_time_seconds': self.stats.processing_time_seconds,
                'embedder_type': self.embedder_type,
                'embedder_dims': embedder_dims,
                'saved_at': datetime.now().isoformat()
            }, f, indent=2)
    
    def load(self, directory: str):
        """
        Load the index from disk
        
        Args:
            directory: Directory containing saved index
        """
        self.index.load(directory)
        
        path = Path(directory)
        
        # Load vocabulary if saved
        vocab_path = path / 'vocabulary.json'
        if vocab_path.exists():
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            self.vocabulary.load_from_data(vocab_data)
        
        # Load pipeline stats and restore embedder type
        stats_path = path / 'pipeline_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                saved_stats = json.load(f)
            
            # Restore embedder type
            saved_embedder_type = saved_stats.get('embedder_type')
            if saved_embedder_type:
                print(f"Restoring embedder type: {saved_embedder_type}")
                self.set_embedder(saved_embedder_type)
        
        # Ensure embedding function is set if we have an embedder
        if self.embedder and not self.index.embedding_fn:
            self.index.set_embedding_function(self.embedder.get_embedding)
    
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


# ============================================================
# LLM Integration Stubs
# ============================================================

class LLMInterface:
    """
    Stub interface for LLM invocation.
    
    Users should subclass this and implement the invoke_llm method
    with their preferred LLM provider (OpenAI, Anthropic, local models, etc.)
    """
    
    def invoke_llm(self,
                   user_prompt: str,
                   system_prompt: str = "",
                   content_type: str = "text") -> str:
        """
        Invoke an LLM with the given prompts.
        
        Args:
            user_prompt: The main prompt/question for the LLM
            system_prompt: System-level instructions (optional)
            content_type: Type of content being processed. One of:
                - "text": General text processing
                - "code": Code analysis/generation
                - "embedding": Text for embedding generation
                - "extraction": Information extraction
                - "summarization": Content summarization
                - "classification": Content classification
                
        Returns:
            The LLM's response as a string
            
        Example implementation:
            def invoke_llm(self, user_prompt, system_prompt="", content_type="text"):
                # Your LLM API call here
                response = your_llm_client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.content
        """
        raise NotImplementedError(
            "Subclass LLMInterface and implement invoke_llm() with your LLM provider. "
            "See docstring for expected signature and example implementation."
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Example implementation:
            def generate_embedding(self, text):
                response = your_embedding_client.embed(text)
                return response.embedding
        """
        raise NotImplementedError(
            "Subclass LLMInterface and implement generate_embedding() with your embedding provider."
        )


class LLMEnhancedPipeline(IndexingPipeline):
    """
    Pipeline with LLM enhancement capabilities.
    
    Extends the base IndexingPipeline with LLM-powered features:
    - Semantic chunk enhancement
    - Query understanding
    - Result summarization
    - Cross-reference explanation
    
    Usage:
        class MyLLM(LLMInterface):
            def invoke_llm(self, user_prompt, system_prompt="", content_type="text"):
                # Your implementation
                pass
            
            def generate_embedding(self, text):
                # Your implementation
                pass
        
        llm = MyLLM()
        pipeline = LLMEnhancedPipeline(
            vocabulary_path="vocab.json",
            llm_interface=llm
        )
    """
    
    def __init__(self,
                 vocabulary_path: Optional[str] = None,
                 vocabulary_data: Optional[List[Dict]] = None,
                 llm_interface: Optional[LLMInterface] = None,
                 tal_parser_path: Optional[str] = None):
        """
        Initialize the LLM-enhanced pipeline.
        
        Args:
            vocabulary_path: Path to vocabulary JSON file
            vocabulary_data: Vocabulary as list of dicts
            llm_interface: LLMInterface implementation for LLM calls
            tal_parser_path: Path to TAL parser modules
        """
        # Initialize with embedding function from LLM interface if provided
        embedding_fn = None
        if llm_interface:
            try:
                # Test if generate_embedding is implemented
                llm_interface.generate_embedding("test")
                embedding_fn = llm_interface.generate_embedding
            except NotImplementedError:
                pass
        
        super().__init__(
            vocabulary_path=vocabulary_path,
            vocabulary_data=vocabulary_data,
            embedding_fn=embedding_fn,
            tal_parser_path=tal_parser_path
        )
        
        self.llm = llm_interface
    
    def set_llm_interface(self, llm_interface: LLMInterface):
        """Set or update the LLM interface"""
        self.llm = llm_interface
        
        # Also set embedding function if available
        try:
            llm_interface.generate_embedding("test")
            self.set_embedding_function(llm_interface.generate_embedding)
        except NotImplementedError:
            pass
    
    def enhance_query(self, query: str) -> str:
        """
        Use LLM to enhance/expand the search query.
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced query with expanded terms
        """
        if not self.llm:
            return query
        
        system_prompt = """You are a search query enhancer for a payment systems codebase.
Given a user query, expand it with relevant technical terms, acronyms, and related concepts.
Return only the enhanced query, no explanations."""
        
        user_prompt = f"Enhance this search query for a payment systems codebase: {query}"
        
        try:
            enhanced = self.llm.invoke_llm(user_prompt, system_prompt, "text")
            return enhanced.strip()
        except NotImplementedError:
            return query
    
    def summarize_results(self, 
                          query: str, 
                          results: List[SearchResult],
                          max_results: int = 5) -> str:
        """
        Use LLM to summarize search results.
        
        Args:
            query: The original search query
            results: Search results to summarize
            max_results: Maximum results to include in summary
            
        Returns:
            Natural language summary of results
        """
        if not self.llm:
            return f"Found {len(results)} results for '{query}'"
        
        # Build context from results
        result_texts = []
        for i, r in enumerate(results[:max_results], 1):
            result_texts.append(
                f"{i}. [{r.chunk.source_type.value}] {r.chunk.source_ref}\n"
                f"   Concepts: {', '.join(r.matched_concepts)}\n"
                f"   Content: {r.chunk.text[:200]}..."
            )
        
        system_prompt = """You are a helpful assistant summarizing search results from a payment systems codebase.
Provide a concise summary of what was found and how it relates to the query."""
        
        user_prompt = f"""Query: {query}

Results:
{chr(10).join(result_texts)}

Summarize these results in 2-3 sentences."""
        
        try:
            return self.llm.invoke_llm(user_prompt, system_prompt, "summarization")
        except NotImplementedError:
            return f"Found {len(results)} results for '{query}'"
    
    def explain_code(self, 
                     chunk: IndexableChunk,
                     context: str = "") -> str:
        """
        Use LLM to explain a code chunk.
        
        Args:
            chunk: Code chunk to explain
            context: Additional context
            
        Returns:
            Natural language explanation
        """
        if not self.llm or chunk.source_type != SourceType.CODE:
            return ""
        
        system_prompt = """You are an expert in legacy payment systems code (TAL, COBOL).
Explain the given code clearly and concisely, focusing on its business purpose."""
        
        concepts = ", ".join(m.canonical_term for m in chunk.domain_matches)
        user_prompt = f"""Explain this code:

```
{chunk.text}
```

Domain concepts found: {concepts}
{f'Additional context: {context}' if context else ''}

Provide a brief explanation of what this code does."""
        
        try:
            return self.llm.invoke_llm(user_prompt, system_prompt, "code")
        except NotImplementedError:
            return ""
    
    def extract_business_rules(self, 
                               chunk: IndexableChunk) -> List[str]:
        """
        Use LLM to extract business rules from code.
        
        Args:
            chunk: Code chunk to analyze
            
        Returns:
            List of extracted business rules
        """
        if not self.llm or chunk.source_type != SourceType.CODE:
            return []
        
        system_prompt = """You are an expert in extracting business rules from payment systems code.
Identify and list the business rules embedded in the code.
Return one rule per line, no numbering or bullets."""
        
        user_prompt = f"""Extract business rules from this code:

```
{chunk.text}
```

List each business rule on a separate line."""
        
        try:
            response = self.llm.invoke_llm(user_prompt, system_prompt, "extraction")
            rules = [r.strip() for r in response.split('\n') if r.strip()]
            return rules
        except NotImplementedError:
            return []


# ============================================================
# Example LLM Implementation Template
# ============================================================

class ExampleLLMImplementation(LLMInterface):
    """
    Example template for implementing LLMInterface.
    
    Copy and modify this for your specific LLM provider.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize your LLM client here.
        
        Example:
            self.client = YourLLMClient(api_key=api_key)
            self.model = model or "default-model"
        """
        self.api_key = api_key
        self.model = model
        # self.client = YourLLMClient(api_key=api_key)
    
    def invoke_llm(self,
                   user_prompt: str,
                   system_prompt: str = "",
                   content_type: str = "text") -> str:
        """
        Implement your LLM invocation here.
        
        Example for OpenAI-style API:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content
            
        Example for Anthropic-style API:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        """
        # TODO: Implement with your LLM provider
        raise NotImplementedError("Implement invoke_llm with your LLM provider")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Implement your embedding generation here.
        
        Example for OpenAI:
            response = self.client.embeddings.create(
                input=text[:8000],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
            
        Example for sentence-transformers:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        """
        # TODO: Implement with your embedding provider
        raise NotImplementedError("Implement generate_embedding with your embedding provider")
