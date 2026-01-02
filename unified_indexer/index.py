"""
Hybrid Index - Combines vector similarity with concept-based retrieval

This index stores chunks in both a vector store (for semantic search)
and a concept index (for exact domain term matching). Search results
are fused using reciprocal rank fusion.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from .models import (
    IndexableChunk,
    SearchResult,
    SourceType,
    DomainMatch
)
from .vocabulary import DomainVocabulary


class VectorStore:
    """
    Simple in-memory vector store for embeddings
    
    For production, replace with ChromaDB, Qdrant, Pinecone, etc.
    """
    
    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}
        self.chunks: Dict[str, IndexableChunk] = {}
    
    def add(self, chunk_id: str, embedding: List[float], chunk: IndexableChunk):
        """Add a chunk with its embedding"""
        self.embeddings[chunk_id] = np.array(embedding)
        self.chunks[chunk_id] = chunk
    
    def search(self, 
               query_embedding: List[float], 
               top_k: int = 10,
               filter_fn: Optional[Callable[[IndexableChunk], bool]] = None) -> List[Tuple[str, float]]:
        """
        Search for similar chunks
        
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if not self.embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        
        # Compute similarities
        results = []
        for chunk_id, embedding in self.embeddings.items():
            # Apply filter if provided
            if filter_fn and not filter_fn(self.chunks[chunk_id]):
                continue
            
            # Cosine similarity
            similarity = np.dot(query_vec, embedding) / (
                np.linalg.norm(query_vec) * np.linalg.norm(embedding) + 1e-8
            )
            results.append((chunk_id, float(similarity)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_chunk(self, chunk_id: str) -> Optional[IndexableChunk]:
        """Get chunk by ID"""
        return self.chunks.get(chunk_id)
    
    def __len__(self):
        return len(self.chunks)


class ConceptIndex:
    """
    Index for exact domain concept matching
    
    Maps canonical terms and business capabilities to chunk IDs
    for fast exact-match retrieval.
    """
    
    def __init__(self):
        # concept (canonical term) -> set of chunk_ids
        self.concept_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        
        # capability -> set of chunk_ids  
        self.capability_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        
        # category -> set of chunk_ids
        self.category_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        
        # source_type -> set of chunk_ids
        self.source_type_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        
        # All chunks
        self.chunks: Dict[str, IndexableChunk] = {}
    
    def add(self, chunk: IndexableChunk):
        """Index a chunk by its domain concepts"""
        chunk_id = chunk.chunk_id
        self.chunks[chunk_id] = chunk
        
        # Index by source type
        self.source_type_to_chunks[chunk.source_type.value].add(chunk_id)
        
        # Index by domain matches
        for match in chunk.domain_matches:
            # By canonical term
            canonical_lower = match.canonical_term.lower()
            self.concept_to_chunks[canonical_lower].add(chunk_id)
            
            # By capability
            for capability in match.capabilities:
                self.capability_to_chunks[capability].add(chunk_id)
            
            # By category
            self.category_to_chunks[match.category].add(chunk_id)
    
    def search_concept(self, concept: str) -> Set[str]:
        """Find chunks containing a specific concept"""
        return self.concept_to_chunks.get(concept.lower(), set())
    
    def search_capability(self, capability: str) -> Set[str]:
        """Find chunks for a business capability"""
        return self.capability_to_chunks.get(capability, set())
    
    def search_category(self, category: str) -> Set[str]:
        """Find chunks in a metadata category"""
        return self.category_to_chunks.get(category, set())
    
    def search_source_type(self, source_type: str) -> Set[str]:
        """Find chunks by source type"""
        return self.source_type_to_chunks.get(source_type, set())
    
    def get_chunk(self, chunk_id: str) -> Optional[IndexableChunk]:
        """Get chunk by ID"""
        return self.chunks.get(chunk_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_chunks': len(self.chunks),
            'unique_concepts': len(self.concept_to_chunks),
            'unique_capabilities': len(self.capability_to_chunks),
            'unique_categories': len(self.category_to_chunks),
            'chunks_by_source': {
                st: len(chunks) 
                for st, chunks in self.source_type_to_chunks.items()
            }
        }
    
    def __len__(self):
        return len(self.chunks)


class HybridIndex:
    """
    Hybrid retrieval index combining vector and concept search
    
    Features:
    - Vector similarity search for semantic matching
    - Concept index for exact domain term matching
    - Reciprocal rank fusion for result combination
    - Filtering by source type, capability, etc.
    """
    
    def __init__(self, 
                 vocabulary: DomainVocabulary,
                 embedding_fn: Optional[Callable[[str], List[float]]] = None):
        """
        Initialize hybrid index
        
        Args:
            vocabulary: Domain vocabulary for query expansion
            embedding_fn: Function to generate embeddings (text -> vector)
        """
        self.vocabulary = vocabulary
        self.embedding_fn = embedding_fn
        
        self.vector_store = VectorStore()
        self.concept_index = ConceptIndex()
        
        # Index metadata
        self.metadata: Dict[str, Any] = {
            'total_indexed': 0,
            'by_source_type': defaultdict(int)
        }
    
    def set_embedding_function(self, fn: Callable[[str], List[float]]):
        """Set the embedding function for vector search"""
        self.embedding_fn = fn
    
    def index_chunk(self, chunk: IndexableChunk):
        """
        Add a chunk to both indexes
        
        Args:
            chunk: Chunk to index
        """
        # Add to concept index
        self.concept_index.add(chunk)
        
        # Add to vector store if embedding function available
        if self.embedding_fn:
            try:
                embedding = self.embedding_fn(chunk.embedding_text)
                chunk.embedding = embedding
                self.vector_store.add(chunk.chunk_id, embedding, chunk)
            except Exception as e:
                print(f"Warning: Failed to embed chunk {chunk.chunk_id}: {e}")
        
        # Update metadata
        self.metadata['total_indexed'] += 1
        self.metadata['by_source_type'][chunk.source_type.value] += 1
    
    def index_chunks(self, chunks: List[IndexableChunk], batch_size: int = 100):
        """
        Index multiple chunks
        
        Args:
            chunks: List of chunks to index
            batch_size: Batch size for embedding (if applicable)
        """
        for chunk in chunks:
            self.index_chunk(chunk)
    
    def search(self,
               query: str,
               top_k: int = 10,
               source_types: Optional[List[SourceType]] = None,
               capabilities: Optional[List[str]] = None,
               vector_weight: float = 0.5,
               concept_weight: float = 0.5) -> List[SearchResult]:
        """
        Hybrid search combining vector and concept matching
        
        Args:
            query: Search query
            top_k: Number of results to return
            source_types: Filter by source types (None = all)
            capabilities: Filter by business capabilities (None = all)
            vector_weight: Weight for vector search scores
            concept_weight: Weight for concept match scores
            
        Returns:
            List of SearchResult objects ranked by combined score
        """
        # Extract concepts from query
        query_concepts = self.vocabulary.match_text(query, deduplicate=True)
        
        # Expand query with synonyms
        expanded_terms = self.vocabulary.expand_query(query)
        
        # ========== Vector Search ==========
        vector_results: Dict[str, float] = {}
        
        if self.embedding_fn and len(self.vector_store) > 0:
            # Create filter function
            def filter_fn(chunk: IndexableChunk) -> bool:
                if source_types and chunk.source_type not in source_types:
                    return False
                if capabilities:
                    chunk_caps = chunk.capability_set
                    if not any(cap in chunk_caps for cap in capabilities):
                        return False
                return True
            
            # Search vector store
            query_embedding = self.embedding_fn(query)
            vector_hits = self.vector_store.search(
                query_embedding, 
                top_k=top_k * 3,  # Get more for fusion
                filter_fn=filter_fn
            )
            
            for chunk_id, score in vector_hits:
                vector_results[chunk_id] = score
        
        # ========== Concept Search ==========
        concept_results: Dict[str, float] = {}
        
        # Search by extracted concepts
        for concept_match in query_concepts:
            chunk_ids = self.concept_index.search_concept(concept_match.canonical_term)
            for chunk_id in chunk_ids:
                if chunk_id not in concept_results:
                    concept_results[chunk_id] = 0.0
                concept_results[chunk_id] += 1.0
        
        # Search by expanded terms
        for term in expanded_terms:
            entry = self.vocabulary.get_entry_by_term(term)
            if entry:
                chunk_ids = self.concept_index.search_concept(entry.canonical_term)
                for chunk_id in chunk_ids:
                    if chunk_id not in concept_results:
                        concept_results[chunk_id] = 0.0
                    concept_results[chunk_id] += 0.5  # Lower weight for expanded terms
        
        # Search by capability filter
        if capabilities:
            for capability in capabilities:
                chunk_ids = self.concept_index.search_capability(capability)
                for chunk_id in chunk_ids:
                    if chunk_id not in concept_results:
                        concept_results[chunk_id] = 0.0
                    concept_results[chunk_id] += 0.5
        
        # Apply source type filter to concept results
        if source_types:
            allowed_chunks = set()
            for st in source_types:
                allowed_chunks.update(
                    self.concept_index.search_source_type(st.value)
                )
            concept_results = {
                cid: score for cid, score in concept_results.items()
                if cid in allowed_chunks
            }
        
        # Normalize concept scores
        if concept_results:
            max_concept_score = max(concept_results.values())
            if max_concept_score > 0:
                concept_results = {
                    cid: score / max_concept_score 
                    for cid, score in concept_results.items()
                }
        
        # ========== Result Fusion ==========
        all_chunk_ids = set(vector_results.keys()) | set(concept_results.keys())
        
        fused_results = []
        for chunk_id in all_chunk_ids:
            v_score = vector_results.get(chunk_id, 0.0)
            c_score = concept_results.get(chunk_id, 0.0)
            
            # Weighted combination
            combined_score = (vector_weight * v_score) + (concept_weight * c_score)
            
            # Get chunk from either store
            chunk = (
                self.vector_store.get_chunk(chunk_id) or 
                self.concept_index.get_chunk(chunk_id)
            )
            
            if chunk:
                # Determine matched concepts
                matched_concepts = []
                matched_capabilities = []
                
                for qc in query_concepts:
                    for cm in chunk.domain_matches:
                        if cm.canonical_term.lower() == qc.canonical_term.lower():
                            matched_concepts.append(cm.canonical_term)
                            matched_capabilities.extend(cm.capabilities)
                
                # Determine retrieval method
                if v_score > 0 and c_score > 0:
                    method = "hybrid"
                elif v_score > 0:
                    method = "vector"
                else:
                    method = "concept"
                
                result = SearchResult(
                    chunk=chunk,
                    vector_score=v_score,
                    concept_score=c_score,
                    combined_score=combined_score,
                    matched_concepts=list(set(matched_concepts)),
                    matched_capabilities=list(set(matched_capabilities)),
                    retrieval_method=method
                )
                fused_results.append(result)
        
        # Sort by combined score
        fused_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(fused_results[:top_k]):
            result.rank = i + 1
        
        return fused_results[:top_k]
    
    def search_by_capability(self, 
                             capability: str,
                             top_k: int = 10) -> List[SearchResult]:
        """
        Find all chunks related to a business capability
        
        Args:
            capability: Business capability name
            top_k: Maximum results
            
        Returns:
            List of SearchResult objects
        """
        chunk_ids = self.concept_index.search_capability(capability)
        
        results = []
        for chunk_id in list(chunk_ids)[:top_k]:
            chunk = self.concept_index.get_chunk(chunk_id)
            if chunk:
                results.append(SearchResult(
                    chunk=chunk,
                    concept_score=1.0,
                    combined_score=1.0,
                    matched_capabilities=[capability],
                    retrieval_method="concept"
                ))
        
        return results
    
    def search_cross_reference(self,
                               query: str,
                               source_type: SourceType,
                               reference_types: List[SourceType],
                               top_k: int = 5) -> Dict[str, List[SearchResult]]:
        """
        Find related content across different source types
        
        Example: Find code that handles errors from logs
        
        Args:
            query: Search query
            source_type: Primary source type to search
            reference_types: Related source types to cross-reference
            top_k: Results per source type
            
        Returns:
            Dict mapping source type to results
        """
        # Search primary source type
        primary_results = self.search(
            query, 
            top_k=top_k, 
            source_types=[source_type]
        )
        
        results = {source_type.value: primary_results}
        
        # Extract concepts from primary results
        all_concepts = set()
        all_capabilities = set()
        
        for result in primary_results:
            all_concepts.update(result.chunk.canonical_terms)
            all_capabilities.update(result.chunk.capability_set)
        
        # Search reference types using extracted concepts
        for ref_type in reference_types:
            ref_results = []
            
            # Search by each concept
            for concept in list(all_concepts)[:5]:  # Limit concepts
                concept_hits = self.concept_index.search_concept(concept)
                
                for chunk_id in concept_hits:
                    chunk = self.concept_index.get_chunk(chunk_id)
                    if chunk and chunk.source_type == ref_type:
                        ref_results.append(SearchResult(
                            chunk=chunk,
                            concept_score=1.0,
                            combined_score=1.0,
                            matched_concepts=[concept],
                            retrieval_method="cross_reference"
                        ))
            
            # Deduplicate and limit
            seen_ids = set()
            unique_results = []
            for r in ref_results:
                if r.chunk.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk.chunk_id)
                    unique_results.append(r)
            
            results[ref_type.value] = unique_results[:top_k]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            **self.metadata,
            'vector_store_size': len(self.vector_store),
            'concept_index': self.concept_index.get_statistics()
        }
    
    def save(self, directory: str):
        """
        Save index to disk
        
        Args:
            directory: Directory to save index files
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        chunks_data = [
            chunk.to_dict() 
            for chunk in self.concept_index.chunks.values()
        ]
        
        with open(path / 'chunks.json', 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        # Save embeddings if available
        if self.vector_store.embeddings:
            embeddings_data = {
                chunk_id: embedding.tolist()
                for chunk_id, embedding in self.vector_store.embeddings.items()
            }
            with open(path / 'embeddings.json', 'w') as f:
                json.dump(embeddings_data, f)
        
        # Save metadata
        with open(path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Index saved to {directory}")
    
    def load(self, directory: str):
        """
        Load index from disk
        
        Args:
            directory: Directory containing index files
        """
        path = Path(directory)
        
        if not path.exists():
            raise FileNotFoundError(f"Index directory not found: {directory}")
        
        # Load chunks
        with open(path / 'chunks.json', 'r') as f:
            chunks_data = json.load(f)
        
        for chunk_data in chunks_data:
            chunk = IndexableChunk.from_dict(chunk_data)
            self.concept_index.add(chunk)
        
        # Load embeddings if available
        embeddings_path = path / 'embeddings.json'
        if embeddings_path.exists():
            with open(embeddings_path, 'r') as f:
                embeddings_data = json.load(f)
            
            for chunk_id, embedding in embeddings_data.items():
                chunk = self.concept_index.get_chunk(chunk_id)
                if chunk:
                    chunk.embedding = embedding
                    self.vector_store.add(chunk_id, embedding, chunk)
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Index loaded from {directory}: {len(self.concept_index)} chunks")
