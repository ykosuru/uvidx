"""
Learned Domain Embeddings

Automatically discovers semantic dimensions from a corpus of documents.
Instead of hardcoded dimensions, this learns the domain structure by:

1. Extracting significant terms from documents
2. Building co-occurrence patterns
3. Clustering related terms into dimensions
4. Learning term-to-dimension weights

This adapts to any domain - payments, insurance, healthcare, etc.
"""

import re
import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum


@dataclass
class LearnedDimension:
    """A learned semantic dimension"""
    id: int
    name: str                          # Primary term representing this dimension
    terms: List[str]                   # All terms in this dimension
    term_weights: Dict[str, float]     # Term -> weight within dimension
    document_frequency: int            # How many docs contain this dimension
    coherence_score: float             # How coherent/tight the cluster is
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'terms': self.terms,
            'term_weights': self.term_weights,
            'document_frequency': self.document_frequency,
            'coherence_score': self.coherence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LearnedDimension':
        return cls(
            id=data['id'],
            name=data['name'],
            terms=data['terms'],
            term_weights=data['term_weights'],
            document_frequency=data['document_frequency'],
            coherence_score=data['coherence_score']
        )


@dataclass
class LearningConfig:
    """Configuration for dimension learning"""
    
    # Term extraction
    min_term_frequency: int = 3        # Minimum times a term must appear
    max_term_frequency_pct: float = 0.8  # Max % of docs (filter common words)
    min_term_length: int = 2           # Minimum term length
    max_terms: int = 1000              # Maximum terms to consider
    
    # N-gram extraction
    extract_bigrams: bool = True       # Extract 2-word phrases
    extract_trigrams: bool = True      # Extract 3-word phrases
    min_ngram_frequency: int = 3       # Minimum n-gram frequency
    
    # Dimension learning
    n_dimensions: int = 80             # Target number of dimensions
    min_dimension_terms: int = 3       # Minimum terms per dimension
    max_dimension_terms: int = 50      # Maximum terms per dimension
    
    # Co-occurrence
    cooccurrence_window: int = 50      # Window size for co-occurrence (chars)
    min_cooccurrence: int = 2          # Minimum co-occurrences to consider
    
    # Clustering
    clustering_method: str = "hierarchical"  # "hierarchical" or "kmeans"
    similarity_threshold: float = 0.3  # For hierarchical clustering


class TermExtractor:
    """Extract significant terms from documents"""
    
    def __init__(self, config: LearningConfig):
        self.config = config
        
        # Common stop words to filter
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'it', 'its', 'if', 'then', 'else',
            'when', 'where', 'which', 'who', 'what', 'how', 'why', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
            'now', 'here', 'there', 'any', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further',
            'once', 'end', 'begin', 'return', 'int', 'string', 'proc', 'call',
            'true', 'false', 'null', 'none', 'def', 'class', 'function', 'var',
            'let', 'const', 'import', 'from', 'export', 'default', 'new', 'delete',
            'type', 'interface', 'enum', 'struct', 'void', 'public', 'private',
            'static', 'final', 'abstract', 'extends', 'implements', 'override'
        }
    
    def extract_terms(self, documents: List[str]) -> Dict[str, Dict]:
        """
        Extract significant terms from corpus.
        
        Returns:
            Dict mapping term -> {frequency, doc_frequency, tfidf, positions}
        """
        term_stats = defaultdict(lambda: {
            'frequency': 0,
            'doc_frequency': 0,
            'documents': set(),
            'positions': []  # (doc_idx, char_pos)
        })
        
        n_docs = len(documents)
        
        for doc_idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            doc_terms = set()
            
            # Extract single terms
            for match in re.finditer(r'\b([a-z][a-z0-9_\-\.]*[a-z0-9]|[a-z])\b', doc_lower):
                term = match.group(1)
                
                if self._is_valid_term(term):
                    term_stats[term]['frequency'] += 1
                    term_stats[term]['positions'].append((doc_idx, match.start()))
                    doc_terms.add(term)
            
            # Extract bigrams
            if self.config.extract_bigrams:
                for match in re.finditer(
                    r'\b([a-z][a-z0-9]*)\s+([a-z][a-z0-9]*)\b', 
                    doc_lower
                ):
                    t1, t2 = match.group(1), match.group(2)
                    if self._is_valid_term(t1) and self._is_valid_term(t2):
                        bigram = f"{t1} {t2}"
                        term_stats[bigram]['frequency'] += 1
                        term_stats[bigram]['positions'].append((doc_idx, match.start()))
                        doc_terms.add(bigram)
            
            # Extract trigrams
            if self.config.extract_trigrams:
                for match in re.finditer(
                    r'\b([a-z][a-z0-9]*)\s+([a-z][a-z0-9]*)\s+([a-z][a-z0-9]*)\b',
                    doc_lower
                ):
                    t1, t2, t3 = match.group(1), match.group(2), match.group(3)
                    if (self._is_valid_term(t1) and self._is_valid_term(t2) 
                        and self._is_valid_term(t3)):
                        trigram = f"{t1} {t2} {t3}"
                        term_stats[trigram]['frequency'] += 1
                        term_stats[trigram]['positions'].append((doc_idx, match.start()))
                        doc_terms.add(trigram)
            
            # Update document frequency
            for term in doc_terms:
                term_stats[term]['doc_frequency'] += 1
                term_stats[term]['documents'].add(doc_idx)
        
        # Filter and compute TF-IDF
        filtered_terms = {}
        max_doc_freq = n_docs * self.config.max_term_frequency_pct
        
        for term, stats in term_stats.items():
            freq = stats['frequency']
            doc_freq = stats['doc_frequency']
            
            # Apply filters
            if freq < self.config.min_term_frequency:
                continue
            if doc_freq > max_doc_freq:
                continue
            if ' ' not in term and freq < self.config.min_ngram_frequency:
                # Stricter filter for n-grams
                pass
            
            # Compute TF-IDF
            tf = 1 + math.log(freq) if freq > 0 else 0
            idf = math.log(n_docs / (1 + doc_freq))
            tfidf = tf * idf
            
            filtered_terms[term] = {
                'frequency': freq,
                'doc_frequency': doc_freq,
                'tfidf': tfidf,
                'positions': stats['positions']
            }
        
        # Sort by TF-IDF and take top terms
        sorted_terms = sorted(
            filtered_terms.items(),
            key=lambda x: x[1]['tfidf'],
            reverse=True
        )[:self.config.max_terms]
        
        return dict(sorted_terms)
    
    def _is_valid_term(self, term: str) -> bool:
        """Check if term is valid"""
        if len(term) < self.config.min_term_length:
            return False
        if term in self.stop_words:
            return False
        if term.isdigit():
            return False
        return True


class CooccurrenceAnalyzer:
    """Analyze term co-occurrence patterns"""
    
    def __init__(self, config: LearningConfig):
        self.config = config
    
    def build_cooccurrence_matrix(
        self, 
        documents: List[str],
        terms: Dict[str, Dict]
    ) -> Dict[Tuple[str, str], int]:
        """
        Build co-occurrence counts for term pairs.
        
        Returns:
            Dict mapping (term1, term2) -> co-occurrence count
        """
        cooccurrence = defaultdict(int)
        term_set = set(terms.keys())
        window = self.config.cooccurrence_window
        
        for doc_idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            
            # Find all term positions in this document
            term_positions = []
            for term in term_set:
                for match in re.finditer(re.escape(term), doc_lower):
                    term_positions.append((match.start(), term))
            
            # Sort by position
            term_positions.sort(key=lambda x: x[0])
            
            # Count co-occurrences within window
            for i, (pos1, term1) in enumerate(term_positions):
                for j in range(i + 1, len(term_positions)):
                    pos2, term2 = term_positions[j]
                    
                    if pos2 - pos1 > window:
                        break
                    
                    if term1 != term2:
                        # Normalize order for symmetric counting
                        pair = tuple(sorted([term1, term2]))
                        cooccurrence[pair] += 1
        
        # Filter by minimum co-occurrence
        filtered = {
            pair: count 
            for pair, count in cooccurrence.items()
            if count >= self.config.min_cooccurrence
        }
        
        return filtered
    
    def compute_similarity_matrix(
        self,
        terms: List[str],
        cooccurrence: Dict[Tuple[str, str], int],
        term_stats: Dict[str, Dict]
    ) -> np.ndarray:
        """
        Compute similarity matrix based on co-occurrence.
        
        Uses PMI (Pointwise Mutual Information) for similarity.
        """
        n_terms = len(terms)
        term_to_idx = {t: i for i, t in enumerate(terms)}
        
        # Total co-occurrences for normalization
        total_cooc = sum(cooccurrence.values())
        term_freqs = {t: term_stats[t]['frequency'] for t in terms}
        total_freq = sum(term_freqs.values())
        
        similarity = np.zeros((n_terms, n_terms))
        
        for (t1, t2), cooc_count in cooccurrence.items():
            if t1 not in term_to_idx or t2 not in term_to_idx:
                continue
            
            i, j = term_to_idx[t1], term_to_idx[t2]
            
            # PMI = log(P(t1,t2) / (P(t1) * P(t2)))
            p_joint = cooc_count / total_cooc if total_cooc > 0 else 0
            p_t1 = term_freqs[t1] / total_freq if total_freq > 0 else 0
            p_t2 = term_freqs[t2] / total_freq if total_freq > 0 else 0
            
            if p_joint > 0 and p_t1 > 0 and p_t2 > 0:
                pmi = math.log(p_joint / (p_t1 * p_t2))
                # Normalize PMI to [0, 1]
                npmi = pmi / (-math.log(p_joint)) if p_joint < 1 else 0
                npmi = max(0, npmi)  # Clip negative values
                
                similarity[i, j] = npmi
                similarity[j, i] = npmi
        
        return similarity


class DimensionClusterer:
    """Cluster terms into semantic dimensions"""
    
    def __init__(self, config: LearningConfig):
        self.config = config
    
    def cluster_hierarchical(
        self,
        terms: List[str],
        similarity: np.ndarray,
        term_stats: Dict[str, Dict]
    ) -> List[LearnedDimension]:
        """
        Hierarchical agglomerative clustering of terms.
        """
        n_terms = len(terms)
        n_dims = min(self.config.n_dimensions, n_terms // 2)
        
        # Convert similarity to distance
        distance = 1 - similarity
        np.fill_diagonal(distance, 0)
        
        # Initialize: each term is its own cluster
        clusters = [[i] for i in range(n_terms)]
        cluster_distances = distance.copy()
        
        # Agglomerate until we have n_dims clusters
        while len(clusters) > n_dims:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Average linkage
                    dist = np.mean([
                        distance[ti, tj]
                        for ti in clusters[i]
                        for tj in clusters[j]
                    ])
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # Convert clusters to dimensions
        dimensions = []
        for dim_id, cluster in enumerate(clusters):
            cluster_terms = [terms[i] for i in cluster]
            
            # Compute term weights based on TF-IDF
            term_weights = {}
            max_tfidf = max(term_stats[t]['tfidf'] for t in cluster_terms)
            
            for term in cluster_terms:
                weight = term_stats[term]['tfidf'] / max_tfidf if max_tfidf > 0 else 1.0
                term_weights[term] = weight
            
            # Name dimension by highest TF-IDF term
            name_term = max(cluster_terms, key=lambda t: term_stats[t]['tfidf'])
            
            # Compute coherence (average intra-cluster similarity)
            if len(cluster) > 1:
                coherence = np.mean([
                    similarity[i, j]
                    for i in cluster
                    for j in cluster
                    if i != j
                ])
            else:
                coherence = 1.0
            
            # Document frequency for dimension
            doc_freq = len(set().union(*[
                set(pos[0] for pos in term_stats[t]['positions'])
                for t in cluster_terms
            ]))
            
            dimensions.append(LearnedDimension(
                id=dim_id,
                name=name_term.upper().replace(' ', '_'),
                terms=cluster_terms,
                term_weights=term_weights,
                document_frequency=doc_freq,
                coherence_score=float(coherence)
            ))
        
        # Sort by document frequency (most common first)
        dimensions.sort(key=lambda d: d.document_frequency, reverse=True)
        
        # Reassign IDs after sorting
        for i, dim in enumerate(dimensions):
            dim.id = i
        
        return dimensions
    
    def cluster_kmeans(
        self,
        terms: List[str],
        similarity: np.ndarray,
        term_stats: Dict[str, Dict]
    ) -> List[LearnedDimension]:
        """
        K-means clustering of terms based on similarity.
        """
        n_terms = len(terms)
        n_dims = min(self.config.n_dimensions, n_terms // 2)
        
        # Use similarity matrix as feature vectors
        features = similarity.copy()
        
        # Simple k-means implementation
        # Initialize centroids randomly
        np.random.seed(42)
        centroid_indices = np.random.choice(n_terms, n_dims, replace=False)
        centroids = features[centroid_indices].copy()
        
        # Iterate
        for _ in range(100):
            # Assign terms to nearest centroid
            assignments = np.argmin(
                np.linalg.norm(features[:, np.newaxis] - centroids, axis=2),
                axis=1
            )
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_dims):
                mask = assignments == k
                if np.any(mask):
                    new_centroids[k] = features[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Build dimensions from clusters
        dimensions = []
        for dim_id in range(n_dims):
            cluster_indices = np.where(assignments == dim_id)[0]
            if len(cluster_indices) == 0:
                continue
            
            cluster_terms = [terms[i] for i in cluster_indices]
            
            # Compute term weights
            term_weights = {}
            max_tfidf = max(term_stats[t]['tfidf'] for t in cluster_terms)
            for term in cluster_terms:
                weight = term_stats[term]['tfidf'] / max_tfidf if max_tfidf > 0 else 1.0
                term_weights[term] = weight
            
            # Name by highest TF-IDF
            name_term = max(cluster_terms, key=lambda t: term_stats[t]['tfidf'])
            
            # Coherence
            if len(cluster_indices) > 1:
                coherence = np.mean([
                    similarity[i, j]
                    for i in cluster_indices
                    for j in cluster_indices
                    if i != j
                ])
            else:
                coherence = 1.0
            
            # Doc frequency
            doc_freq = len(set().union(*[
                set(pos[0] for pos in term_stats[t]['positions'])
                for t in cluster_terms
            ]))
            
            dimensions.append(LearnedDimension(
                id=len(dimensions),
                name=name_term.upper().replace(' ', '_'),
                terms=cluster_terms,
                term_weights=term_weights,
                document_frequency=doc_freq,
                coherence_score=float(coherence)
            ))
        
        return dimensions


class LearnedDomainEmbedder:
    """
    Embedder that learns domain dimensions from a corpus.
    
    Usage:
        embedder = LearnedDomainEmbedder()
        embedder.fit(documents)  # Learn dimensions
        embedding = embedder.get_embedding(text)
        embedder.save("dimensions.json")
        
        # Later:
        embedder2 = LearnedDomainEmbedder.load("dimensions.json")
    """
    
    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        
        self.dimensions: List[LearnedDimension] = []
        self.term_to_dimensions: Dict[str, List[Tuple[int, float]]] = {}
        self._fitted = False
    
    @property
    def n_dimensions(self) -> int:
        return len(self.dimensions)
    
    @property
    def n_features(self) -> int:
        return self.n_dimensions
    
    def fit(self, documents: List[str], verbose: bool = True) -> 'LearnedDomainEmbedder':
        """
        Learn dimensions from a corpus of documents.
        
        Args:
            documents: List of document texts
            verbose: Print progress
            
        Returns:
            self (for chaining)
        """
        if verbose:
            print(f"Learning dimensions from {len(documents)} documents...")
        
        # Step 1: Extract terms
        if verbose:
            print("  Extracting terms...")
        extractor = TermExtractor(self.config)
        term_stats = extractor.extract_terms(documents)
        
        if verbose:
            print(f"  Found {len(term_stats)} significant terms")
        
        if len(term_stats) < self.config.n_dimensions:
            print(f"  Warning: Only {len(term_stats)} terms found, reducing dimensions")
            self.config.n_dimensions = max(10, len(term_stats) // 2)
        
        # Step 2: Build co-occurrence matrix
        if verbose:
            print("  Building co-occurrence matrix...")
        cooc_analyzer = CooccurrenceAnalyzer(self.config)
        cooccurrence = cooc_analyzer.build_cooccurrence_matrix(documents, term_stats)
        
        if verbose:
            print(f"  Found {len(cooccurrence)} co-occurrence pairs")
        
        # Step 3: Compute similarity matrix
        if verbose:
            print("  Computing similarity matrix...")
        terms = list(term_stats.keys())
        similarity = cooc_analyzer.compute_similarity_matrix(terms, cooccurrence, term_stats)
        
        # Step 4: Cluster into dimensions
        if verbose:
            print(f"  Clustering into {self.config.n_dimensions} dimensions...")
        clusterer = DimensionClusterer(self.config)
        
        if self.config.clustering_method == "hierarchical":
            self.dimensions = clusterer.cluster_hierarchical(terms, similarity, term_stats)
        else:
            self.dimensions = clusterer.cluster_kmeans(terms, similarity, term_stats)
        
        # Step 5: Build term lookup
        self._build_term_lookup()
        
        self._fitted = True
        
        if verbose:
            print(f"  Created {len(self.dimensions)} dimensions")
            print("\n  Top 10 dimensions:")
            for dim in self.dimensions[:10]:
                top_terms = sorted(dim.term_weights.items(), key=lambda x: -x[1])[:5]
                terms_str = ", ".join(t for t, _ in top_terms)
                print(f"    {dim.name}: {terms_str}")
        
        return self
    
    def _build_term_lookup(self):
        """Build term -> [(dim_id, weight), ...] lookup"""
        self.term_to_dimensions = defaultdict(list)
        
        for dim in self.dimensions:
            for term, weight in dim.term_weights.items():
                self.term_to_dimensions[term].append((dim.id, weight))
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using learned dimensions.
        
        Returns:
            Vector of dimension activations
        """
        if not self._fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")
        
        vector = np.zeros(len(self.dimensions))
        text_lower = text.lower()
        
        # Match terms and activate dimensions
        for term, dim_weights in self.term_to_dimensions.items():
            if term in text_lower:
                for dim_id, weight in dim_weights:
                    vector[dim_id] = max(vector[dim_id], weight)
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def transform(self, text: str) -> np.ndarray:
        """Alias returning numpy array"""
        return np.array(self.get_embedding(text))
    
    def explain_embedding(self, text: str, top_k: int = 10) -> List[Tuple[str, float, List[str]]]:
        """
        Explain which dimensions are activated.
        
        Returns:
            List of (dimension_name, weight, matched_terms)
        """
        if not self._fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")
        
        embedding = self.transform(text)
        text_lower = text.lower()
        
        # Find which terms matched for each dimension
        dim_matched_terms = defaultdict(list)
        for term, dim_weights in self.term_to_dimensions.items():
            if term in text_lower:
                for dim_id, weight in dim_weights:
                    dim_matched_terms[dim_id].append(term)
        
        # Get top activated dimensions
        top_indices = np.argsort(embedding)[::-1][:top_k]
        
        explanations = []
        for idx in top_indices:
            if embedding[idx] > 0:
                dim = self.dimensions[idx]
                matched = dim_matched_terms.get(idx, [])
                explanations.append((dim.name, float(embedding[idx]), matched))
        
        return explanations
    
    def get_dimension_info(self) -> List[Dict]:
        """Get information about all learned dimensions"""
        return [
            {
                'id': dim.id,
                'name': dim.name,
                'n_terms': len(dim.terms),
                'top_terms': sorted(dim.term_weights.items(), key=lambda x: -x[1])[:10],
                'doc_frequency': dim.document_frequency,
                'coherence': dim.coherence_score
            }
            for dim in self.dimensions
        ]
    
    def save(self, path: str):
        """Save learned dimensions to file"""
        data = {
            'config': {
                'n_dimensions': self.config.n_dimensions,
                'min_term_frequency': self.config.min_term_frequency,
                'cooccurrence_window': self.config.cooccurrence_window,
                'clustering_method': self.config.clustering_method
            },
            'dimensions': [d.to_dict() for d in self.dimensions]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.dimensions)} dimensions to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LearnedDomainEmbedder':
        """Load learned dimensions from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = LearningConfig(**data.get('config', {}))
        embedder = cls(config)
        
        embedder.dimensions = [
            LearnedDimension.from_dict(d) 
            for d in data['dimensions']
        ]
        embedder._build_term_lookup()
        embedder._fitted = True
        
        print(f"Loaded {len(embedder.dimensions)} dimensions from {path}")
        return embedder


class HybridLearnedEmbedder:
    """
    Combines learned domain dimensions with text embeddings.
    
    Total dimensions: n_learned + text_dim
    """
    
    def __init__(self,
                 learned_embedder: LearnedDomainEmbedder,
                 text_dim: int = 512,
                 learned_weight: float = 0.7,
                 text_weight: float = 0.3):
        """
        Initialize hybrid embedder.
        
        Args:
            learned_embedder: Pre-fitted LearnedDomainEmbedder
            text_dim: Dimension for text embeddings
            learned_weight: Weight for learned component
            text_weight: Weight for text component
        """
        self.learned_embedder = learned_embedder
        self.text_embedder = None  # Lazy load
        self._text_dim = text_dim
        
        self.learned_weight = learned_weight
        self.text_weight = text_weight
    
    def _get_text_embedder(self):
        if self.text_embedder is None:
            from .embeddings import HashEmbedder
            self.text_embedder = HashEmbedder(n_features=self._text_dim)
        return self.text_embedder
    
    @property
    def n_dimensions(self) -> int:
        return self.learned_embedder.n_dimensions + self._text_dim
    
    @property
    def n_features(self) -> int:
        return self.n_dimensions
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate hybrid embedding"""
        learned_vec = self.learned_embedder.transform(text) * self.learned_weight
        text_vec = self._get_text_embedder().transform(text) * self.text_weight
        
        combined = np.concatenate([learned_vec, text_vec])
        return combined.tolist()
    
    def transform(self, text: str) -> np.ndarray:
        return np.array(self.get_embedding(text))
    
    def explain_embedding(self, text: str, top_k: int = 10):
        """Explain learned dimensions"""
        return self.learned_embedder.explain_embedding(text, top_k)


# ============================================================
# Convenience Functions
# ============================================================

def learn_dimensions_from_files(
    file_paths: List[str],
    output_path: str,
    n_dimensions: int = 80,
    **kwargs
) -> LearnedDomainEmbedder:
    """
    Learn dimensions from a list of files.
    
    Args:
        file_paths: List of file paths to learn from
        output_path: Where to save learned dimensions
        n_dimensions: Target number of dimensions
        **kwargs: Additional config options
        
    Returns:
        Fitted LearnedDomainEmbedder
    """
    documents = []
    
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                documents.append(f.read())
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
    
    config = LearningConfig(n_dimensions=n_dimensions, **kwargs)
    embedder = LearnedDomainEmbedder(config)
    embedder.fit(documents)
    embedder.save(output_path)
    
    return embedder


def learn_dimensions_from_directory(
    directory: str,
    output_path: str,
    extensions: List[str] = ['.txt', '.tal', '.pdf', '.md'],
    n_dimensions: int = 80,
    recursive: bool = True,
    **kwargs
) -> LearnedDomainEmbedder:
    """
    Learn dimensions from all files in a directory.
    
    Args:
        directory: Directory to scan
        output_path: Where to save learned dimensions
        extensions: File extensions to include
        n_dimensions: Target number of dimensions
        recursive: Scan subdirectories
        **kwargs: Additional config options
        
    Returns:
        Fitted LearnedDomainEmbedder
    """
    from pathlib import Path
    
    dir_path = Path(directory)
    pattern = '**/*' if recursive else '*'
    
    file_paths = []
    for ext in extensions:
        file_paths.extend(str(p) for p in dir_path.glob(f'{pattern}{ext}'))
    
    print(f"Found {len(file_paths)} files to analyze")
    return learn_dimensions_from_files(file_paths, output_path, n_dimensions, **kwargs)
