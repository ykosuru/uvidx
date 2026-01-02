"""
Local Embeddings Module - Domain-aware embeddings without external APIs

This module provides lightweight, local embedding methods optimized for
payment systems domain retrieval. No external API calls or large model
downloads required.

Approaches:
1. DomainAwareEmbedder: TF-IDF weighted by domain vocabulary importance
2. HashEmbedder: Feature hashing for fixed-dimension sparse embeddings
3. HybridEmbedder: Combines domain concepts + text features
4. LSHEmbedder: Locality-sensitive hashing for approximate similarity
"""

import math
import re
import hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from .vocabulary import DomainVocabulary


# ============================================================
# Text Preprocessing
# ============================================================

class TextPreprocessor:
    """Shared text preprocessing utilities"""
    
    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
        'the', 'to', 'was', 'were', 'will', 'with', 'this', 'but', 'they',
        'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'can', 'should', 'now', 'i', 'you',
        'we', 'our', 'your', 'if', 'then', 'else', 'end', 'begin', 'proc',
        'int', 'string', 'call', 'return'  # TAL keywords to ignore
    }
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', text.lower())
        return tokens
    
    @staticmethod
    def tokenize_with_bigrams(text: str) -> List[str]:
        """Tokenize with unigrams and bigrams"""
        tokens = TextPreprocessor.tokenize(text)
        # Add bigrams
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        return tokens + bigrams
    
    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [t for t in tokens if t not in TextPreprocessor.STOPWORDS]
    
    @staticmethod
    def stem(word: str) -> str:
        """Simple suffix stripping stemmer"""
        # Simple rules for common suffixes
        if len(word) > 6:
            if word.endswith('ation'):
                return word[:-5]
            if word.endswith('ment'):
                return word[:-4]
            if word.endswith('ing'):
                return word[:-3]
            if word.endswith('ed'):
                return word[:-2]
            if word.endswith('ly'):
                return word[:-2]
            if word.endswith('er'):
                return word[:-2]
            if word.endswith('es'):
                return word[:-2]
            if word.endswith('s') and not word.endswith('ss'):
                return word[:-1]
        return word


# ============================================================
# TF-IDF Embedder
# ============================================================

@dataclass
class TFIDFConfig:
    """Configuration for TF-IDF embedder"""
    use_idf: bool = True
    use_bigrams: bool = True
    remove_stopwords: bool = True
    use_stemming: bool = True
    normalize: bool = True
    sublinear_tf: bool = True  # Use 1 + log(tf) instead of tf
    min_df: int = 1  # Minimum document frequency
    max_features: int = 10000  # Max vocabulary size


class TFIDFEmbedder:
    """
    TF-IDF based embeddings with domain vocabulary boosting.
    
    Features:
    - Builds vocabulary from corpus
    - Weights domain terms higher
    - Produces sparse or dense vectors
    """
    
    def __init__(self, 
                 config: Optional[TFIDFConfig] = None,
                 domain_vocabulary: Optional[DomainVocabulary] = None,
                 domain_boost: float = 2.0):
        """
        Initialize TF-IDF embedder.
        
        Args:
            config: TF-IDF configuration
            domain_vocabulary: Domain vocabulary for term boosting
            domain_boost: Multiplier for domain terms
        """
        self.config = config or TFIDFConfig()
        self.domain_vocab = domain_vocabulary
        self.domain_boost = domain_boost
        
        # Built from corpus
        self.vocabulary: Dict[str, int] = {}  # term -> index
        self.idf: Dict[str, float] = {}  # term -> idf score
        self.doc_count = 0
        
        # Domain term set for fast lookup
        self.domain_terms: Set[str] = set()
        if domain_vocabulary:
            for term in domain_vocabulary.term_to_entry.keys():
                # Add both full term and individual words
                self.domain_terms.add(term.lower())
                for word in term.lower().split():
                    self.domain_terms.add(word)
    
    def _preprocess(self, text: str) -> List[str]:
        """Preprocess text into tokens"""
        if self.config.use_bigrams:
            tokens = TextPreprocessor.tokenize_with_bigrams(text)
        else:
            tokens = TextPreprocessor.tokenize(text)
        
        if self.config.remove_stopwords:
            tokens = TextPreprocessor.remove_stopwords(tokens)
        
        if self.config.use_stemming:
            tokens = [TextPreprocessor.stem(t) for t in tokens]
        
        return tokens
    
    def fit(self, documents: List[str]):
        """
        Build vocabulary and IDF from corpus.
        
        Args:
            documents: List of document texts
        """
        self.doc_count = len(documents)
        term_doc_freq: Dict[str, int] = defaultdict(int)
        term_total_freq: Dict[str, int] = defaultdict(int)
        
        # Count term frequencies
        for doc in documents:
            tokens = self._preprocess(doc)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                term_doc_freq[token] += 1
            
            for token in tokens:
                term_total_freq[token] += 1
        
        # Filter by min_df and build vocabulary
        valid_terms = [
            (term, freq) for term, freq in term_total_freq.items()
            if term_doc_freq[term] >= self.config.min_df
        ]
        
        # Sort by frequency and take top max_features
        valid_terms.sort(key=lambda x: -x[1])
        valid_terms = valid_terms[:self.config.max_features]
        
        # Build vocabulary index
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(valid_terms)}
        
        # Compute IDF
        if self.config.use_idf:
            for term in self.vocabulary:
                df = term_doc_freq[term]
                # Smooth IDF: log((N + 1) / (df + 1)) + 1
                self.idf[term] = math.log((self.doc_count + 1) / (df + 1)) + 1
        else:
            self.idf = {term: 1.0 for term in self.vocabulary}
    
    def transform(self, text: str) -> np.ndarray:
        """
        Transform text to TF-IDF vector.
        
        Args:
            text: Input text
            
        Returns:
            Dense numpy array of TF-IDF features
        """
        if not self.vocabulary:
            raise ValueError("Embedder not fitted. Call fit() first or use fit_transform().")
        
        tokens = self._preprocess(text)
        tf = Counter(tokens)
        
        # Create vector
        vector = np.zeros(len(self.vocabulary))
        
        for term, count in tf.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                
                # Compute TF
                if self.config.sublinear_tf:
                    tf_score = 1 + math.log(count) if count > 0 else 0
                else:
                    tf_score = count
                
                # Apply IDF
                idf_score = self.idf.get(term, 1.0)
                
                # Apply domain boost
                if term in self.domain_terms:
                    boost = self.domain_boost
                else:
                    boost = 1.0
                
                vector[idx] = tf_score * idf_score * boost
        
        # Normalize
        if self.config.normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        return vector
    
    def fit_transform(self, documents: List[str]) -> List[np.ndarray]:
        """Fit and transform in one step"""
        self.fit(documents)
        return [self.transform(doc) for doc in documents]
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding as list (compatible with pipeline interface)"""
        return self.transform(text).tolist()


# ============================================================
# Hash Embedder (Feature Hashing)
# ============================================================

class HashEmbedder:
    """
    Feature hashing embedder for fixed-dimension sparse embeddings.
    
    Advantages:
    - No vocabulary to build/store
    - Fixed dimension regardless of corpus
    - Handles out-of-vocabulary terms
    - Very fast
    
    Uses the "hashing trick" to map terms to fixed-size vectors.
    """
    
    def __init__(self,
                 n_features: int = 1024,
                 use_bigrams: bool = True,
                 domain_vocabulary: Optional[DomainVocabulary] = None,
                 domain_boost: float = 2.0):
        """
        Initialize hash embedder.
        
        Args:
            n_features: Dimension of output vectors
            use_bigrams: Include bigrams in features
            domain_vocabulary: Domain vocabulary for boosting
            domain_boost: Multiplier for domain terms
        """
        self.n_features = n_features
        self.use_bigrams = use_bigrams
        self.domain_vocab = domain_vocabulary
        self.domain_boost = domain_boost
        
        # Domain term set
        self.domain_terms: Set[str] = set()
        if domain_vocabulary:
            for term in domain_vocabulary.term_to_entry.keys():
                self.domain_terms.add(term.lower())
                for word in term.lower().split():
                    self.domain_terms.add(word)
    
    def _hash_term(self, term: str) -> Tuple[int, int]:
        """
        Hash term to (index, sign).
        
        Uses two hash functions:
        - One for the index
        - One for the sign (to reduce collision effects)
        """
        # Primary hash for index
        h1 = int(hashlib.md5(term.encode()).hexdigest(), 16)
        idx = h1 % self.n_features
        
        # Secondary hash for sign
        h2 = int(hashlib.sha1(term.encode()).hexdigest(), 16)
        sign = 1 if h2 % 2 == 0 else -1
        
        return idx, sign
    
    def transform(self, text: str) -> np.ndarray:
        """
        Transform text to hash-based embedding.
        
        Args:
            text: Input text
            
        Returns:
            Dense numpy array
        """
        # Tokenize
        tokens = TextPreprocessor.tokenize(text)
        tokens = TextPreprocessor.remove_stopwords(tokens)
        
        if self.use_bigrams:
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            tokens = tokens + bigrams
        
        # Count terms
        tf = Counter(tokens)
        
        # Create vector
        vector = np.zeros(self.n_features)
        
        for term, count in tf.items():
            idx, sign = self._hash_term(term)
            
            # Apply domain boost
            if term in self.domain_terms or term.replace('_', ' ') in self.domain_terms:
                boost = self.domain_boost
            else:
                boost = 1.0
            
            # Use log TF
            tf_score = 1 + math.log(count) if count > 0 else 0
            vector[idx] += sign * tf_score * boost
        
        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding as list"""
        return self.transform(text).tolist()


# ============================================================
# Domain Concept Embedder
# ============================================================

class DomainConceptEmbedder:
    """
    Embedder based purely on domain concept matching.
    
    Creates embeddings where each dimension represents a domain concept
    or business capability. Very interpretable and domain-focused.
    """
    
    def __init__(self, 
                 domain_vocabulary: DomainVocabulary,
                 include_capabilities: bool = True,
                 include_categories: bool = True):
        """
        Initialize domain concept embedder.
        
        Args:
            domain_vocabulary: Domain vocabulary (required)
            include_capabilities: Include capability dimensions
            include_categories: Include category dimensions
        """
        self.domain_vocab = domain_vocabulary
        self.include_capabilities = include_capabilities
        self.include_categories = include_categories
        
        # Build dimension mapping
        self.concept_to_idx: Dict[str, int] = {}
        self._build_dimensions()
    
    def _build_dimensions(self):
        """Build the dimension mapping from concepts"""
        idx = 0
        
        # Add canonical terms
        for entry in self.domain_vocab.entries:
            if entry.canonical_term not in self.concept_to_idx:
                self.concept_to_idx[entry.canonical_term.lower()] = idx
                idx += 1
        
        # Add capabilities
        if self.include_capabilities:
            for entry in self.domain_vocab.entries:
                for cap in entry.business_capabilities:
                    if cap.lower() not in self.concept_to_idx:
                        self.concept_to_idx[cap.lower()] = idx
                        idx += 1
        
        # Add categories
        if self.include_categories:
            for entry in self.domain_vocab.entries:
                cat = entry.metadata_category.lower()
                if cat not in self.concept_to_idx:
                    self.concept_to_idx[cat] = idx
                    idx += 1
    
    @property
    def n_dimensions(self) -> int:
        """Number of dimensions in embedding"""
        return len(self.concept_to_idx)
    
    def transform(self, text: str) -> np.ndarray:
        """
        Transform text to concept-based embedding.
        
        Args:
            text: Input text
            
        Returns:
            Dense numpy array with concept activations
        """
        vector = np.zeros(self.n_dimensions)
        
        # Match domain concepts
        matches = self.domain_vocab.match_text(text)
        
        for match in matches:
            # Activate canonical term dimension
            term_key = match.canonical_term.lower()
            if term_key in self.concept_to_idx:
                vector[self.concept_to_idx[term_key]] += match.confidence
            
            # Activate capability dimensions
            if self.include_capabilities:
                for cap in match.capabilities:
                    cap_key = cap.lower()
                    if cap_key in self.concept_to_idx:
                        vector[self.concept_to_idx[cap_key]] += match.confidence * 0.5
            
            # Activate category dimension
            if self.include_categories:
                cat_key = match.category.lower()
                if cat_key in self.concept_to_idx:
                    vector[self.concept_to_idx[cat_key]] += match.confidence * 0.3
        
        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding as list"""
        return self.transform(text).tolist()
    
    def explain_embedding(self, text: str) -> Dict[str, float]:
        """
        Get human-readable explanation of embedding.
        
        Returns:
            Dict mapping concept names to their activation values
        """
        vector = self.transform(text)
        
        explanation = {}
        for concept, idx in self.concept_to_idx.items():
            if vector[idx] > 0:
                explanation[concept] = float(vector[idx])
        
        return dict(sorted(explanation.items(), key=lambda x: -x[1]))


# ============================================================
# Hybrid Embedder
# ============================================================

class HybridEmbedder:
    """
    Combines multiple embedding approaches for robust retrieval.
    
    Concatenates:
    1. Domain concept embeddings (interpretable, domain-focused)
    2. TF-IDF or hash embeddings (captures full text)
    
    This gives best of both worlds:
    - Strong domain term matching
    - General text similarity
    """
    
    def __init__(self,
                 domain_vocabulary: DomainVocabulary,
                 text_embedder: str = "hash",  # "tfidf" or "hash"
                 text_dim: int = 512,
                 domain_weight: float = 0.6,
                 text_weight: float = 0.4):
        """
        Initialize hybrid embedder.
        
        Args:
            domain_vocabulary: Domain vocabulary
            text_embedder: Type of text embedder ("tfidf" or "hash")
            text_dim: Dimension for text embeddings
            domain_weight: Weight for domain component
            text_weight: Weight for text component
        """
        self.domain_weight = domain_weight
        self.text_weight = text_weight
        
        # Domain embedder
        self.domain_embedder = DomainConceptEmbedder(domain_vocabulary)
        
        # Text embedder
        if text_embedder == "hash":
            self.text_embedder = HashEmbedder(
                n_features=text_dim,
                domain_vocabulary=domain_vocabulary
            )
        else:
            self.text_embedder = TFIDFEmbedder(
                config=TFIDFConfig(max_features=text_dim),
                domain_vocabulary=domain_vocabulary
            )
        
        self._fitted = text_embedder == "hash"  # Hash doesn't need fitting
    
    def fit(self, documents: List[str]):
        """Fit the text embedder (if TF-IDF)"""
        if hasattr(self.text_embedder, 'fit'):
            self.text_embedder.fit(documents)
        self._fitted = True
    
    @property
    def n_dimensions(self) -> int:
        """Total embedding dimensions"""
        domain_dim = self.domain_embedder.n_dimensions
        if hasattr(self.text_embedder, 'n_features'):
            text_dim = self.text_embedder.n_features
        else:
            text_dim = len(self.text_embedder.vocabulary) if self.text_embedder.vocabulary else 0
        return domain_dim + text_dim
    
    def transform(self, text: str) -> np.ndarray:
        """
        Transform text to hybrid embedding.
        
        Args:
            text: Input text
            
        Returns:
            Concatenated domain + text embedding
        """
        # Get component embeddings
        domain_vec = self.domain_embedder.transform(text) * self.domain_weight
        text_vec = self.text_embedder.transform(text) * self.text_weight
        
        # Concatenate
        return np.concatenate([domain_vec, text_vec])
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding as list"""
        return self.transform(text).tolist()


# ============================================================
# BM25 Embedder
# ============================================================

class BM25Embedder:
    """
    BM25-based embeddings optimized for retrieval.
    
    BM25 is a ranking function that works well for information retrieval.
    This creates dense embeddings using BM25 term weights.
    """
    
    def __init__(self,
                 k1: float = 1.5,
                 b: float = 0.75,
                 domain_vocabulary: Optional[DomainVocabulary] = None,
                 domain_boost: float = 2.0):
        """
        Initialize BM25 embedder.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            domain_vocabulary: Domain vocabulary for boosting
            domain_boost: Multiplier for domain terms
        """
        self.k1 = k1
        self.b = b
        self.domain_vocab = domain_vocabulary
        self.domain_boost = domain_boost
        
        # Built from corpus
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avg_doc_len: float = 0
        self.doc_count: int = 0
        
        # Domain terms
        self.domain_terms: Set[str] = set()
        if domain_vocabulary:
            for term in domain_vocabulary.term_to_entry.keys():
                self.domain_terms.add(term.lower())
                for word in term.lower().split():
                    self.domain_terms.add(word)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        tokens = TextPreprocessor.tokenize(text)
        tokens = TextPreprocessor.remove_stopwords(tokens)
        return tokens
    
    def fit(self, documents: List[str]):
        """Fit BM25 on corpus"""
        self.doc_count = len(documents)
        term_doc_freq: Dict[str, int] = defaultdict(int)
        all_tokens: List[List[str]] = []
        total_len = 0
        
        for doc in documents:
            tokens = self._tokenize(doc)
            all_tokens.append(tokens)
            total_len += len(tokens)
            
            for token in set(tokens):
                term_doc_freq[token] += 1
        
        self.avg_doc_len = total_len / len(documents) if documents else 1
        
        # Build vocabulary
        self.vocabulary = {term: idx for idx, term in enumerate(term_doc_freq.keys())}
        
        # Compute IDF (BM25 variant)
        for term, df in term_doc_freq.items():
            # IDF = log((N - df + 0.5) / (df + 0.5))
            self.idf[term] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
    
    def transform(self, text: str) -> np.ndarray:
        """Transform text to BM25 embedding"""
        if not self.vocabulary:
            raise ValueError("Embedder not fitted")
        
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        doc_len = len(tokens)
        
        vector = np.zeros(len(self.vocabulary))
        
        for term, count in tf.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                idf = self.idf.get(term, 0)
                
                # BM25 term weight
                numerator = count * (self.k1 + 1)
                denominator = count + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                weight = idf * numerator / denominator
                
                # Domain boost
                if term in self.domain_terms:
                    weight *= self.domain_boost
                
                vector[idx] = weight
        
        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding as list"""
        return self.transform(text).tolist()


# ============================================================
# Factory Functions
# ============================================================

def create_embedder(
    embedder_type: str = "hybrid",
    domain_vocabulary: Optional[DomainVocabulary] = None,
    **kwargs
) -> Any:
    """
    Factory function to create embedders.
    
    Args:
        embedder_type: One of "hybrid", "tfidf", "hash", "domain", "bm25"
        domain_vocabulary: Domain vocabulary (required for some types)
        **kwargs: Additional arguments for the embedder
        
    Returns:
        Configured embedder instance
    """
    if embedder_type == "hybrid":
        if not domain_vocabulary:
            raise ValueError("domain_vocabulary required for hybrid embedder")
        return HybridEmbedder(domain_vocabulary, **kwargs)
    
    elif embedder_type == "tfidf":
        config = TFIDFConfig(**{k: v for k, v in kwargs.items() if hasattr(TFIDFConfig, k)})
        return TFIDFEmbedder(config, domain_vocabulary, kwargs.get('domain_boost', 2.0))
    
    elif embedder_type == "hash":
        return HashEmbedder(
            n_features=kwargs.get('n_features', 1024),
            domain_vocabulary=domain_vocabulary,
            domain_boost=kwargs.get('domain_boost', 2.0)
        )
    
    elif embedder_type == "domain":
        if not domain_vocabulary:
            raise ValueError("domain_vocabulary required for domain embedder")
        return DomainConceptEmbedder(domain_vocabulary, **kwargs)
    
    elif embedder_type == "bm25":
        return BM25Embedder(
            domain_vocabulary=domain_vocabulary,
            domain_boost=kwargs.get('domain_boost', 2.0)
        )
    
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


# ============================================================
# Similarity Functions
# ============================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot / (norm1 * norm2))


def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple vectors.
    
    Args:
        query: Query vector (1D)
        vectors: Matrix of vectors (2D, each row is a vector)
        
    Returns:
        Array of similarity scores
    """
    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(vectors))
    query_normalized = query / query_norm
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    vectors_normalized = vectors / norms
    
    # Dot product
    return np.dot(vectors_normalized, query_normalized)
