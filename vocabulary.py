"""
Domain Vocabulary - Fast multi-pattern matching for payment systems concepts

Uses Aho-Corasick automaton for O(n) scanning of all vocabulary terms
simultaneously. This is critical for performance when processing
thousands of code procedures or log entries.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .models import VocabularyEntry, DomainMatch


class AhoCorasickNode:
    """Node in the Aho-Corasick automaton"""
    def __init__(self):
        self.children: Dict[str, 'AhoCorasickNode'] = {}
        self.failure: Optional['AhoCorasickNode'] = None
        self.output: List[Tuple[str, VocabularyEntry]] = []  # (matched_term, entry)


class AhoCorasickAutomaton:
    """
    Aho-Corasick automaton for efficient multi-pattern matching
    
    Allows scanning text once to find all vocabulary term matches
    in O(n + m) time where n is text length and m is total matches.
    """
    
    def __init__(self):
        self.root = AhoCorasickNode()
        self._built = False
    
    def add_pattern(self, pattern: str, entry: VocabularyEntry):
        """Add a pattern to the automaton"""
        node = self.root
        pattern_lower = pattern.lower().strip()
        
        for char in pattern_lower:
            if char not in node.children:
                node.children[char] = AhoCorasickNode()
            node = node.children[char]
        
        node.output.append((pattern, entry))
        self._built = False
    
    def build(self):
        """Build failure links using BFS"""
        from collections import deque
        
        queue = deque()
        
        # Initialize depth-1 nodes
        for char, child in self.root.children.items():
            child.failure = self.root
            queue.append(child)
        
        # BFS to build failure links
        while queue:
            current = queue.popleft()
            
            for char, child in current.children.items():
                queue.append(child)
                
                # Find failure link
                failure = current.failure
                while failure and char not in failure.children:
                    failure = failure.failure
                
                child.failure = failure.children[char] if failure else self.root
                
                # Merge output from failure link
                if child.failure:
                    child.output = child.output + child.failure.output
        
        self._built = True
    
    def search(self, text: str) -> List[Tuple[int, int, str, VocabularyEntry]]:
        """
        Search text for all pattern matches
        
        Returns:
            List of (start_pos, end_pos, matched_term, vocabulary_entry)
        """
        if not self._built:
            self.build()
        
        results = []
        node = self.root
        text_lower = text.lower()
        
        for i, char in enumerate(text_lower):
            # Follow failure links until match or root
            while node and char not in node.children:
                node = node.failure
            
            if not node:
                node = self.root
                continue
            
            node = node.children[char]
            
            # Collect all matches at this position
            for pattern, entry in node.output:
                start = i - len(pattern) + 1
                end = i + 1
                results.append((start, end, pattern, entry))
        
        return results


class DomainVocabulary:
    """
    Domain vocabulary for payment systems concept matching
    
    Loads the vocabulary JSON and builds an efficient matcher
    for extracting domain concepts from any text content.
    """
    
    def __init__(self, vocabulary_path: Optional[str] = None):
        """
        Initialize the domain vocabulary
        
        Args:
            vocabulary_path: Path to vocabulary JSON file
        """
        self.entries: List[VocabularyEntry] = []
        self.automaton = AhoCorasickAutomaton()
        
        # Indexes for fast lookup
        self.by_canonical: Dict[str, VocabularyEntry] = {}
        self.by_capability: Dict[str, List[VocabularyEntry]] = defaultdict(list)
        self.by_category: Dict[str, List[VocabularyEntry]] = defaultdict(list)
        
        # Term to entry mapping (for quick lookup after matching)
        self.term_to_entry: Dict[str, VocabularyEntry] = {}
        
        if vocabulary_path:
            self.load(vocabulary_path)
    
    def load(self, vocabulary_path: str):
        """Load vocabulary from JSON file"""
        path = Path(vocabulary_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocabulary_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._build_from_data(data)
    
    def load_from_data(self, data: List[Dict[str, Any]]):
        """Load vocabulary from parsed JSON data"""
        self._build_from_data(data)
    
    def _build_from_data(self, data: List[Dict[str, Any]]):
        """Build vocabulary structures from data"""
        self.entries = []
        self.automaton = AhoCorasickAutomaton()
        self.by_canonical = {}
        self.by_capability = defaultdict(list)
        self.by_category = defaultdict(list)
        self.term_to_entry = {}
        
        for item in data:
            entry = VocabularyEntry.from_dict(item)
            self.entries.append(entry)
            
            # Index by canonical term
            self.by_canonical[entry.canonical_term.lower()] = entry
            
            # Index by capability
            for capability in entry.business_capabilities:
                self.by_capability[capability].append(entry)
            
            # Index by category
            self.by_category[entry.metadata_category].append(entry)
            
            # Add all terms to automaton
            for term in entry.all_terms():
                term = term.strip()
                if term and len(term) >= 2:  # Skip very short terms
                    self.automaton.add_pattern(term, entry)
                    self.term_to_entry[term.lower()] = entry
        
        # Build the automaton
        self.automaton.build()
        
        print(f"Loaded vocabulary: {len(self.entries)} entries, "
              f"{len(self.term_to_entry)} searchable terms")
    
    def match_text(self, text: str, 
                   context_window: int = 50,
                   deduplicate: bool = True) -> List[DomainMatch]:
        """
        Find all domain concept matches in text
        
        Args:
            text: Text to scan for domain concepts
            context_window: Characters of context to capture around match
            deduplicate: Remove duplicate matches of same canonical term
            
        Returns:
            List of DomainMatch objects
        """
        raw_matches = self.automaton.search(text)
        
        matches = []
        seen_canonical = set()
        
        for start, end, matched_term, entry in raw_matches:
            # Skip if deduplicating and we've seen this canonical term
            if deduplicate:
                if entry.canonical_term.lower() in seen_canonical:
                    continue
                seen_canonical.add(entry.canonical_term.lower())
            
            # Extract context
            context_start = max(0, start - context_window)
            context_end = min(len(text), end + context_window)
            context = text[context_start:context_end]
            
            match = DomainMatch(
                matched_term=matched_term,
                canonical_term=entry.canonical_term,
                capabilities=entry.business_capabilities.copy(),
                category=entry.metadata_category,
                span=(start, end),
                confidence=1.0,
                context=context
            )
            matches.append(match)
        
        return matches
    
    def match_keywords(self, keywords: List[str]) -> List[DomainMatch]:
        """
        Match a list of keywords against the vocabulary
        
        Useful for matching extracted keywords from code/logs
        against the domain vocabulary.
        
        Args:
            keywords: List of keywords to match
            
        Returns:
            List of DomainMatch objects for matched keywords
        """
        matches = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            
            if keyword_lower in self.term_to_entry:
                entry = self.term_to_entry[keyword_lower]
                match = DomainMatch(
                    matched_term=keyword,
                    canonical_term=entry.canonical_term,
                    capabilities=entry.business_capabilities.copy(),
                    category=entry.metadata_category,
                    confidence=1.0
                )
                matches.append(match)
        
        return matches
    
    def get_capabilities(self) -> List[str]:
        """Get all unique business capabilities in vocabulary"""
        return list(self.by_capability.keys())
    
    def get_categories(self) -> List[str]:
        """Get all metadata categories in vocabulary"""
        return list(self.by_category.keys())
    
    def get_entries_by_capability(self, capability: str) -> List[VocabularyEntry]:
        """Get all vocabulary entries for a business capability"""
        return self.by_capability.get(capability, [])
    
    def get_entry_by_term(self, term: str) -> Optional[VocabularyEntry]:
        """Look up vocabulary entry by any of its terms"""
        return self.term_to_entry.get(term.lower())
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a search query with synonyms from vocabulary
        
        Takes a query and adds related terms from the vocabulary
        to improve recall.
        
        Args:
            query: Original search query
            
        Returns:
            List of expanded query terms
        """
        expanded = [query]
        
        # Find matches in query
        matches = self.match_text(query, deduplicate=True)
        
        for match in matches:
            entry = self.by_canonical.get(match.canonical_term.lower())
            if entry:
                # Add canonical term if different
                if entry.canonical_term.lower() != match.matched_term.lower():
                    expanded.append(entry.canonical_term)
                
                # Add a few related keywords
                for related in entry.related_keywords[:3]:
                    if related.lower() not in [e.lower() for e in expanded]:
                        expanded.append(related)
        
        return expanded
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vocabulary statistics"""
        return {
            'total_entries': len(self.entries),
            'total_terms': len(self.term_to_entry),
            'categories': len(self.by_category),
            'capabilities': len(self.by_capability),
            'terms_per_entry': round(len(self.term_to_entry) / max(1, len(self.entries)), 2),
            'top_categories': sorted(
                [(cat, len(entries)) for cat, entries in self.by_category.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """Export vocabulary as list of dicts"""
        return [
            {
                'canonical_term': e.canonical_term,
                'keywords': e.keywords,
                'related_keywords': e.related_keywords,
                'description': e.description,
                'metadata_category': e.metadata_category,
                'business_capabilities': e.business_capabilities
            }
            for e in self.entries
        ]
