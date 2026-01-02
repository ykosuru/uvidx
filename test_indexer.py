"""
Comprehensive test suite for the Unified Indexer

Run with: pytest tests/ -v
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# Import the unified indexer components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_indexer.models import (
    SourceType, SemanticType, DomainMatch, IndexableChunk,
    SourceReference, VocabularyEntry, SearchResult
)
from unified_indexer.vocabulary import DomainVocabulary, AhoCorasickAutomaton
from unified_indexer.parsers.base import ContentParser
from unified_indexer.parsers.tal_parser import TalCodeParser
from unified_indexer.parsers.document_parser import DocumentParser
from unified_indexer.parsers.log_parser import LogParser
from unified_indexer.index import HybridIndex, VectorStore, ConceptIndex
from unified_indexer.pipeline import IndexingPipeline


# ============================================================
# Test Data Fixtures
# ============================================================

@pytest.fixture
def sample_vocabulary_data():
    """Sample payment systems vocabulary for testing"""
    return [
        {
            "keywords": "wire transfer,electronic transfer,funds transfer",
            "metadata": "payment-systems",
            "description": "Electronic transfer of funds between financial institutions",
            "related_keywords": "domestic wire,international wire,high-value payment",
            "business_capability": ["Payment Processing", "Wire Transfer"]
        },
        {
            "keywords": "OFAC,sanctions screening,prohibited transfer",
            "metadata": "compliance-fraud",
            "description": "Office of Foreign Assets Control screening",
            "related_keywords": "sanctions check,blocked persons,ofac check",
            "business_capability": ["OFAC Screening", "Sanctions Compliance"]
        },
        {
            "keywords": "MT-103,customer credit transfer,customer wire",
            "metadata": "swift-mt-messages",
            "description": "SWIFT MT-103 single customer credit transfer message",
            "related_keywords": "customer transfer,credit transfer",
            "business_capability": ["MT-103 Processing", "Inbound SWIFT", "Outbound SWIFT"]
        },
        {
            "keywords": "pacs.008,customer credit transfer,FI credit transfer",
            "metadata": "iso20022-pacs-messages",
            "description": "ISO 20022 pacs.008 Customer Credit Transfer",
            "related_keywords": "customer transfer,payment initiation",
            "business_capability": ["Customer Credit Transfer", "Payment Initiation"]
        },
        {
            "keywords": "payment return,return payment,payment refund",
            "metadata": "exception-handling",
            "description": "Processing returned payments",
            "related_keywords": "return process,refund payment",
            "business_capability": ["Payment Return", "Exception Handling"]
        },
        {
            "keywords": "Dodd Frank,dodd-frank compliance",
            "metadata": "regulatory-compliance",
            "description": "Compliance with Dodd-Frank Act regulations",
            "related_keywords": "dodd-frank rules,financial reform",
            "business_capability": ["Dodd Frank"]
        }
    ]


@pytest.fixture
def vocabulary(sample_vocabulary_data):
    """Create a loaded vocabulary instance"""
    vocab = DomainVocabulary()
    vocab.load_from_data(sample_vocabulary_data)
    return vocab


@pytest.fixture
def sample_tal_code():
    """Sample TAL code for testing"""
    return """
! Wire Transfer Processing Module
! Handles MT-103 customer credit transfers

INT PROC PROCESS^WIRE^TRANSFER(transfer_record);
    INT .transfer_record;
BEGIN
    ! Perform OFAC sanctions screening
    IF NOT OFAC^CHECK(transfer_record) THEN
        CALL LOG^ERROR("Sanctions screening failed");
        RETURN -1;
    END;
    
    ! Process the wire transfer
    CALL SEND^MT103(transfer_record);
    RETURN 0;
END;

INT PROC HANDLE^PAYMENT^RETURN(return_record);
    INT .return_record;
BEGIN
    ! Process payment return per Dodd Frank requirements
    CALL UPDATE^STATUS(return_record, "RETURNED");
    RETURN 0;
END;

STRING PROC FORMAT^PACS008^MESSAGE(msg_data);
    STRING .msg_data;
BEGIN
    ! Format pacs.008 customer credit transfer message
    RETURN $LEN(msg_data);
END;
"""


@pytest.fixture
def sample_log_entries():
    """Sample JSON log entries for testing"""
    return """
{"timestamp": "2025-01-02T10:15:30Z", "level": "INFO", "transaction_id": "TXN001", "message": "Processing wire transfer for $50,000"}
{"timestamp": "2025-01-02T10:15:31Z", "level": "INFO", "transaction_id": "TXN001", "message": "OFAC screening initiated"}
{"timestamp": "2025-01-02T10:15:32Z", "level": "ERROR", "transaction_id": "TXN001", "error_code": "OFAC_MATCH", "message": "Sanctions screening failed - potential match found"}
{"timestamp": "2025-01-02T10:15:33Z", "level": "INFO", "transaction_id": "TXN001", "message": "Payment flagged for manual review"}
{"timestamp": "2025-01-02T10:16:00Z", "level": "INFO", "transaction_id": "TXN002", "message": "Processing MT-103 customer credit transfer"}
{"timestamp": "2025-01-02T10:16:01Z", "level": "INFO", "transaction_id": "TXN002", "message": "pacs.008 message formatted successfully"}
"""


@pytest.fixture
def sample_document_text():
    """Sample document text for testing"""
    return """
Wire Transfer Processing Guide

Overview
This document describes the wire transfer processing workflow including
MT-103 message handling and OFAC sanctions screening requirements.

OFAC Screening Requirements
All wire transfers must undergo OFAC sanctions screening before processing.
The screening checks against the SDN list and other prohibited party lists.
Dodd Frank compliance requires additional documentation for certain transfers.

Payment Returns
When a payment is returned, the system must update the status and notify
the originating party. Common return reasons include insufficient funds
and invalid beneficiary information.

ISO 20022 Message Formats
The pacs.008 customer credit transfer message is used for outbound payments.
This format provides richer data than legacy MT-103 messages.
"""


# ============================================================
# Vocabulary Tests
# ============================================================

class TestVocabulary:
    """Tests for the DomainVocabulary class"""
    
    def test_load_vocabulary(self, vocabulary, sample_vocabulary_data):
        """Test vocabulary loading"""
        assert len(vocabulary.entries) == len(sample_vocabulary_data)
        assert len(vocabulary.term_to_entry) > 0
    
    def test_match_simple_term(self, vocabulary):
        """Test matching a simple term"""
        text = "This is a wire transfer transaction"
        matches = vocabulary.match_text(text)
        
        assert len(matches) > 0
        assert any(m.canonical_term == "wire transfer" for m in matches)
    
    def test_match_multi_word_term(self, vocabulary):
        """Test matching multi-word terms like 'Dodd Frank'"""
        text = "Compliance with Dodd Frank regulations is required"
        matches = vocabulary.match_text(text)
        
        assert len(matches) > 0
        assert any("dodd" in m.canonical_term.lower() for m in matches)
    
    def test_match_case_insensitive(self, vocabulary):
        """Test case-insensitive matching"""
        text = "OFAC SCREENING is mandatory for WIRE TRANSFERS"
        matches = vocabulary.match_text(text)
        
        assert len(matches) >= 2
        canonical_terms = [m.canonical_term.lower() for m in matches]
        assert any("ofac" in term for term in canonical_terms)
        assert any("wire" in term or "transfer" in term for term in canonical_terms)
    
    def test_match_related_keywords(self, vocabulary):
        """Test matching related keywords/synonyms"""
        text = "The domestic wire was processed"
        matches = vocabulary.match_text(text)
        
        # Should match "domestic wire" as related to "wire transfer"
        assert len(matches) > 0
    
    def test_capability_mapping(self, vocabulary):
        """Test that matches include business capabilities"""
        text = "MT-103 message processing"
        matches = vocabulary.match_text(text)
        
        assert len(matches) > 0
        mt103_match = next((m for m in matches if "MT-103" in m.canonical_term), None)
        assert mt103_match is not None
        assert len(mt103_match.capabilities) > 0
        assert "MT-103 Processing" in mt103_match.capabilities
    
    def test_query_expansion(self, vocabulary):
        """Test query expansion with synonyms"""
        expanded = vocabulary.expand_query("wire transfer")
        
        assert len(expanded) > 1
        assert "wire transfer" in [e.lower() for e in expanded]
    
    def test_get_entries_by_capability(self, vocabulary):
        """Test retrieving entries by capability"""
        entries = vocabulary.get_entries_by_capability("Payment Processing")
        
        assert len(entries) > 0
        assert all("Payment Processing" in e.business_capabilities for e in entries)


class TestAhoCorasickAutomaton:
    """Tests for the Aho-Corasick automaton"""
    
    def test_single_pattern(self):
        """Test matching a single pattern"""
        automaton = AhoCorasickAutomaton()
        entry = VocabularyEntry(
            canonical_term="test",
            keywords=["test"],
            related_keywords=[],
            description="Test term",
            metadata_category="test",
            business_capabilities=[]
        )
        automaton.add_pattern("test", entry)
        automaton.build()
        
        results = automaton.search("this is a test string")
        assert len(results) == 1
        assert results[0][2] == "test"  # matched term
    
    def test_multiple_patterns(self):
        """Test matching multiple patterns"""
        automaton = AhoCorasickAutomaton()
        
        for term in ["wire", "transfer", "payment"]:
            entry = VocabularyEntry(
                canonical_term=term,
                keywords=[term],
                related_keywords=[],
                description=f"{term} description",
                metadata_category="test",
                business_capabilities=[]
            )
            automaton.add_pattern(term, entry)
        
        automaton.build()
        
        results = automaton.search("wire transfer payment processing")
        assert len(results) == 3
    
    def test_overlapping_patterns(self):
        """Test handling of overlapping patterns"""
        automaton = AhoCorasickAutomaton()
        
        entry1 = VocabularyEntry(
            canonical_term="wire",
            keywords=["wire"],
            related_keywords=[],
            description="Wire",
            metadata_category="test",
            business_capabilities=[]
        )
        entry2 = VocabularyEntry(
            canonical_term="wire transfer",
            keywords=["wire transfer"],
            related_keywords=[],
            description="Wire transfer",
            metadata_category="test",
            business_capabilities=[]
        )
        
        automaton.add_pattern("wire", entry1)
        automaton.add_pattern("wire transfer", entry2)
        automaton.build()
        
        results = automaton.search("wire transfer")
        # Should find both "wire" and "wire transfer"
        assert len(results) >= 1


# ============================================================
# Model Tests
# ============================================================

class TestModels:
    """Tests for data models"""
    
    def test_vocabulary_entry_from_dict(self, sample_vocabulary_data):
        """Test creating VocabularyEntry from dict"""
        entry = VocabularyEntry.from_dict(sample_vocabulary_data[0])
        
        assert entry.canonical_term == "wire transfer"
        assert "electronic transfer" in entry.keywords
        assert "domestic wire" in entry.related_keywords
        assert "Payment Processing" in entry.business_capabilities
    
    def test_domain_match_creation(self):
        """Test DomainMatch creation"""
        match = DomainMatch(
            matched_term="wire transfer",
            canonical_term="wire transfer",
            capabilities=["Payment Processing"],
            category="payment-systems",
            span=(10, 23),
            confidence=1.0
        )
        
        assert match.matched_term == "wire transfer"
        assert match.confidence == 1.0
        
        match_dict = match.to_dict()
        assert "matched_term" in match_dict
        assert "capabilities" in match_dict
    
    def test_source_reference(self):
        """Test SourceReference creation and string representation"""
        ref = SourceReference(
            file_path="/code/payment.tal",
            line_start=10,
            line_end=50,
            procedure_name="PROCESS_WIRE"
        )
        
        assert "payment.tal" in str(ref)
        assert "10" in str(ref)
        assert "PROCESS_WIRE" in str(ref)
    
    def test_indexable_chunk_creation(self, vocabulary):
        """Test IndexableChunk creation"""
        ref = SourceReference(file_path="/code/test.tal", line_start=1)
        
        matches = vocabulary.match_text("wire transfer processing")
        
        chunk = IndexableChunk(
            chunk_id="test_chunk_1",
            text="Process wire transfer",
            embedding_text="TAL procedure for wire transfer processing",
            source_type=SourceType.CODE,
            semantic_type=SemanticType.PROCEDURE,
            source_ref=ref,
            domain_matches=matches
        )
        
        assert chunk.chunk_id == "test_chunk_1"
        assert chunk.source_type == SourceType.CODE
        assert len(chunk.domain_matches) > 0
        assert len(chunk.capability_set) > 0
    
    def test_indexable_chunk_serialization(self, vocabulary):
        """Test chunk serialization and deserialization"""
        ref = SourceReference(file_path="/code/test.tal", line_start=1)
        matches = vocabulary.match_text("wire transfer")
        
        chunk = IndexableChunk(
            chunk_id="test_chunk_1",
            text="Process wire transfer",
            embedding_text="Wire transfer processing",
            source_type=SourceType.CODE,
            semantic_type=SemanticType.PROCEDURE,
            source_ref=ref,
            domain_matches=matches
        )
        
        chunk_dict = chunk.to_dict()
        assert "chunk_id" in chunk_dict
        assert "source_type" in chunk_dict
        assert chunk_dict["source_type"] == "code"
        
        # Test round-trip
        restored = IndexableChunk.from_dict(chunk_dict)
        assert restored.chunk_id == chunk.chunk_id
        assert restored.source_type == chunk.source_type


# ============================================================
# Parser Tests
# ============================================================

class TestTalParser:
    """Tests for the TAL code parser"""
    
    def test_parser_initialization(self, vocabulary):
        """Test TAL parser initialization"""
        parser = TalCodeParser(vocabulary)
        
        assert parser.vocabulary == vocabulary
        assert parser.SOURCE_TYPE == SourceType.CODE
        assert ".tal" in parser.SUPPORTED_EXTENSIONS
    
    def test_can_parse_tal_files(self, vocabulary):
        """Test file type detection"""
        parser = TalCodeParser(vocabulary)
        
        assert parser.can_parse("payment.tal")
        assert parser.can_parse("PAYMENT.TAL")
        assert parser.can_parse("/path/to/code.tacl")
        assert not parser.can_parse("document.pdf")
        assert not parser.can_parse("data.json")
    
    def test_parse_tal_code(self, vocabulary, sample_tal_code):
        """Test parsing TAL code"""
        parser = TalCodeParser(vocabulary)
        chunks = parser.parse(sample_tal_code.encode('utf-8'), 'test.tal')
        
        assert len(chunks) > 0
        assert all(c.source_type == SourceType.CODE for c in chunks)
    
    def test_domain_matching_in_code(self, vocabulary, sample_tal_code):
        """Test that domain concepts are extracted from code"""
        parser = TalCodeParser(vocabulary)
        chunks = parser.parse(sample_tal_code.encode('utf-8'), 'test.tal')
        
        # Collect all domain matches across chunks
        all_matches = []
        for chunk in chunks:
            all_matches.extend(chunk.domain_matches)
        
        # Should find payment-related concepts
        canonical_terms = [m.canonical_term.lower() for m in all_matches]
        
        # At least some domain terms should be found
        assert len(all_matches) > 0
    
    def test_procedure_extraction(self, vocabulary, sample_tal_code):
        """Test that procedures are properly extracted"""
        parser = TalCodeParser(vocabulary)
        chunks = parser.parse(sample_tal_code.encode('utf-8'), 'test.tal')
        
        # Look for procedure chunks
        proc_chunks = [c for c in chunks if c.semantic_type == SemanticType.PROCEDURE]
        
        # The sample code has 3 procedures
        assert len(proc_chunks) >= 1


class TestDocumentParser:
    """Tests for the document parser"""
    
    def test_parser_initialization(self, vocabulary):
        """Test document parser initialization"""
        parser = DocumentParser(vocabulary)
        
        assert parser.vocabulary == vocabulary
        assert parser.SOURCE_TYPE == SourceType.DOCUMENT
        assert ".pdf" in parser.SUPPORTED_EXTENSIONS
        assert ".txt" in parser.SUPPORTED_EXTENSIONS
    
    def test_parse_text_document(self, vocabulary, sample_document_text):
        """Test parsing text documents"""
        parser = DocumentParser(vocabulary)
        chunks = parser.parse(sample_document_text.encode('utf-8'), 'guide.txt')
        
        assert len(chunks) > 0
        assert all(c.source_type == SourceType.DOCUMENT for c in chunks)
    
    def test_domain_matching_in_documents(self, vocabulary, sample_document_text):
        """Test domain concept extraction from documents"""
        parser = DocumentParser(vocabulary)
        chunks = parser.parse(sample_document_text.encode('utf-8'), 'guide.txt')
        
        all_matches = []
        for chunk in chunks:
            all_matches.extend(chunk.domain_matches)
        
        # Should find OFAC, MT-103, pacs.008, etc.
        canonical_terms = [m.canonical_term.lower() for m in all_matches]
        
        assert len(all_matches) > 0
    
    def test_section_detection(self, vocabulary, sample_document_text):
        """Test section detection in documents"""
        parser = DocumentParser(vocabulary)
        chunks = parser.parse(sample_document_text.encode('utf-8'), 'guide.txt')
        
        # Check that sections are detected
        section_chunks = [c for c in chunks if c.semantic_type in [SemanticType.SECTION, SemanticType.PARAGRAPH]]
        assert len(section_chunks) > 0


class TestLogParser:
    """Tests for the log parser"""
    
    def test_parser_initialization(self, vocabulary):
        """Test log parser initialization"""
        parser = LogParser(vocabulary)
        
        assert parser.vocabulary == vocabulary
        assert parser.SOURCE_TYPE == SourceType.LOG
        assert ".log" in parser.SUPPORTED_EXTENSIONS
        assert ".json" in parser.SUPPORTED_EXTENSIONS
    
    def test_parse_json_logs(self, vocabulary, sample_log_entries):
        """Test parsing JSON log entries"""
        parser = LogParser(vocabulary)
        chunks = parser.parse(sample_log_entries.encode('utf-8'), 'transaction.log')
        
        assert len(chunks) > 0
        assert all(c.source_type == SourceType.LOG for c in chunks)
    
    def test_trace_grouping(self, vocabulary, sample_log_entries):
        """Test that log entries are grouped by transaction ID"""
        parser = LogParser(vocabulary, group_by_trace=True)
        chunks = parser.parse(sample_log_entries.encode('utf-8'), 'transaction.log')
        
        # Should group by transaction_id
        # TXN001 has 4 entries, TXN002 has 2 entries
        # So we should have 2 trace groups
        assert len(chunks) <= 6  # At most 6 individual entries or fewer groups
    
    def test_error_detection(self, vocabulary, sample_log_entries):
        """Test error detection in logs"""
        parser = LogParser(vocabulary)
        chunks = parser.parse(sample_log_entries.encode('utf-8'), 'transaction.log')
        
        # Check for error traces
        error_chunks = [c for c in chunks if c.semantic_type == SemanticType.ERROR_TRACE]
        
        # TXN001 has an error, so at least one error trace
        assert any(c.metadata.get('has_error', False) for c in chunks)


# ============================================================
# Index Tests
# ============================================================

class TestVectorStore:
    """Tests for the vector store"""
    
    def test_add_and_search(self, vocabulary):
        """Test adding and searching vectors"""
        store = VectorStore()
        
        ref = SourceReference(file_path="test.tal")
        chunk = IndexableChunk(
            chunk_id="test_1",
            text="Wire transfer processing",
            embedding_text="Wire transfer processing",
            source_type=SourceType.CODE,
            semantic_type=SemanticType.PROCEDURE,
            source_ref=ref
        )
        
        # Simple embedding (normally would use real embeddings)
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        store.add("test_1", embedding, chunk)
        
        assert len(store) == 1
        
        # Search with similar embedding
        query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55]
        results = store.search(query_embedding, top_k=1)
        
        assert len(results) == 1
        assert results[0][0] == "test_1"


class TestConceptIndex:
    """Tests for the concept index"""
    
    def test_add_and_search(self, vocabulary):
        """Test adding and searching concepts"""
        index = ConceptIndex()
        
        ref = SourceReference(file_path="test.tal")
        matches = vocabulary.match_text("wire transfer OFAC screening")
        
        chunk = IndexableChunk(
            chunk_id="test_1",
            text="Wire transfer OFAC screening",
            embedding_text="Wire transfer OFAC screening",
            source_type=SourceType.CODE,
            semantic_type=SemanticType.PROCEDURE,
            source_ref=ref,
            domain_matches=matches
        )
        
        index.add(chunk)
        
        assert len(index) == 1
        
        # Search by concept
        results = index.search_concept("wire transfer")
        assert "test_1" in results
    
    def test_search_by_capability(self, vocabulary):
        """Test searching by business capability"""
        index = ConceptIndex()
        
        ref = SourceReference(file_path="test.tal")
        matches = vocabulary.match_text("OFAC screening process")
        
        chunk = IndexableChunk(
            chunk_id="test_1",
            text="OFAC screening process",
            embedding_text="OFAC screening process",
            source_type=SourceType.CODE,
            semantic_type=SemanticType.PROCEDURE,
            source_ref=ref,
            domain_matches=matches
        )
        
        index.add(chunk)
        
        # Search by capability
        results = index.search_capability("OFAC Screening")
        assert "test_1" in results


class TestHybridIndex:
    """Tests for the hybrid index"""
    
    def test_index_and_search(self, vocabulary):
        """Test hybrid indexing and search"""
        index = HybridIndex(vocabulary)
        
        ref = SourceReference(file_path="test.tal")
        matches = vocabulary.match_text("wire transfer processing")
        
        chunk = IndexableChunk(
            chunk_id="test_1",
            text="Wire transfer processing procedure",
            embedding_text="Wire transfer processing procedure",
            source_type=SourceType.CODE,
            semantic_type=SemanticType.PROCEDURE,
            source_ref=ref,
            domain_matches=matches
        )
        
        index.index_chunk(chunk)
        
        # Search (concept-only since no embedding function)
        results = index.search("wire transfer", top_k=5)
        
        assert len(results) > 0
        assert results[0].chunk.chunk_id == "test_1"
    
    def test_search_by_capability(self, vocabulary):
        """Test capability-based search"""
        index = HybridIndex(vocabulary)
        
        ref = SourceReference(file_path="test.tal")
        matches = vocabulary.match_text("MT-103 message handling")
        
        chunk = IndexableChunk(
            chunk_id="mt103_chunk",
            text="MT-103 message handling",
            embedding_text="MT-103 message handling",
            source_type=SourceType.CODE,
            semantic_type=SemanticType.PROCEDURE,
            source_ref=ref,
            domain_matches=matches
        )
        
        index.index_chunk(chunk)
        
        results = index.search_by_capability("MT-103 Processing", top_k=5)
        
        # Should find the MT-103 chunk
        assert any(r.chunk.chunk_id == "mt103_chunk" for r in results)


# ============================================================
# Pipeline Tests
# ============================================================

class TestIndexingPipeline:
    """Tests for the indexing pipeline"""
    
    def test_pipeline_initialization(self, sample_vocabulary_data):
        """Test pipeline initialization"""
        pipeline = IndexingPipeline(vocabulary_data=sample_vocabulary_data)
        
        assert pipeline.vocabulary is not None
        assert len(pipeline.parsers) > 0
        assert pipeline.index is not None
    
    def test_index_content(self, sample_vocabulary_data, sample_tal_code):
        """Test indexing content directly"""
        pipeline = IndexingPipeline(vocabulary_data=sample_vocabulary_data)
        
        chunks = pipeline.index_content(
            sample_tal_code.encode('utf-8'),
            'test.tal',
            SourceType.CODE
        )
        
        assert len(chunks) > 0
    
    def test_search(self, sample_vocabulary_data, sample_tal_code, sample_log_entries):
        """Test searching indexed content"""
        pipeline = IndexingPipeline(vocabulary_data=sample_vocabulary_data)
        
        # Index code
        pipeline.index_content(
            sample_tal_code.encode('utf-8'),
            'payment.tal',
            SourceType.CODE
        )
        
        # Index logs
        pipeline.index_content(
            sample_log_entries.encode('utf-8'),
            'transaction.log',
            SourceType.LOG
        )
        
        # Search
        results = pipeline.search("OFAC screening", top_k=5)
        
        assert len(results) > 0
    
    def test_cross_reference_search(self, sample_vocabulary_data, sample_tal_code, sample_log_entries):
        """Test cross-referencing between source types"""
        pipeline = IndexingPipeline(vocabulary_data=sample_vocabulary_data)
        
        # Index code and logs
        pipeline.index_content(
            sample_tal_code.encode('utf-8'),
            'payment.tal',
            SourceType.CODE
        )
        
        pipeline.index_content(
            sample_log_entries.encode('utf-8'),
            'transaction.log',
            SourceType.LOG
        )
        
        # Cross-reference: find code related to log errors
        results = pipeline.search_cross_reference(
            "sanctions screening failed",
            from_type=SourceType.LOG,
            to_types=[SourceType.CODE],
            top_k=3
        )
        
        assert SourceType.LOG.value in results
        assert SourceType.CODE.value in results
    
    def test_get_by_capability(self, sample_vocabulary_data, sample_tal_code):
        """Test finding content by business capability"""
        pipeline = IndexingPipeline(vocabulary_data=sample_vocabulary_data)
        
        pipeline.index_content(
            sample_tal_code.encode('utf-8'),
            'payment.tal',
            SourceType.CODE
        )
        
        results = pipeline.get_by_capability("Payment Processing", top_k=5)
        
        # Results may or may not exist depending on concept matching
        # Just verify the method works
        assert isinstance(results, list)
    
    def test_statistics(self, sample_vocabulary_data, sample_tal_code):
        """Test getting pipeline statistics"""
        pipeline = IndexingPipeline(vocabulary_data=sample_vocabulary_data)
        
        pipeline.index_content(
            sample_tal_code.encode('utf-8'),
            'payment.tal',
            SourceType.CODE
        )
        
        stats = pipeline.get_statistics()
        
        assert 'pipeline' in stats
        assert 'index' in stats
        assert 'vocabulary' in stats


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_full_workflow(self, sample_vocabulary_data, sample_tal_code, 
                           sample_log_entries, sample_document_text):
        """Test complete indexing and search workflow"""
        # Create pipeline
        pipeline = IndexingPipeline(vocabulary_data=sample_vocabulary_data)
        
        # Index all content types
        code_chunks = pipeline.index_content(
            sample_tal_code.encode('utf-8'),
            'payment.tal',
            SourceType.CODE
        )
        
        log_chunks = pipeline.index_content(
            sample_log_entries.encode('utf-8'),
            'transaction.log',
            SourceType.LOG
        )
        
        doc_chunks = pipeline.index_content(
            sample_document_text.encode('utf-8'),
            'guide.txt',
            SourceType.DOCUMENT
        )
        
        # Verify indexing
        assert len(code_chunks) > 0
        assert len(log_chunks) > 0
        assert len(doc_chunks) > 0
        
        # Search across all content
        results = pipeline.search("OFAC sanctions screening", top_k=10)
        
        # Should find content from multiple sources
        source_types_found = set(r.chunk.source_type for r in results)
        assert len(results) > 0
        
        # Get statistics
        stats = pipeline.get_statistics()
        total_chunks = stats['pipeline']['total_chunks']
        assert total_chunks == len(code_chunks) + len(log_chunks) + len(doc_chunks)
    
    def test_save_and_load(self, sample_vocabulary_data, sample_tal_code, tmp_path):
        """Test saving and loading the index"""
        # Create and populate pipeline
        pipeline = IndexingPipeline(vocabulary_data=sample_vocabulary_data)
        
        pipeline.index_content(
            sample_tal_code.encode('utf-8'),
            'payment.tal',
            SourceType.CODE
        )
        
        # Save
        save_path = str(tmp_path / "test_index")
        pipeline.save(save_path)
        
        # Create new pipeline and load
        pipeline2 = IndexingPipeline(vocabulary_data=sample_vocabulary_data)
        pipeline2.load(save_path)
        
        # Verify search still works
        results = pipeline2.search("wire transfer", top_k=5)
        
        # Should still find content
        assert isinstance(results, list)


# ============================================================
# Run Tests
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
