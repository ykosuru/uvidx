#!/usr/bin/env python3
"""
Example Usage of the Unified Indexer

This script demonstrates:
1. Loading the domain vocabulary
2. Parsing different content types (code, documents, logs)
3. Building a hybrid index
4. Performing various search operations
5. Cross-referencing between content types

Run from the parent directory:
    python examples/usage_example.py
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_indexer import (
    IndexingPipeline,
    DomainVocabulary,
    TalCodeParser,
    DocumentParser,
    LogParser,
    HybridIndex,
    SourceType,
    IndexableChunk
)


# ============================================================
# Example 1: Using the Pipeline (Recommended)
# ============================================================

def example_pipeline_usage():
    """Demonstrate the high-level pipeline interface"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Using the IndexingPipeline")
    print("="*60)
    
    # Sample vocabulary (subset of the full vocabulary)
    vocabulary_data = [
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
            "description": "Office of Foreign Assets Control screening for sanctions compliance",
            "related_keywords": "sanctions check,blocked persons,ofac check",
            "business_capability": ["OFAC Screening", "Sanctions Compliance"]
        },
        {
            "keywords": "MT-103,customer credit transfer,customer wire",
            "metadata": "swift-mt-messages",
            "description": "SWIFT MT-103 single customer credit transfer message",
            "related_keywords": "customer transfer,credit transfer,pacs.008 equivalent",
            "business_capability": ["MT-103 Processing", "Inbound SWIFT", "Outbound SWIFT"]
        },
        {
            "keywords": "payment return,return payment,payment refund",
            "metadata": "exception-handling",
            "description": "Processing returned payments",
            "related_keywords": "return process,refund payment,payment reversal",
            "business_capability": ["Payment Return", "Exception Handling"]
        }
    ]
    
    # Create pipeline with embedded vocabulary
    pipeline = IndexingPipeline(vocabulary_data=vocabulary_data)
    
    # Sample TAL code
    tal_code = """
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
        ! Process payment return
        CALL UPDATE^STATUS(return_record, "RETURNED");
        RETURN 0;
    END;
    """
    
    # Sample log entries
    log_entries = """
    {"timestamp": "2025-01-02T10:15:30Z", "level": "INFO", "transaction_id": "TXN001", "message": "Processing wire transfer for $50,000"}
    {"timestamp": "2025-01-02T10:15:31Z", "level": "INFO", "transaction_id": "TXN001", "message": "OFAC screening initiated"}
    {"timestamp": "2025-01-02T10:15:32Z", "level": "ERROR", "transaction_id": "TXN001", "error_code": "OFAC_MATCH", "message": "Sanctions screening failed - potential match found"}
    {"timestamp": "2025-01-02T10:15:33Z", "level": "INFO", "transaction_id": "TXN001", "message": "Payment flagged for manual review"}
    """
    
    # Sample document text
    document_text = """
    Wire Transfer Processing Guide
    
    Overview
    This document describes the wire transfer processing workflow including
    MT-103 message handling and OFAC sanctions screening requirements.
    
    OFAC Screening Requirements
    All wire transfers must undergo OFAC sanctions screening before processing.
    The screening checks against the SDN list and other prohibited party lists.
    
    Payment Returns
    When a payment is returned, the system must update the status and notify
    the originating party. Common return reasons include insufficient funds
    and invalid beneficiary information.
    """
    
    # Index the content
    print("\nIndexing sample content...")
    
    # Index TAL code
    code_chunks = pipeline.index_content(
        tal_code.encode('utf-8'),
        'wire_transfer.tal',
        SourceType.CODE
    )
    print(f"  Indexed {len(code_chunks)} code chunks")
    
    # Index logs
    log_chunks = pipeline.index_content(
        log_entries.encode('utf-8'),
        'transaction.log',
        SourceType.LOG
    )
    print(f"  Indexed {len(log_chunks)} log chunks")
    
    # Index document
    doc_chunks = pipeline.index_content(
        document_text.encode('utf-8'),
        'wire_transfer_guide.txt',
        SourceType.DOCUMENT
    )
    print(f"  Indexed {len(doc_chunks)} document chunks")
    
    # Perform searches
    print("\n" + "-"*40)
    print("SEARCH RESULTS")
    print("-"*40)
    
    # Search 1: Find OFAC-related content
    print("\n1. Searching for 'OFAC sanctions screening':")
    results = pipeline.search("OFAC sanctions screening", top_k=5)
    for r in results:
        print(f"   [{r.chunk.source_type.value}] Score: {r.combined_score:.3f}")
        print(f"   Concepts: {r.matched_concepts}")
        print(f"   Preview: {r.chunk.text[:100]}...")
        print()
    
    # Search 2: Find by capability
    print("\n2. Finding content by 'Payment Return' capability:")
    results = pipeline.get_by_capability("Payment Return", top_k=5)
    for r in results:
        print(f"   [{r.chunk.source_type.value}] {r.chunk.source_ref}")
        print(f"   Preview: {r.chunk.text[:100]}...")
        print()
    
    # Search 3: Cross-reference - find code related to log errors
    print("\n3. Cross-referencing: Finding code related to 'sanctions screening failed':")
    xref_results = pipeline.search_cross_reference(
        "sanctions screening failed",
        from_type=SourceType.LOG,
        to_types=[SourceType.CODE, SourceType.DOCUMENT],
        top_k=3
    )
    
    for source_type, results in xref_results.items():
        print(f"\n   {source_type} results:")
        for r in results:
            print(f"      Source: {r.chunk.source_ref}")
            print(f"      Concepts: {r.matched_concepts}")
    
    # Print statistics
    pipeline.print_statistics()
    
    return pipeline


# ============================================================
# Example 2: Using Individual Parsers
# ============================================================

def example_parser_usage():
    """Demonstrate using parsers directly"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Using Individual Parsers")
    print("="*60)
    
    # Create vocabulary
    vocab = DomainVocabulary()
    vocab.load_from_data([
        {
            "keywords": "pacs.008,customer credit transfer",
            "metadata": "iso20022-pacs-messages",
            "description": "ISO 20022 pacs.008 Customer Credit Transfer",
            "related_keywords": "customer transfer,MT-103 equivalent",
            "business_capability": ["Customer Credit Transfer", "Payment Initiation"]
        }
    ])
    
    # Create TAL parser
    tal_parser = TalCodeParser(vocab)
    
    # Parse some code
    code = """
    INT PROC SEND^PACS008(message);
        INT .message;
    BEGIN
        ! Format as pacs.008 customer credit transfer
        CALL FORMAT^ISO20022(message);
        RETURN 0;
    END;
    """
    
    chunks = tal_parser.parse(code.encode('utf-8'), 'iso_converter.tal')
    
    print(f"\nParsed {len(chunks)} chunks from TAL code:")
    for chunk in chunks:
        print(f"\n  Chunk ID: {chunk.chunk_id}")
        print(f"  Type: {chunk.semantic_type.value}")
        print(f"  Domain Matches: {len(chunk.domain_matches)}")
        for match in chunk.domain_matches:
            print(f"    - {match.canonical_term} -> {match.capabilities}")


# ============================================================
# Example 3: Working with the Vocabulary
# ============================================================

def example_vocabulary_usage():
    """Demonstrate vocabulary features"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Working with Domain Vocabulary")
    print("="*60)
    
    vocab = DomainVocabulary()
    vocab.load_from_data([
        {
            "keywords": "SWIFT,society for worldwide interbank financial telecommunication",
            "metadata": "payment-networks",
            "description": "Global messaging network for secure financial instructions",
            "related_keywords": "swift messaging,international messaging,BIC code",
            "business_capability": ["SWIFT Messaging", "International Messaging"]
        },
        {
            "keywords": "Fedwire,federal reserve wire",
            "metadata": "payment-networks",
            "description": "U.S. Federal Reserve's RTGS system for domestic high-value wires",
            "related_keywords": "fed wire,RTGS system,domestic settlement",
            "business_capability": ["Fedwire Processing", "RTGS Settlement"]
        }
    ])
    
    # Test concept matching
    text = "The payment was sent via SWIFT messaging to the correspondent bank"
    matches = vocab.match_text(text)
    
    print(f"\nText: '{text}'")
    print(f"Found {len(matches)} domain matches:")
    for match in matches:
        print(f"  - '{match.matched_term}' -> canonical: '{match.canonical_term}'")
        print(f"    Capabilities: {match.capabilities}")
    
    # Query expansion
    query = "fedwire payment"
    expanded = vocab.expand_query(query)
    print(f"\nQuery expansion for '{query}':")
    print(f"  Original: {query}")
    print(f"  Expanded: {expanded}")
    
    # Statistics
    stats = vocab.get_statistics()
    print(f"\nVocabulary statistics:")
    print(f"  Entries: {stats['total_entries']}")
    print(f"  Searchable terms: {stats['total_terms']}")


# ============================================================
# Example 4: Building a Search Application
# ============================================================

def example_search_application():
    """Demonstrate building a search interface"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Building a Search Application")
    print("="*60)
    
    class PaymentCodeSearch:
        """Simple search application for payment code"""
        
        def __init__(self, vocabulary_data):
            self.pipeline = IndexingPipeline(vocabulary_data=vocabulary_data)
        
        def index_codebase(self, directory: str):
            """Index all TAL files in a directory"""
            stats = self.pipeline.index_directory(
                directory,
                extensions=['.tal'],
                max_workers=2
            )
            return stats
        
        def find_procedure(self, name: str):
            """Find a specific procedure by name"""
            results = self.pipeline.search(
                f"procedure {name}",
                top_k=5,
                source_types=[SourceType.CODE]
            )
            return results
        
        def find_capability_code(self, capability: str):
            """Find code implementing a business capability"""
            return self.pipeline.get_by_capability(capability, top_k=10)
        
        def find_error_handlers(self, error_type: str):
            """Find code that handles specific errors"""
            return self.pipeline.search(
                f"error {error_type} handling",
                top_k=5,
                source_types=[SourceType.CODE]
            )
    
    # Demo the search application
    vocab_data = [
        {
            "keywords": "validation,verification",
            "metadata": "security-compliance",
            "description": "Verification of payment details",
            "related_keywords": "account validation,message validation",
            "business_capability": ["Payment Validation", "Account Validation"]
        }
    ]
    
    search_app = PaymentCodeSearch(vocab_data)
    
    # Index some sample code
    sample_code = """
    INT PROC VALIDATE^ACCOUNT(account_num);
        STRING .account_num;
    BEGIN
        IF $LEN(account_num) < 10 THEN
            RETURN -1;
        END;
        RETURN 0;
    END;
    """
    
    search_app.pipeline.index_content(
        sample_code.encode('utf-8'),
        'validation.tal',
        SourceType.CODE
    )
    
    print("\nSearch application demo:")
    print("  Indexed sample validation code")
    
    # Search by capability
    results = search_app.find_capability_code("Payment Validation")
    print(f"  Found {len(results)} results for 'Payment Validation' capability")


# ============================================================
# Main
# ============================================================

def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("UNIFIED INDEXER - USAGE EXAMPLES")
    print("="*60)
    
    # Run examples
    example_pipeline_usage()
    example_parser_usage()
    example_vocabulary_usage()
    example_search_application()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
