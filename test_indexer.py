#!/usr/bin/env python3
"""
Standalone test script - run from the directory containing unified_indexer/

Usage:
    python test_indexer.py
"""

import sys
import os

# Add the current directory to path so we can import unified_indexer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import
from unified_indexer import (
    IndexingPipeline, 
    SourceType,
    DomainVocabulary,
    HashEmbedder,
    DomainConceptEmbedder
)

def main():
    print("=" * 60)
    print("UNIFIED INDEXER - QUICK TEST")
    print("=" * 60)
    
    # Sample vocabulary (inline, no file needed)
    vocab_data = [
        {
            "keywords": "wire transfer,funds transfer,electronic transfer",
            "metadata": "payment-systems",
            "description": "Electronic transfer of funds between institutions",
            "related_keywords": "domestic wire,international wire",
            "business_capability": ["Payment Processing", "Wire Transfer"]
        },
        {
            "keywords": "OFAC,sanctions screening,sanctions check",
            "metadata": "compliance-fraud",
            "description": "Office of Foreign Assets Control screening",
            "related_keywords": "blocked persons,sdn list",
            "business_capability": ["OFAC Screening", "Sanctions Compliance"]
        },
        {
            "keywords": "MT-103,customer credit transfer",
            "metadata": "swift-mt-messages",
            "description": "SWIFT MT-103 message for customer transfers",
            "related_keywords": "swift message,credit transfer",
            "business_capability": ["MT-103 Processing", "SWIFT Messaging"]
        },
        {
            "keywords": "payment return,return payment",
            "metadata": "exception-handling",
            "description": "Processing of returned payments",
            "related_keywords": "refund,reversal",
            "business_capability": ["Payment Return", "Exception Handling"]
        }
    ]
    
    print("\n1. Creating pipeline with local embeddings...")
    pipeline = IndexingPipeline(
        vocabulary_data=vocab_data,
        embedder_type="hash"  # No fitting needed
    )
    print(f"   ✓ Pipeline created with {len(vocab_data)} vocabulary entries")
    print(f"   ✓ Embedder: {pipeline.embedder_type} ({pipeline.embedder.n_features} dimensions)")
    
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
    
    print("\n2. Indexing TAL code...")
    code_chunks = pipeline.index_content(
        tal_code.encode('utf-8'),
        'wire_transfer.tal',
        SourceType.CODE
    )
    print(f"   ✓ Indexed {len(code_chunks)} code chunks")
    
    for i, chunk in enumerate(code_chunks):
        concepts = [m.canonical_term for m in chunk.domain_matches]
        print(f"   Chunk {i+1}: {chunk.semantic_type.value}")
        if concepts:
            print(f"            Concepts: {concepts[:3]}")
    
    # Sample log entries
    log_entries = """
{"timestamp": "2025-01-02T10:15:30Z", "level": "INFO", "transaction_id": "TXN001", "message": "Processing wire transfer for $50,000"}
{"timestamp": "2025-01-02T10:15:31Z", "level": "INFO", "transaction_id": "TXN001", "message": "OFAC screening initiated"}
{"timestamp": "2025-01-02T10:15:32Z", "level": "ERROR", "transaction_id": "TXN001", "error_code": "OFAC_MATCH", "message": "Sanctions screening failed - potential match found"}
{"timestamp": "2025-01-02T10:16:00Z", "level": "INFO", "transaction_id": "TXN002", "message": "MT-103 customer credit transfer processed successfully"}
"""
    
    print("\n3. Indexing transaction logs...")
    log_chunks = pipeline.index_content(
        log_entries.encode('utf-8'),
        'transactions.log',
        SourceType.LOG
    )
    print(f"   ✓ Indexed {len(log_chunks)} log chunks")
    
    # Sample document
    doc_text = """
Wire Transfer Processing Guide

Overview
This document describes the wire transfer processing workflow including
MT-103 message handling and OFAC sanctions screening requirements.

OFAC Screening Requirements
All wire transfers must undergo OFAC sanctions screening before processing.
The screening checks against the SDN list and other prohibited party lists.

Payment Returns
When a payment is returned, the system must update the status and notify
the originating party.
"""
    
    print("\n4. Indexing documentation...")
    doc_chunks = pipeline.index_content(
        doc_text.encode('utf-8'),
        'wire_guide.txt',
        SourceType.DOCUMENT
    )
    print(f"   ✓ Indexed {len(doc_chunks)} document chunks")
    
    # Get statistics
    stats = pipeline.get_statistics()
    total = stats['pipeline']['total_chunks']
    print(f"\n   Total indexed: {total} chunks")
    
    # Search tests
    print("\n5. Testing search...")
    
    queries = [
        "OFAC sanctions screening",
        "wire transfer",
        "payment return"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = pipeline.search(query, top_k=3)
        
        if results:
            for r in results:
                print(f"      [{r.chunk.source_type.value:8}] Score: {r.combined_score:.3f}")
                print(f"                 Concepts: {r.matched_concepts[:3]}")
        else:
            print("      No results found")
    
    # Cross-reference search
    print("\n6. Cross-reference search (find code for log errors)...")
    xref = pipeline.search_cross_reference(
        "sanctions screening failed",
        from_type=SourceType.LOG,
        to_types=[SourceType.CODE, SourceType.DOCUMENT],
        top_k=2
    )
    
    for source_type, results in xref.items():
        print(f"\n   {source_type}:")
        for r in results:
            print(f"      Score: {r.combined_score:.3f} - {r.chunk.source_ref}")
    
    # Test domain embedder
    print("\n7. Testing domain concept embedder...")
    vocab = DomainVocabulary()
    vocab.load_from_data(vocab_data)
    
    domain_embedder = DomainConceptEmbedder(vocab)
    test_text = "Process wire transfer with OFAC sanctions screening"
    
    explanation = domain_embedder.explain_embedding(test_text)
    print(f"   Text: '{test_text}'")
    print(f"   Activated concepts:")
    for concept, weight in list(explanation.items())[:5]:
        print(f"      {concept}: {weight:.3f}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE - All components working!")
    print("=" * 60)
    
    print("""
Next steps:
1. Replace vocab_data with your vocabulary JSON file:
   pipeline = IndexingPipeline(vocabulary_path="your_vocab.json")

2. Index your codebase:
   pipeline.index_directory("./your_code", recursive=True)

3. Save/load index:
   pipeline.save("./my_index")
   pipeline.load("./my_index")
""")


if __name__ == "__main__":
    main()
