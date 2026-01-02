#!/usr/bin/env python3
"""
Example Usage of the Unified Indexer

This script demonstrates:
1. Loading the domain vocabulary
2. Parsing different content types (code, documents, logs)
3. Building a hybrid index with LOCAL embeddings (no external APIs!)
4. Performing various search operations
5. Cross-referencing between content types
6. LLM integration for enhanced processing (optional)

Run from the parent directory:
    python examples/usage_example.py
"""

import json
import sys
from pathlib import Path
from typing import List

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
    IndexableChunk,
    LLMInterface,
    LLMEnhancedPipeline,
    # Local embeddings - no external APIs needed!
    create_embedder,
    HybridEmbedder,
    HashEmbedder,
    TFIDFEmbedder,
    DomainConceptEmbedder,
    BM25Embedder
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
# Example 5: Local Embeddings (No External APIs!)
# ============================================================

def example_local_embeddings():
    """Demonstrate local embeddings without any external API calls"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Local Embeddings (No External APIs)")
    print("="*60)
    
    # Sample vocabulary
    vocabulary_data = [
        {
            "keywords": "wire transfer,electronic transfer",
            "metadata": "payment-systems",
            "description": "Electronic transfer of funds",
            "related_keywords": "domestic wire,international wire",
            "business_capability": ["Payment Processing", "Wire Transfer"]
        },
        {
            "keywords": "OFAC,sanctions screening",
            "metadata": "compliance-fraud",
            "description": "OFAC sanctions screening",
            "related_keywords": "sanctions check,blocked persons",
            "business_capability": ["OFAC Screening", "Sanctions Compliance"]
        },
        {
            "keywords": "MT-103,customer credit transfer",
            "metadata": "swift-mt-messages",
            "description": "SWIFT MT-103 message",
            "related_keywords": "credit transfer,pacs.008",
            "business_capability": ["MT-103 Processing", "Inbound SWIFT"]
        }
    ]
    
    print("\n--- Available Embedder Types ---")
    print("""
    1. 'hash'   - Feature hashing (default, no fitting needed)
    2. 'hybrid' - Domain concepts + text features
    3. 'tfidf'  - TF-IDF with domain boosting (requires fitting)
    4. 'domain' - Pure domain concept matching
    5. 'bm25'   - BM25 ranking (requires fitting)
    """)
    
    # Example 1: Hash embedder (default, no fitting)
    print("\n--- 1. Hash Embedder (Default) ---")
    pipeline_hash = IndexingPipeline(
        vocabulary_data=vocabulary_data,
        embedder_type="hash"  # Default, works immediately
    )
    print(f"Embedder type: {pipeline_hash.embedder_type}")
    print(f"Embedding dimension: {pipeline_hash.embedder.n_features}")
    
    # Test embedding
    test_text = "Processing wire transfer with OFAC screening"
    embedding = pipeline_hash.embedder.get_embedding(test_text)
    print(f"Embedding sample (first 10 dims): {embedding[:10]}")
    
    # Example 2: Domain concept embedder
    print("\n--- 2. Domain Concept Embedder ---")
    vocab = DomainVocabulary()
    vocab.load_from_data(vocabulary_data)
    
    domain_embedder = DomainConceptEmbedder(vocab)
    print(f"Embedding dimension: {domain_embedder.n_dimensions} (one per concept)")
    
    embedding = domain_embedder.get_embedding(test_text)
    explanation = domain_embedder.explain_embedding(test_text)
    print(f"Activated concepts: {explanation}")
    
    # Example 3: Hybrid embedder
    print("\n--- 3. Hybrid Embedder (Domain + Text) ---")
    hybrid_embedder = HybridEmbedder(
        vocab,
        text_embedder="hash",
        text_dim=256,
        domain_weight=0.6,
        text_weight=0.4
    )
    print(f"Total dimensions: {hybrid_embedder.n_dimensions}")
    print(f"  Domain dimensions: {hybrid_embedder.domain_embedder.n_dimensions}")
    print(f"  Text dimensions: 256")
    
    # Example 4: TF-IDF embedder (requires fitting)
    print("\n--- 4. TF-IDF Embedder (Requires Fitting) ---")
    
    # Sample corpus to fit on
    corpus = [
        "Wire transfer processing with OFAC sanctions screening",
        "MT-103 customer credit transfer message handling",
        "Payment return and exception processing",
        "Sanctions compliance and blocked persons list check",
        "International wire transfer via SWIFT network"
    ]
    
    tfidf_embedder = TFIDFEmbedder(domain_vocabulary=vocab)
    tfidf_embedder.fit(corpus)
    print(f"Vocabulary size: {len(tfidf_embedder.vocabulary)}")
    
    embedding = tfidf_embedder.get_embedding(test_text)
    print(f"Non-zero features: {sum(1 for x in embedding if x > 0)}")
    
    # Example 5: Using pipeline with TF-IDF
    print("\n--- 5. Pipeline with TF-IDF Embedder ---")
    pipeline_tfidf = IndexingPipeline(
        vocabulary_data=vocabulary_data,
        embedder_type="tfidf"
    )
    
    # Fit the embedder on some documents
    pipeline_tfidf.fit_embedder(corpus)
    print("TF-IDF embedder fitted on corpus")
    
    # Now index content
    tal_code = """
    INT PROC PROCESS^WIRE^TRANSFER(transfer_record);
    BEGIN
        IF NOT OFAC^CHECK(transfer_record) THEN
            RETURN -1;
        END;
        CALL SEND^MT103(transfer_record);
        RETURN 0;
    END;
    """
    
    chunks = pipeline_tfidf.index_content(
        tal_code.encode('utf-8'),
        'wire.tal',
        SourceType.CODE
    )
    print(f"Indexed {len(chunks)} chunks with TF-IDF embeddings")
    
    # Search
    results = pipeline_tfidf.search("OFAC sanctions", top_k=3)
    print(f"Search results: {len(results)}")
    for r in results:
        print(f"  Score: {r.combined_score:.3f} - {r.matched_concepts}")
    
    # Example 6: Similarity computation
    print("\n--- 6. Computing Similarity ---")
    from unified_indexer import cosine_similarity, batch_cosine_similarity
    import numpy as np
    
    query = "wire transfer OFAC"
    doc1 = "Processing wire transfer with sanctions screening"
    doc2 = "MT-103 message format specification"
    doc3 = "OFAC blocked persons list update"
    
    query_emb = np.array(pipeline_hash.embedder.get_embedding(query))
    doc_embs = np.array([
        pipeline_hash.embedder.get_embedding(doc1),
        pipeline_hash.embedder.get_embedding(doc2),
        pipeline_hash.embedder.get_embedding(doc3)
    ])
    
    similarities = batch_cosine_similarity(query_emb, doc_embs)
    print(f"Query: '{query}'")
    print(f"  Similarity to doc1 (wire/sanctions): {similarities[0]:.3f}")
    print(f"  Similarity to doc2 (MT-103):         {similarities[1]:.3f}")
    print(f"  Similarity to doc3 (OFAC):           {similarities[2]:.3f}")
    
    print("\n" + "-"*40)
    print("Local Embeddings Summary")
    print("-"*40)
    print("""
    Key advantages of local embeddings:
    ✓ No external API calls
    ✓ No API keys or costs
    ✓ No network latency
    ✓ Works offline
    ✓ Domain-aware with vocabulary boosting
    ✓ Interpretable (especially domain embedder)
    
    Recommended choices:
    - Quick start: 'hash' (no fitting needed)
    - Best accuracy: 'hybrid' or 'tfidf' (fit on your corpus)
    - Interpretability: 'domain' (see which concepts matched)
    """)


# ============================================================
# Example 6: LLM Integration
# ============================================================

def example_llm_integration():
    """Demonstrate LLM integration for enhanced processing"""
    print("\n" + "="*60)
    print("EXAMPLE 6: LLM Integration (Optional)")
    print("="*60)
    
    # Define your LLM implementation
    class MyLLM(LLMInterface):
        """
        Example LLM implementation.
        Replace the invoke_llm and generate_embedding methods
        with your actual LLM provider calls.
        """
        
        def __init__(self):
            # Initialize your LLM client here
            # self.client = YourLLMClient(api_key="...")
            pass
        
        def invoke_llm(self,
                       user_prompt: str,
                       system_prompt: str = "",
                       content_type: str = "text") -> str:
            """
            Implement with your LLM provider.
            
            Content types help you customize behavior:
            - "text": General text processing
            - "code": Code analysis/generation
            - "embedding": Text for embedding
            - "extraction": Information extraction
            - "summarization": Content summarization
            - "classification": Content classification
            """
            # Example implementation (replace with your LLM):
            # 
            # if content_type == "code":
            #     # Use a model optimized for code
            #     model = "claude-3-sonnet"
            # else:
            #     model = "claude-3-haiku"
            # 
            # response = self.client.messages.create(
            #     model=model,
            #     system=system_prompt,
            #     messages=[{"role": "user", "content": user_prompt}]
            # )
            # return response.content[0].text
            
            # Stub response for demo
            print(f"    [LLM] invoke_llm called with content_type='{content_type}'")
            print(f"    [LLM] User prompt: {user_prompt[:100]}...")
            return f"[Stub response for {content_type}]"
        
        def generate_embedding(self, text: str) -> List[float]:
            """
            Implement with your embedding provider.
            """
            # Example implementation:
            # response = self.client.embeddings.create(
            #     input=text[:8000],
            #     model="text-embedding-3-small"
            # )
            # return response.data[0].embedding
            
            # Stub embedding for demo (normally 1536 dimensions for OpenAI)
            import hashlib
            # Generate deterministic pseudo-embedding from text hash
            h = hashlib.md5(text.encode()).hexdigest()
            return [float(int(h[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
    
    # Sample vocabulary
    vocabulary_data = [
        {
            "keywords": "wire transfer,electronic transfer",
            "metadata": "payment-systems",
            "description": "Electronic transfer of funds",
            "related_keywords": "domestic wire,international wire",
            "business_capability": ["Payment Processing", "Wire Transfer"]
        },
        {
            "keywords": "OFAC,sanctions screening",
            "metadata": "compliance-fraud",
            "description": "OFAC sanctions screening",
            "related_keywords": "sanctions check,blocked persons",
            "business_capability": ["OFAC Screening", "Sanctions Compliance"]
        }
    ]
    
    # Create LLM instance
    llm = MyLLM()
    
    # Create enhanced pipeline with LLM
    pipeline = LLMEnhancedPipeline(
        vocabulary_data=vocabulary_data,
        llm_interface=llm
    )
    
    print("\nCreated LLMEnhancedPipeline with custom LLM interface")
    
    # Sample code to analyze
    tal_code = """
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
    """
    
    # Index the code
    chunks = pipeline.index_content(
        tal_code.encode('utf-8'),
        'wire_transfer.tal',
        SourceType.CODE
    )
    print(f"\nIndexed {len(chunks)} code chunks")
    
    # Demonstrate LLM-enhanced features
    print("\n--- Query Enhancement ---")
    original_query = "wire transfer"
    enhanced_query = pipeline.enhance_query(original_query)
    print(f"Original: {original_query}")
    print(f"Enhanced: {enhanced_query}")
    
    print("\n--- Code Explanation ---")
    if chunks:
        explanation = pipeline.explain_code(chunks[0])
        print(f"Explanation: {explanation}")
    
    print("\n--- Business Rule Extraction ---")
    if chunks:
        rules = pipeline.extract_business_rules(chunks[0])
        print(f"Rules: {rules}")
    
    print("\n--- Result Summarization ---")
    results = pipeline.search("OFAC screening", top_k=3)
    summary = pipeline.summarize_results("OFAC screening", results)
    print(f"Summary: {summary}")
    
    print("\n" + "-"*40)
    print("LLM Integration Template")
    print("-"*40)
    print("""
To use with your LLM provider, implement LLMInterface:

    from unified_indexer import LLMInterface, LLMEnhancedPipeline
    
    class MyLLM(LLMInterface):
        def __init__(self, api_key):
            self.client = YourLLMClient(api_key=api_key)
        
        def invoke_llm(self, user_prompt, system_prompt="", content_type="text"):
            response = self.client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content
        
        def generate_embedding(self, text):
            response = self.client.embed(text)
            return response.embedding
    
    llm = MyLLM(api_key="your-api-key")
    pipeline = LLMEnhancedPipeline(
        vocabulary_path="vocab.json",
        llm_interface=llm
    )
""")


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
    example_local_embeddings()  # New! Local embeddings demo
    example_llm_integration()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
