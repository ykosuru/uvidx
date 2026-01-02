#!/usr/bin/env python3
"""
Index Search - Search through indexed PDF documents and TAL code

Usage:
    python search_index.py --index ./my_index --query "OFAC sanctions"
    python search_index.py --index ./my_index --query "wire transfer" --top 10
    python search_index.py --index ./my_index --query "payment" --type code
    python search_index.py --index ./my_index --interactive
    
    # With LLM analysis
    python search_index.py --index ./my_index --query "OFAC" --analyze
    python search_index.py --index ./my_index --query "OFAC" --analyze --provider openai
    python search_index.py --index ./my_index --query "OFAC" --analyze --min-score 0.60

Arguments:
    --index       Directory containing the saved index
    --query       Search query string
    --top         Number of results to return (default: 5)
    --type        Filter by source type: code, document, or all (default: all)
    --interactive Start interactive search mode
    --capability  Search by business capability instead of text
    --verbose     Show more details in results
    --analyze     Send results to LLM for analysis
    --provider    LLM provider: anthropic, openai, ollama, stub (default: anthropic)
    --model       LLM model name (provider-specific)
    --min-score   Minimum score for LLM analysis (default: 0.70)
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_indexer import IndexingPipeline, SourceType

# Import LLM provider
try:
    from llm_provider import (
        LLMProvider,
        create_provider,
        analyze_search_results,
        format_search_results_for_llm,
        WIRE_PAYMENTS_SYSTEM_PROMPT,
        ContentType
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Default keywords file location
DEFAULT_KEYWORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keywords.json")


def load_vocabulary(vocab_path: str) -> list:
    """Load vocabulary from JSON file"""
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        print(f"Please ensure 'keywords.json' exists in the same directory as this script,")
        print(f"or specify a custom vocabulary file with --vocab")
        sys.exit(1)
    
    with open(vocab_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('entries', [data])
    else:
        print(f"Error: Invalid vocabulary format in {vocab_path}")
        sys.exit(1)


def print_result(result, index: int, verbose: bool = False):
    """Print a single search result"""
    chunk = result.chunk
    
    print(f"\n{'─' * 60}")
    print(f"Result #{index + 1}  |  Score: {result.combined_score:.3f}  |  Type: {chunk.source_type.value.upper()}")
    print(f"{'─' * 60}")
    
    source_ref = chunk.source_ref
    if source_ref.file_path:
        print(f"📁 File: {source_ref.file_path}")
    
    if source_ref.line_start:
        line_info = f"Lines {source_ref.line_start}"
        if source_ref.line_end and source_ref.line_end != source_ref.line_start:
            line_info += f"-{source_ref.line_end}"
        print(f"📍 {line_info}")
    
    if source_ref.procedure_name:
        print(f"🔧 Procedure: {source_ref.procedure_name}")
    
    if source_ref.page_number:
        print(f"📄 Page: {source_ref.page_number}")
    
    if result.matched_concepts:
        concepts = result.matched_concepts[:5]
        print(f"🏷️  Concepts: {', '.join(concepts)}")
    
    capabilities = list(chunk.capability_set)[:3]
    if capabilities:
        print(f"💼 Capabilities: {', '.join(capabilities)}")
    
    print(f"\n📝 Content:")
    text = chunk.text.strip()
    
    max_len = 500 if verbose else 200
    if len(text) > max_len:
        text = text[:max_len] + "..."
    
    for line in text.split('\n')[:10]:
        print(f"   {line}")
    
    if verbose and chunk.metadata:
        print(f"\n🔍 Metadata: {chunk.metadata}")


def print_results(results, verbose: bool = False):
    """Print all search results"""
    if not results:
        print("\n⚠️  No results found.")
        return
    
    print(f"\n{'═' * 60}")
    print(f"Found {len(results)} result(s)")
    print(f"{'═' * 60}")
    
    for i, result in enumerate(results):
        print_result(result, i, verbose)
    
    print(f"\n{'═' * 60}")


def print_llm_analysis(response, verbose: bool = False):
    """Print LLM analysis response"""
    print(f"\n{'═' * 60}")
    print("🤖 LLM ANALYSIS")
    print(f"{'═' * 60}")
    
    if not response.success:
        print(f"\n❌ Error: {response.error}")
        return
    
    print(f"\nProvider: {response.provider} | Model: {response.model}")
    if response.tokens_used:
        print(f"Tokens used: {response.tokens_used}")
    
    print(f"\n{'─' * 60}")
    print(response.content)
    print(f"{'─' * 60}")


def search_once(pipeline: IndexingPipeline, 
                query: str, 
                top_k: int = 5,
                source_type: str = "all",
                verbose: bool = False):
    """Perform a single search"""
    
    source_types = None
    if source_type == "code":
        source_types = [SourceType.CODE]
    elif source_type == "document":
        source_types = [SourceType.DOCUMENT]
    elif source_type == "log":
        source_types = [SourceType.LOG]
    
    print(f"\n🔎 Searching for: \"{query}\"")
    if source_types:
        print(f"   Filtered to: {source_type}")
    
    results = pipeline.search(query, top_k=top_k, source_types=source_types)
    print_results(results, verbose)
    
    return results


def search_and_analyze(pipeline: IndexingPipeline,
                       query: str,
                       provider: 'LLMProvider',
                       top_k: int = 20,
                       source_type: str = "all",
                       min_score: float = 0.70,
                       verbose: bool = False):
    """Search and analyze with LLM"""
    
    source_types = None
    if source_type == "code":
        source_types = [SourceType.CODE]
    elif source_type == "document":
        source_types = [SourceType.DOCUMENT]
    elif source_type == "log":
        source_types = [SourceType.LOG]
    
    print(f"\n🔎 Searching for: \"{query}\"")
    if source_types:
        print(f"   Filtered to: {source_type}")
    print(f"   Min score for analysis: {min_score}")
    
    # Get more results for analysis
    results = pipeline.search(query, top_k=top_k, source_types=source_types)
    
    # Show results summary
    high_score_count = len([r for r in results if r.combined_score >= min_score])
    print(f"\n📊 Found {len(results)} results, {high_score_count} with score >= {min_score}")
    
    if verbose:
        print_results(results, verbose)
    else:
        # Show brief summary
        print("\nTop results:")
        for i, r in enumerate(results[:5]):
            chunk = r.chunk
            source = chunk.source_ref.file_path or "unknown"
            print(f"  {i+1}. [{r.combined_score:.3f}] {chunk.source_type.value}: {Path(source).name}")
    
    # Analyze with LLM
    print(f"\n🤖 Sending to LLM for analysis...")
    
    response = analyze_search_results(
        query=query,
        results=results,
        provider=provider,
        min_score=min_score,
        max_chunks=20,
        verbose=verbose
    )
    
    print_llm_analysis(response, verbose)
    
    return results, response


def search_by_capability(pipeline: IndexingPipeline,
                         capability: str,
                         top_k: int = 5,
                         verbose: bool = False):
    """Search by business capability"""
    print(f"\n🔎 Searching by capability: \"{capability}\"")
    
    results = pipeline.get_by_capability(capability, top_k=top_k)
    print_results(results, verbose)
    
    return results


def list_capabilities(pipeline: IndexingPipeline):
    """List all available business capabilities"""
    stats = pipeline.index.get_statistics()
    
    if 'concept_index' in stats and 'capabilities' in stats['concept_index']:
        capabilities = stats['concept_index']['capabilities']
        print("\n📋 Available Business Capabilities:")
        for cap in sorted(capabilities):
            print(f"   • {cap}")
    else:
        print("\n⚠️  No capabilities indexed yet.")


def interactive_mode(pipeline: IndexingPipeline, 
                     provider: 'LLMProvider' = None,
                     min_score: float = 0.70,
                     verbose: bool = False):
    """Run interactive search mode"""
    print("\n" + "=" * 60)
    print("INTERACTIVE SEARCH MODE")
    print("=" * 60)
    
    llm_status = "✓ enabled" if provider else "✗ disabled"
    print(f"""
Commands:
  <query>           Search for text
  :analyze <query>  Search and analyze with LLM ({llm_status})
  :cap <capability> Search by business capability
  :caps             List all capabilities
  :code <query>     Search only in code
  :doc <query>      Search only in documents
  :top <n>          Set number of results (default: 5)
  :minscore <n>     Set min score for LLM (default: {min_score})
  :verbose          Toggle verbose output
  :stats            Show index statistics
  :help             Show this help
  :quit             Exit

Examples:
  OFAC sanctions
  :analyze wire transfer processing
  :cap Payment Processing
  :code wire transfer
""")
    
    top_k = 5
    
    while True:
        try:
            query = input("\n🔍 Search> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in [":quit", ":exit", ":q"]:
            print("Goodbye!")
            break
        
        elif query.lower() == ":help":
            print(f"""
Commands:
  <query>           Search for text
  :analyze <query>  Search and analyze with LLM
  :cap <capability> Search by business capability
  :caps             List all capabilities  
  :code <query>     Search only in code
  :doc <query>      Search only in documents
  :top <n>          Set number of results
  :minscore <n>     Set min score for LLM analysis (current: {min_score})
  :verbose          Toggle verbose output
  :stats            Show index statistics
  :quit             Exit
""")
        
        elif query.lower() == ":verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
        
        elif query.lower() == ":caps":
            list_capabilities(pipeline)
        
        elif query.lower() == ":stats":
            stats = pipeline.get_statistics()
            print("\n📊 Index Statistics:")
            print(f"   Total chunks: {stats['pipeline']['total_chunks']}")
            print(f"   By type:")
            for t, count in stats['pipeline'].get('by_source_type', {}).items():
                print(f"      {t}: {count}")
            print(f"   Vocabulary entries: {stats['vocabulary'].get('total_entries', 0)}")
        
        elif query.lower().startswith(":top "):
            try:
                top_k = int(query[5:].strip())
                print(f"Results per query set to: {top_k}")
            except ValueError:
                print("Invalid number")
        
        elif query.lower().startswith(":minscore "):
            try:
                min_score = float(query[10:].strip())
                print(f"Min score for LLM analysis set to: {min_score}")
            except ValueError:
                print("Invalid number")
        
        elif query.lower().startswith(":analyze "):
            q = query[9:].strip()
            if not provider:
                print("❌ LLM analysis not available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
            else:
                search_and_analyze(pipeline, q, provider, top_k=20, 
                                   min_score=min_score, verbose=verbose)
        
        elif query.lower().startswith(":cap "):
            capability = query[5:].strip()
            search_by_capability(pipeline, capability, top_k, verbose)
        
        elif query.lower().startswith(":code "):
            q = query[6:].strip()
            search_once(pipeline, q, top_k, "code", verbose)
        
        elif query.lower().startswith(":doc "):
            q = query[5:].strip()
            search_once(pipeline, q, top_k, "document", verbose)
        
        elif query.startswith(":"):
            print(f"Unknown command: {query}. Type :help for available commands.")
        
        else:
            search_once(pipeline, query, top_k, "all", verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Search through indexed PDF documents and TAL code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search_index.py --index ./my_index --query "OFAC sanctions"
  python search_index.py --index ./my_index --query "wire transfer" --top 10
  python search_index.py --index ./my_index --query "payment" --type code
  python search_index.py --index ./my_index --interactive
  
  # With LLM analysis
  python search_index.py --index ./my_index --query "OFAC" --analyze
  python search_index.py --index ./my_index --query "OFAC" --analyze --provider openai
  python search_index.py --index ./my_index --query "wire transfer" --analyze --min-score 0.60
        """
    )
    
    parser.add_argument("--index", "-i", type=str, required=True, 
                        help="Directory containing the saved index")
    parser.add_argument("--query", "-q", type=str, 
                        help="Search query string")
    parser.add_argument("--top", "-n", type=int, default=5,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--type", "-t", type=str, default="all",
                        choices=["code", "document", "log", "all"],
                        help="Filter by source type (default: all)")
    parser.add_argument("--interactive", "-I", action="store_true",
                        help="Start interactive search mode")
    parser.add_argument("--capability", "-c", type=str,
                        help="Search by business capability")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show more details in results")
    parser.add_argument("--vocab", type=str, default=DEFAULT_KEYWORDS_FILE,
                        help="Path to vocabulary JSON file (default: keywords.json)")
    
    # LLM options
    parser.add_argument("--analyze", "-a", action="store_true",
                        help="Send results to LLM for analysis")
    parser.add_argument("--provider", "-p", type=str, default="anthropic",
                        choices=["anthropic", "openai", "ollama", "stub"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="LLM model name (provider-specific)")
    parser.add_argument("--min-score", type=float, default=0.70,
                        help="Minimum score for LLM analysis (default: 0.70)")
    
    args = parser.parse_args()
    
    # Validate
    if not args.interactive and not args.query and not args.capability:
        print("Error: Either --query, --capability, or --interactive is required")
        sys.exit(1)
    
    if not os.path.exists(args.index):
        print(f"Error: Index directory not found: {args.index}")
        sys.exit(1)
    
    print("=" * 60)
    print("UNIFIED INDEXER - SEARCH")
    print("=" * 60)
    
    # Load vocabulary
    print(f"\nLoading vocabulary from: {args.vocab}")
    vocab_data = load_vocabulary(args.vocab)
    print(f"Vocabulary entries: {len(vocab_data)}")
    
    # Create pipeline and load index
    print(f"Loading index from: {args.index}")
    pipeline = IndexingPipeline(
        vocabulary_data=vocab_data,
        embedder_type=None  # Will be restored from saved index
    )
    pipeline.load(args.index)
    
    stats = pipeline.get_statistics()
    total_chunks = stats['pipeline']['total_chunks']
    print(f"Index loaded: {total_chunks} chunks")
    
    # Setup LLM provider if needed
    llm_provider = None
    if args.analyze or args.interactive:
        if LLM_AVAILABLE:
            try:
                llm_provider = create_provider(args.provider, args.model)
                print(f"LLM Provider: {args.provider} ({llm_provider.model})")
            except Exception as e:
                print(f"⚠️  LLM setup failed: {e}")
                if args.analyze:
                    print("   Continuing without LLM analysis...")
        else:
            print("⚠️  LLM provider module not available")
    
    # Run search
    if args.interactive:
        interactive_mode(pipeline, llm_provider, args.min_score, args.verbose)
    elif args.analyze and args.query:
        if llm_provider:
            search_and_analyze(
                pipeline, args.query, llm_provider,
                top_k=20,
                source_type=args.type,
                min_score=args.min_score,
                verbose=args.verbose
            )
        else:
            print("❌ LLM analysis requires a valid provider. Check API keys.")
            search_once(pipeline, args.query, args.top, args.type, args.verbose)
    elif args.capability:
        search_by_capability(pipeline, args.capability, args.top, args.verbose)
    else:
        search_once(pipeline, args.query, args.top, args.type, args.verbose)


if __name__ == "__main__":
    main()
