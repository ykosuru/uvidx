#!/usr/bin/env python3
"""
Unified Indexer CLI - Command-line interface for indexing and searching

Usage:
    # Index a directory
    python -m unified_indexer index --vocabulary vocab.json --directory ./code
    
    # Search the index
    python -m unified_indexer search --index ./index "OFAC screening"
    
    # Cross-reference search
    python -m unified_indexer xref --index ./index "payment failed" --from log --to code
"""

import argparse
import json
import sys
from pathlib import Path

from .models import SourceType
from .pipeline import IndexingPipeline, create_pipeline_with_sentence_transformers


def cmd_index(args):
    """Index files from a directory"""
    print(f"Initializing pipeline with vocabulary: {args.vocabulary}")
    
    # Create pipeline
    if args.embeddings == 'local':
        try:
            pipeline = create_pipeline_with_sentence_transformers(args.vocabulary)
        except ImportError:
            print("Warning: sentence-transformers not available, using concept-only search")
            pipeline = IndexingPipeline(vocabulary_path=args.vocabulary)
    else:
        pipeline = IndexingPipeline(vocabulary_path=args.vocabulary)
    
    # Parse extensions
    extensions = None
    if args.extensions:
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in args.extensions.split(',')]
    
    # Index directory
    print(f"\nIndexing directory: {args.directory}")
    
    def progress(file_path, current, total):
        if current % 10 == 0 or current == total:
            print(f"  [{current}/{total}] {Path(file_path).name}")
    
    stats = pipeline.index_directory(
        args.directory,
        recursive=not args.no_recursive,
        extensions=extensions,
        max_workers=args.workers,
        progress_callback=progress if args.verbose else None
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {stats.files_processed}")
    print(f"Files failed: {stats.files_failed}")
    print(f"Total chunks: {stats.total_chunks}")
    print(f"Time: {stats.processing_time_seconds:.2f}s")
    
    if stats.by_source_type:
        print("\nBy source type:")
        for st, count in stats.by_source_type.items():
            print(f"  {st}: {count} chunks")
    
    if stats.errors and args.verbose:
        print(f"\nErrors ({len(stats.errors)}):")
        for err in stats.errors[:10]:
            print(f"  - {err}")
    
    # Save index
    if args.output:
        print(f"\nSaving index to: {args.output}")
        pipeline.save(args.output)
    
    return 0


def cmd_search(args):
    """Search the index"""
    # Load pipeline
    pipeline = IndexingPipeline()
    pipeline.load(args.index)
    
    # Parse source types
    source_types = None
    if args.types:
        source_types = [SourceType(t) for t in args.types.split(',')]
    
    # Parse capabilities
    capabilities = None
    if args.capabilities:
        capabilities = args.capabilities.split(',')
    
    # Search
    query = ' '.join(args.query)
    print(f"Searching for: {query}")
    
    results = pipeline.search(
        query,
        top_k=args.top_k,
        source_types=source_types,
        capabilities=capabilities
    )
    
    # Display results
    print(f"\nFound {len(results)} results:\n")
    
    for result in results:
        print(f"{'='*60}")
        print(f"[{result.rank}] Score: {result.combined_score:.4f} ({result.retrieval_method})")
        print(f"Source: {result.chunk.source_ref}")
        print(f"Type: {result.chunk.source_type.value} / {result.chunk.semantic_type.value}")
        
        if result.matched_concepts:
            print(f"Concepts: {', '.join(result.matched_concepts[:5])}")
        
        if result.matched_capabilities:
            print(f"Capabilities: {', '.join(result.matched_capabilities[:5])}")
        
        # Show text preview
        text = result.chunk.text
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"\n{text}\n")
    
    return 0


def cmd_xref(args):
    """Cross-reference search"""
    # Load pipeline
    pipeline = IndexingPipeline()
    pipeline.load(args.index)
    
    # Parse source types
    from_type = SourceType(args.from_type)
    to_types = [SourceType(t) for t in args.to_types.split(',')]
    
    # Search
    query = ' '.join(args.query)
    print(f"Cross-referencing: {query}")
    print(f"From: {from_type.value} -> To: {[t.value for t in to_types]}")
    
    results = pipeline.search_cross_reference(
        query,
        from_type=from_type,
        to_types=to_types,
        top_k=args.top_k
    )
    
    # Display results
    for source_type, type_results in results.items():
        print(f"\n{'='*60}")
        print(f"SOURCE TYPE: {source_type} ({len(type_results)} results)")
        print(f"{'='*60}")
        
        for result in type_results:
            print(f"\n[Score: {result.combined_score:.4f}]")
            print(f"Source: {result.chunk.source_ref}")
            
            if result.matched_concepts:
                print(f"Concepts: {', '.join(result.matched_concepts[:3])}")
            
            text = result.chunk.text
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"Preview: {text}")
    
    return 0


def cmd_stats(args):
    """Show index statistics"""
    pipeline = IndexingPipeline()
    pipeline.load(args.index)
    
    pipeline.print_statistics()
    
    return 0


def cmd_capabilities(args):
    """List all business capabilities in the index"""
    pipeline = IndexingPipeline()
    pipeline.load(args.index)
    
    caps = pipeline.vocabulary.get_capabilities()
    
    print(f"Business Capabilities ({len(caps)}):\n")
    
    for cap in sorted(caps):
        results = pipeline.get_by_capability(cap, top_k=1)
        count = len(pipeline.index.concept_index.search_capability(cap))
        print(f"  {cap}: {count} chunks")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Unified Indexer - Index and search code, documents, and logs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index files from a directory')
    index_parser.add_argument('--vocabulary', '-v', required=True,
                             help='Path to vocabulary JSON file')
    index_parser.add_argument('--directory', '-d', required=True,
                             help='Directory to index')
    index_parser.add_argument('--output', '-o',
                             help='Directory to save index')
    index_parser.add_argument('--extensions', '-e',
                             help='Comma-separated list of extensions (e.g., .tal,.pdf)')
    index_parser.add_argument('--no-recursive', action='store_true',
                             help='Do not recurse into subdirectories')
    index_parser.add_argument('--workers', '-w', type=int, default=4,
                             help='Number of parallel workers')
    index_parser.add_argument('--embeddings', choices=['local', 'none'], default='none',
                             help='Embedding method (local = sentence-transformers)')
    index_parser.add_argument('--verbose', action='store_true',
                             help='Verbose output')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search the index')
    search_parser.add_argument('--index', '-i', required=True,
                              help='Path to index directory')
    search_parser.add_argument('query', nargs='+',
                              help='Search query')
    search_parser.add_argument('--top-k', '-k', type=int, default=10,
                              help='Number of results')
    search_parser.add_argument('--types', '-t',
                              help='Filter by source types (code,document,log)')
    search_parser.add_argument('--capabilities', '-c',
                              help='Filter by capabilities (comma-separated)')
    
    # Cross-reference command
    xref_parser = subparsers.add_parser('xref', help='Cross-reference search')
    xref_parser.add_argument('--index', '-i', required=True,
                            help='Path to index directory')
    xref_parser.add_argument('query', nargs='+',
                            help='Search query')
    xref_parser.add_argument('--from-type', '-f', required=True,
                            choices=['code', 'document', 'log'],
                            help='Primary source type')
    xref_parser.add_argument('--to-types', '-t', required=True,
                            help='Reference types (comma-separated)')
    xref_parser.add_argument('--top-k', '-k', type=int, default=5,
                            help='Results per type')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show index statistics')
    stats_parser.add_argument('--index', '-i', required=True,
                             help='Path to index directory')
    
    # Capabilities command
    caps_parser = subparsers.add_parser('capabilities', help='List business capabilities')
    caps_parser.add_argument('--index', '-i', required=True,
                            help='Path to index directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        'index': cmd_index,
        'search': cmd_search,
        'xref': cmd_xref,
        'stats': cmd_stats,
        'capabilities': cmd_capabilities
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
