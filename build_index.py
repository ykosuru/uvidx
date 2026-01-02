#!/usr/bin/env python3
"""
Index Builder - Indexes PDF documents and TAL code files

Usage:
    python build_index.py --pdf-dir ./docs --tal-dir ./code --output ./my_index
    python build_index.py --pdf-dir ./docs --tal-dir ./code --output ./my_index --vocab ./custom_vocab.json

Arguments:
    --pdf-dir    Directory containing PDF documents
    --tal-dir    Directory containing TAL code (.txt files)
    --output     Directory to save the index
    --vocab      (Optional) Path to vocabulary JSON file (default: keywords.json)
    --recursive  (Optional) Search directories recursively (default: True)
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_indexer import IndexingPipeline, SourceType

# Default keywords file location (same directory as this script)
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
    
    # Handle both formats: list or dict with 'entries' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('entries', [data])
    else:
        print(f"Error: Invalid vocabulary format in {vocab_path}")
        sys.exit(1)


def index_pdf_directory(pipeline: IndexingPipeline, 
                        pdf_dir: str, 
                        recursive: bool = True) -> dict:
    """Index all PDF files in a directory"""
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "chunks_created": 0,
        "errors": []
    }
    
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"Warning: PDF directory does not exist: {pdf_dir}")
        return stats
    
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(pdf_path.glob(pattern))
    # Also check for uppercase
    pdf_files.extend(pdf_path.glob(pattern.replace('.pdf', '.PDF')))
    pdf_files = list(set(pdf_files))  # Remove duplicates
    
    print(f"\nIndexing {len(pdf_files)} PDF files from {pdf_dir}...")
    
    for pdf_file in pdf_files:
        try:
            print(f"  Processing: {pdf_file.name}...", end=" ", flush=True)
            
            with open(pdf_file, 'rb') as f:
                content = f.read()
            
            chunks = pipeline.index_content(
                content,
                str(pdf_file),
                SourceType.DOCUMENT
            )
            
            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)
            print(f"✓ ({len(chunks)} chunks)")
            
        except Exception as e:
            stats["files_failed"] += 1
            stats["errors"].append(f"{pdf_file}: {str(e)}")
            print(f"✗ Error: {str(e)[:50]}")
    
    return stats


def index_tal_directory(pipeline: IndexingPipeline, 
                        tal_dir: str, 
                        recursive: bool = True) -> dict:
    """Index all TAL code files (.txt, .tal) in a directory"""
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "chunks_created": 0,
        "errors": []
    }
    
    tal_path = Path(tal_dir)
    if not tal_path.exists():
        print(f"Warning: TAL directory does not exist: {tal_dir}")
        return stats
    
    # Look for various extensions
    extensions = ["*.txt", "*.tal", "*.tacl", "*.TAL", "*.TXT", "*.TACL"]
    tal_files = []
    
    for ext in extensions:
        pattern = f"**/{ext}" if recursive else ext
        tal_files.extend(tal_path.glob(pattern))
    
    # Remove duplicates
    tal_files = list(set(tal_files))
    
    print(f"\nIndexing {len(tal_files)} TAL files from {tal_dir}...")
    
    for tal_file in tal_files:
        try:
            print(f"  Processing: {tal_file.name}...", end=" ", flush=True)
            
            with open(tal_file, 'rb') as f:
                content = f.read()
            
            chunks = pipeline.index_content(
                content,
                str(tal_file),
                SourceType.CODE
            )
            
            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)
            print(f"✓ ({len(chunks)} chunks)")
            
        except Exception as e:
            stats["files_failed"] += 1
            stats["errors"].append(f"{tal_file}: {str(e)}")
            print(f"✗ Error: {str(e)[:50]}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build search index from PDF documents and TAL code files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_index.py --pdf-dir ./docs --tal-dir ./code --output ./my_index
  python build_index.py --pdf-dir ./docs --tal-dir ./code --output ./my_index --vocab ./custom_vocab.json
  python build_index.py --tal-dir ./code --output ./my_index  # TAL only
  python build_index.py --pdf-dir ./docs --output ./my_index  # PDF only
        """
    )
    
    parser.add_argument("--pdf-dir", type=str, help="Directory containing PDF documents")
    parser.add_argument("--tal-dir", type=str, help="Directory containing TAL code (.txt files)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for index")
    parser.add_argument("--vocab", "-v", type=str, default=DEFAULT_KEYWORDS_FILE,
                        help=f"Path to vocabulary JSON file (default: keywords.json)")
    parser.add_argument("--recursive", "-r", action="store_true", default=True, 
                        help="Search directories recursively (default: True)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search recursively")
    parser.add_argument("--embedder", "-e", type=str, default="hash",
                        choices=["hash", "hybrid", "tfidf", "domain", "bm25"],
                        help="Embedder type (default: hash)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.pdf_dir and not args.tal_dir:
        print("Error: At least one of --pdf-dir or --tal-dir is required")
        sys.exit(1)
    
    recursive = not args.no_recursive
    
    print("=" * 60)
    print("UNIFIED INDEXER - BUILD INDEX")
    print("=" * 60)
    
    # Load vocabulary from keywords.json
    print(f"\nLoading vocabulary from: {args.vocab}")
    vocab_data = load_vocabulary(args.vocab)
    print(f"Vocabulary entries: {len(vocab_data)}")
    
    # Create pipeline
    print(f"\nInitializing pipeline with '{args.embedder}' embedder...")
    pipeline = IndexingPipeline(
        vocabulary_data=vocab_data,
        embedder_type=args.embedder
    )
    
    total_stats = {
        "pdf": {"files_processed": 0, "chunks_created": 0, "files_failed": 0},
        "tal": {"files_processed": 0, "chunks_created": 0, "files_failed": 0}
    }
    
    # Index PDFs
    if args.pdf_dir:
        pdf_stats = index_pdf_directory(pipeline, args.pdf_dir, recursive)
        total_stats["pdf"] = pdf_stats
    
    # Index TAL code
    if args.tal_dir:
        tal_stats = index_tal_directory(pipeline, args.tal_dir, recursive)
        total_stats["tal"] = tal_stats
    
    # Save index
    print(f"\nSaving index to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    pipeline.save(args.output)
    
    # Also save reference to vocabulary used
    index_meta = {
        "vocabulary_file": os.path.basename(args.vocab),
        "embedder_type": args.embedder,
        "pdf_dir": args.pdf_dir,
        "tal_dir": args.tal_dir,
        "stats": total_stats
    }
    with open(os.path.join(args.output, "index_meta.json"), 'w') as f:
        json.dump(index_meta, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    
    print(f"\nPDF Documents:")
    print(f"  Files processed: {total_stats['pdf'].get('files_processed', 0)}")
    print(f"  Files failed:    {total_stats['pdf'].get('files_failed', 0)}")
    print(f"  Chunks created:  {total_stats['pdf'].get('chunks_created', 0)}")
    
    print(f"\nTAL Code:")
    print(f"  Files processed: {total_stats['tal'].get('files_processed', 0)}")
    print(f"  Files failed:    {total_stats['tal'].get('files_failed', 0)}")
    print(f"  Chunks created:  {total_stats['tal'].get('chunks_created', 0)}")
    
    total_chunks = (total_stats['pdf'].get('chunks_created', 0) + 
                    total_stats['tal'].get('chunks_created', 0))
    print(f"\nTotal chunks indexed: {total_chunks}")
    print(f"Index saved to: {args.output}")
    
    # Print errors if any
    all_errors = total_stats['pdf'].get('errors', []) + total_stats['tal'].get('errors', [])
    if all_errors:
        print(f"\n⚠️  Errors encountered ({len(all_errors)}):")
        for err in all_errors[:5]:
            print(f"   {err}")
        if len(all_errors) > 5:
            print(f"   ... and {len(all_errors) - 5} more")
    
    print(f"\nTo search the index, run:")
    print(f"  python search_index.py --index {args.output} --query \"your search query\"")


if __name__ == "__main__":
    main()
