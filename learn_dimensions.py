#!/usr/bin/env python3
"""
Learn Domain Dimensions

Discovers semantic dimensions from a corpus of documents.
Creates a dimensions.json file that can be used with the unified indexer.

Supports:
- PDF documents (extracts text from pages)
- TAL code files
- COBOL code files  
- Text/Markdown files
- Log files

Usage:
    # Learn from documents
    python learn_dimensions.py --input ./docs --output ./dimensions.json
    
    # Learn from TAL code
    python learn_dimensions.py --input ./code --extensions .tal .txt --output ./tal_dimensions.json
    
    # Custom number of dimensions
    python learn_dimensions.py --input ./docs --output ./dimensions.json --dims 100
    
    # Then use with indexer:
    python build_index.py --pdf-dir ./docs -o ./index --embedder learned --dimensions ./dimensions.json
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except ImportError:
        try:
            # Fallback to pdfplumber
            import pdfplumber
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts)
        except ImportError:
            print(f"  Warning: No PDF library available. Install pymupdf or pdfplumber.")
            return ""
    except Exception as e:
        print(f"  Warning: Could not extract text from {file_path}: {e}")
        return ""


def extract_text_from_file(file_path: str) -> Tuple[str, str]:
    """
    Extract text from a file based on its extension.
    
    Returns:
        Tuple of (text_content, file_type)
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    try:
        if ext == '.pdf':
            text = extract_text_from_pdf(file_path)
            return text, 'pdf'
        
        elif ext in ['.tal', '.cbl', '.cob', '.cobol']:
            # Code files - read as text
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            return text, 'code'
        
        elif ext in ['.txt', '.md', '.markdown', '.rst']:
            # Text/documentation files
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            return text, 'text'
        
        elif ext in ['.log', '.out']:
            # Log files
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            return text, 'log'
        
        elif ext in ['.json', '.xml', '.yaml', '.yml']:
            # Structured data files
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            return text, 'data'
        
        elif ext in ['.py', '.java', '.c', '.cpp', '.h', '.js', '.ts']:
            # Other code files
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            return text, 'code'
        
        else:
            # Try to read as text
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            return text, 'unknown'
            
    except Exception as e:
        print(f"  Warning: Could not read {file_path}: {e}")
        return "", 'error'


def collect_documents(
    input_dir: str,
    extensions: List[str],
    recursive: bool = True
) -> Tuple[List[str], Dict[str, int]]:
    """
    Collect and extract text from all documents in a directory.
    
    Returns:
        Tuple of (list of text contents, stats dict)
    """
    documents = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory not found: {input_dir}")
        return documents, {}
    
    # Normalize extensions
    exts = set()
    for ext in extensions:
        ext_clean = ext if ext.startswith('.') else f'.{ext}'
        exts.add(ext_clean.lower())
    
    # Find all matching files
    all_files = []
    
    if recursive:
        for f in input_path.rglob('*'):
            if f.is_file() and f.suffix.lower() in exts:
                all_files.append(f)
    else:
        for f in input_path.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                all_files.append(f)
    
    # Sort for consistent ordering
    all_files = sorted(set(all_files))
    
    print(f"\nFound {len(all_files)} files to process")
    
    # Extract text from each file
    stats = {'pdf': 0, 'code': 0, 'text': 0, 'log': 0, 'data': 0, 'other': 0, 'failed': 0}
    
    for i, file_path in enumerate(all_files):
        # Progress indicator
        if len(all_files) > 10 and (i + 1) % 10 == 0:
            print(f"  Processing file {i + 1}/{len(all_files)}...")
        
        text, file_type = extract_text_from_file(str(file_path))
        
        if text and text.strip():
            documents.append(text)
            if file_type in stats:
                stats[file_type] += 1
            else:
                stats['other'] += 1
        else:
            stats['failed'] += 1
    
    # Print stats
    print(f"\nExtracted text from:")
    if stats['pdf'] > 0:
        print(f"  PDF documents: {stats['pdf']}")
    if stats['code'] > 0:
        print(f"  Code files: {stats['code']}")
    if stats['text'] > 0:
        print(f"  Text files: {stats['text']}")
    if stats['log'] > 0:
        print(f"  Log files: {stats['log']}")
    if stats['data'] > 0:
        print(f"  Data files: {stats['data']}")
    if stats['other'] > 0:
        print(f"  Other files: {stats['other']}")
    if stats['failed'] > 0:
        print(f"  Failed/empty: {stats['failed']}")
    
    return documents, stats


def main():
    parser = argparse.ArgumentParser(
        description="Learn domain dimensions from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Learn from PDF and text documents
  python learn_dimensions.py --input ./docs --output dimensions.json
  
  # Learn from TAL code with more dimensions
  python learn_dimensions.py --input ./code --extensions .tal .txt -o tal_dims.json --dims 100
  
  # Learn from mixed content (PDFs, code, logs)
  python learn_dimensions.py --input ./project -e .pdf .tal .txt .log -o project_dims.json
  
  # Fine-tune parameters
  python learn_dimensions.py --input ./docs -o dims.json --dims 80 --min-freq 5 --window 100
  
  # Analyze existing dimensions
  python learn_dimensions.py --analyze dimensions.json
  
  # Test embedding with learned dimensions
  python learn_dimensions.py --analyze dimensions.json --test "OFAC sanctions screening"
        """
    )
    
    parser.add_argument("--input", "-i", type=str, help="Input directory to scan")
    parser.add_argument("--output", "-o", type=str, help="Output dimensions JSON file")
    parser.add_argument("--extensions", "-e", nargs="+", 
                        default=['.pdf', '.txt', '.tal', '.md', '.cbl', '.cob', '.log'],
                        help="File extensions to include (default: .pdf .txt .tal .md .cbl .cob .log)")
    parser.add_argument("--dims", "-d", type=int, default=80,
                        help="Number of dimensions to learn (default: 80)")
    parser.add_argument("--min-freq", type=int, default=3,
                        help="Minimum term frequency (default: 3)")
    parser.add_argument("--window", type=int, default=50,
                        help="Co-occurrence window size in characters (default: 50)")
    parser.add_argument("--no-bigrams", action="store_true",
                        help="Don't extract bigrams")
    parser.add_argument("--no-trigrams", action="store_true",
                        help="Don't extract trigrams")
    parser.add_argument("--recursive", "-r", action="store_true", default=True,
                        help="Scan directories recursively (default: True)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Don't scan recursively")
    parser.add_argument("--analyze", "-a", type=str,
                        help="Analyze existing dimensions file")
    parser.add_argument("--test", "-t", type=str,
                        help="Test embedding a text string")
    
    args = parser.parse_args()
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    from unified_indexer.learned_embeddings import (
        LearnedDomainEmbedder, 
        LearningConfig
    )
    
    # Analyze existing dimensions
    if args.analyze:
        print(f"\nAnalyzing dimensions from: {args.analyze}")
        embedder = LearnedDomainEmbedder.load(args.analyze)
        
        print(f"\nTotal dimensions: {embedder.n_dimensions}")
        print(f"Total terms: {sum(len(d.terms) for d in embedder.dimensions)}")
        
        print("\n" + "=" * 70)
        print("DIMENSION DETAILS")
        print("=" * 70)
        
        for dim in embedder.dimensions:
            top_terms = sorted(dim.term_weights.items(), key=lambda x: -x[1])[:8]
            terms_str = ", ".join(f"{t}({w:.2f})" for t, w in top_terms)
            print(f"\n{dim.id:3d}. {dim.name}")
            print(f"     Terms: {len(dim.terms)}, Docs: {dim.document_frequency}, Coherence: {dim.coherence_score:.3f}")
            print(f"     Top: {terms_str}")
        
        # Test embedding if requested
        if args.test:
            print("\n" + "=" * 70)
            print(f"TEST EMBEDDING: \"{args.test}\"")
            print("=" * 70)
            
            explanations = embedder.explain_embedding(args.test, top_k=10)
            if explanations:
                for dim_name, weight, matched_terms in explanations:
                    print(f"  {dim_name}: {weight:.3f} (matched: {', '.join(matched_terms)})")
            else:
                print("  No dimensions activated for this text.")
        
        return
    
    # Validate inputs for learning
    if not args.input:
        print("Error: --input directory required")
        sys.exit(1)
    
    if not args.output:
        print("Error: --output file required")
        sys.exit(1)
    
    if not os.path.isdir(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    recursive = not args.no_recursive
    
    print("=" * 70)
    print("LEARN DOMAIN DIMENSIONS")
    print("=" * 70)
    print(f"\nInput directory: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Extensions: {args.extensions}")
    print(f"Dimensions: {args.dims}")
    print(f"Min frequency: {args.min_freq}")
    print(f"Window size: {args.window}")
    print(f"Recursive: {recursive}")
    
    # Collect and extract documents
    documents, stats = collect_documents(args.input, args.extensions, recursive)
    
    if len(documents) < 3:
        print("\nError: Need at least 3 documents with text content to learn dimensions")
        sys.exit(1)
    
    print(f"\nTotal documents with text: {len(documents)}")
    
    # Adjust min_freq for small corpora
    min_freq = args.min_freq
    if len(documents) < 20 and min_freq > 2:
        min_freq = 2
        print(f"Note: Adjusted min_freq to {min_freq} for small corpus")
    
    # Configure and learn
    config = LearningConfig(
        n_dimensions=args.dims,
        min_term_frequency=min_freq,
        cooccurrence_window=args.window,
        extract_bigrams=not args.no_bigrams,
        extract_trigrams=not args.no_trigrams
    )
    
    print("\n" + "-" * 70)
    print("LEARNING DIMENSIONS")
    print("-" * 70)
    
    embedder = LearnedDomainEmbedder(config)
    embedder.fit(documents, verbose=True)
    
    # Create output directory if needed
    output_path = Path(args.output)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    embedder.save(args.output)
    
    print("\n" + "=" * 70)
    print("LEARNING COMPLETE")
    print("=" * 70)
    print(f"\nDimensions learned: {embedder.n_dimensions}")
    print(f"Output saved to: {args.output}")
    
    print("\nUsage with indexer:")
    print(f"  python build_index.py --pdf-dir ./docs -o ./index --embedder learned --dimensions {args.output}")
    
    # Test embedding if requested
    if args.test:
        print("\n" + "=" * 70)
        print(f"TEST EMBEDDING: \"{args.test}\"")
        print("=" * 70)
        
        explanations = embedder.explain_embedding(args.test, top_k=10)
        if explanations:
            for dim_name, weight, matched_terms in explanations:
                print(f"  {dim_name}: {weight:.3f} (matched: {', '.join(matched_terms)})")
        else:
            print("  No dimensions activated for this text.")


if __name__ == "__main__":
    main()
