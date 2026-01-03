#!/usr/bin/env python3
"""
Learn Domain Dimensions

Discovers semantic dimensions from a corpus of documents.
Creates a dimensions.json file that can be used with the unified indexer.

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


def main():
    parser = argparse.ArgumentParser(
        description="Learn domain dimensions from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Learn from PDF/text documents
  python learn_dimensions.py --input ./docs --output dimensions.json
  
  # Learn from TAL code with more dimensions
  python learn_dimensions.py --input ./code --extensions .tal .txt -o tal_dims.json --dims 100
  
  # Fine-tune parameters
  python learn_dimensions.py --input ./docs -o dims.json --dims 80 --min-freq 5 --window 100
  
  # Analyze existing dimensions
  python learn_dimensions.py --analyze dimensions.json
        """
    )
    
    parser.add_argument("--input", "-i", type=str, help="Input directory to scan")
    parser.add_argument("--output", "-o", type=str, help="Output dimensions JSON file")
    parser.add_argument("--extensions", "-e", nargs="+", 
                        default=['.txt', '.tal', '.md', '.json', '.log'],
                        help="File extensions to include (default: .txt .tal .md .json .log)")
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
        LearningConfig,
        learn_dimensions_from_directory
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
            for dim_name, weight, matched_terms in explanations:
                print(f"  {dim_name}: {weight:.3f} (matched: {', '.join(matched_terms)})")
        
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
    
    # Collect files
    print("\nScanning for files...")
    input_path = Path(args.input)
    pattern = '**/*' if recursive else '*'
    
    all_files = []
    for ext in args.extensions:
        ext_clean = ext if ext.startswith('.') else f'.{ext}'
        files = list(input_path.glob(f'{pattern}{ext_clean}'))
        all_files.extend(files)
        if files:
            print(f"  {ext_clean}: {len(files)} files")
    
    if not all_files:
        print("Error: No files found with specified extensions")
        sys.exit(1)
    
    print(f"\nTotal files: {len(all_files)}")
    
    # Read documents
    print("\nReading documents...")
    documents = []
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                if content.strip():
                    documents.append(content)
        except Exception as e:
            print(f"  Warning: Could not read {file_path}: {e}")
    
    print(f"Read {len(documents)} documents")
    
    if len(documents) < 10:
        print("Warning: Very few documents. Consider adding more for better dimensions.")
    
    # Configure and learn
    config = LearningConfig(
        n_dimensions=args.dims,
        min_term_frequency=args.min_freq,
        cooccurrence_window=args.window,
        extract_bigrams=not args.no_bigrams,
        extract_trigrams=not args.no_trigrams
    )
    
    print("\nLearning dimensions...")
    embedder = LearnedDomainEmbedder(config)
    embedder.fit(documents, verbose=True)
    
    # Save
    embedder.save(args.output)
    
    print("\n" + "=" * 70)
    print("LEARNING COMPLETE")
    print("=" * 70)
    print(f"\nDimensions learned: {embedder.n_dimensions}")
    print(f"Output saved to: {args.output}")
    
    print("\nUsage with indexer:")
    print(f"  python build_index.py --input ./docs -o ./index --embedder learned --dimensions {args.output}")
    
    # Test embedding if requested
    if args.test:
        print("\n" + "=" * 70)
        print(f"TEST EMBEDDING: \"{args.test}\"")
        print("=" * 70)
        
        explanations = embedder.explain_embedding(args.test, top_k=10)
        for dim_name, weight, matched_terms in explanations:
            print(f"  {dim_name}: {weight:.3f} (matched: {', '.join(matched_terms)})")


if __name__ == "__main__":
    main()
