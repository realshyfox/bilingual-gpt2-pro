#!/usr/bin/env python3
"""
Tokenizer Training Script
Train BPE or Unigram tokenizer on corpus.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import create_tokenizer


def main():
    """Main entry point for tokenizer training."""
    parser = argparse.ArgumentParser(
        description="Train tokenizer on corpus"
    )
    
    parser.add_argument(
        "corpus_path",
        type=str,
        help="Path to training corpus"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        choices=["bpe", "unigram"],
        default="unigram",
        help="Tokenizer type (default: unigram)"
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
        help="Vocabulary size (default: 50257)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer",
        help="Output directory/file (default: tokenizer)"
    )
    
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage for SentencePiece (default: 0.9995)"
    )
    
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for BPE merges (default: 2)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  TOKENIZER TRAINING")
    print("="*70 + "\n")
    
    print(f"Corpus: {args.corpus_path}")
    print(f"Type: {args.type.upper()}")
    print(f"Vocab size: {args.vocab_size:,}")
    print(f"Output: {args.output}\n")
    
    # Create tokenizer
    tokenizer = create_tokenizer(
        tokenizer_type=args.type,
        vocab_size=args.vocab_size
    )
    
    # Train tokenizer
    print("Training tokenizer...\n")
    
    try:
        if args.type == 'unigram':
            tokenizer.train(
                texts=args.corpus_path,
                vocab_size=args.vocab_size,
                character_coverage=args.character_coverage
            )
        else:  # bpe
            tokenizer.train(
                texts=args.corpus_path,
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency
            )
        
        # Save tokenizer
        print(f"\nSaving tokenizer to: {args.output}")
        tokenizer.save(args.output)
        
        # Test tokenizer
        print("\n" + "="*70)
        print("  TESTING TOKENIZER")
        print("="*70 + "\n")
        
        test_texts = [
            "Hello world!",
            "This is a test.",
            "Bilingual GPT-2 training system."
        ]
        
        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            print(f"Original: {text}")
            print(f"Tokens:   {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"Decoded:  {decoded}")
            print()
        
        print("✅ Tokenizer training complete!\n")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
