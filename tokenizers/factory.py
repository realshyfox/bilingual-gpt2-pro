"""
Tokenizer Factory
Creates tokenizer instances based on type specification.
"""

from typing import Optional

from .base import BaseTokenizer
from .bpe import BPETokenizer
from .unigram import UnigramTokenizer


def create_tokenizer(
    tokenizer_type: str,
    vocab_size: int = 50257,
    **kwargs
) -> BaseTokenizer:
    """
    Create tokenizer instance.
    
    Args:
        tokenizer_type: Type of tokenizer ('bpe', 'unigram', 'wordpiece')
        vocab_size: Target vocabulary size
        **kwargs: Additional tokenizer-specific parameters
        
    Returns:
        Tokenizer instance
        
    Raises:
        ValueError: If tokenizer type is unknown
    """
    tokenizer_type = tokenizer_type.lower()
    
    if tokenizer_type == 'bpe':
        return BPETokenizer(vocab_size=vocab_size)
    
    elif tokenizer_type == 'unigram':
        return UnigramTokenizer(vocab_size=vocab_size)
    
    elif tokenizer_type == 'wordpiece':
        # TODO: Implement WordPiece
        raise NotImplementedError(
            "WordPiece tokenizer not yet implemented. "
            "Use 'bpe' or 'unigram' instead."
        )
    
    else:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type}. "
            f"Available: 'bpe', 'unigram'"
        )


def get_recommended_tokenizer(num_languages: int = 1) -> str:
    """
    Get recommended tokenizer type based on number of languages.
    
    Args:
        num_languages: Number of languages in training data
        
    Returns:
        Recommended tokenizer type
    """
    if num_languages <= 2:
        return 'bpe'  # BPE works well for 1-2 languages
    else:
        return 'unigram'  # Unigram better for 3+ languages


__all__ = [
    'BaseTokenizer',
    'BPETokenizer',
    'UnigramTokenizer',
    'create_tokenizer',
    'get_recommended_tokenizer',
]
