"""
Base Tokenizer Class
Abstract base class for all tokenizer implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union
import json


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    def __init__(self, vocab_size: int = 50257):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {}
        self.trained = False
    
    @abstractmethod
    def train(
        self,
        texts: Union[List[str], str],
        vocab_size: int,
        **kwargs
    ) -> None:
        """
        Train tokenizer on corpus.
        
        Args:
            texts: Training texts or path to corpus
            vocab_size: Target vocabulary size
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer to file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'type': self.__class__.__name__
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load tokenizer from file.
        
        Args:
            path: Input file path
        """
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.vocab_size = config['vocab_size']
        self.vocab = config['vocab']
        self.special_tokens = config['special_tokens']
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.trained = True
    
    def add_special_tokens(self, tokens: List[str]) -> None:
        """
        Add special tokens to vocabulary.
        
        Args:
            tokens: List of special tokens
        """
        for token in tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.reverse_vocab[idx] = token
                self.special_tokens[token] = idx
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size."""
        return len(self.vocab)
    
    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.get_vocab_size()
