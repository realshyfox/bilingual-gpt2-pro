"""
Unigram SentencePiece Tokenizer
Recommended for multilingual training (2+ languages).
"""

from pathlib import Path
from typing import List, Union, Optional
import tempfile

try:
    import sentencepiece as spm
except ImportError:
    spm = None

from .base import BaseTokenizer


class UnigramTokenizer(BaseTokenizer):
    """
    Unigram SentencePiece tokenizer.
    
    Best for multilingual scenarios with 2+ languages.
    More efficient than BPE (typically 13-15% smaller vocab for same coverage).
    """
    
    def __init__(self, vocab_size: int = 50257):
        """
        Initialize Unigram tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
        """
        super().__init__(vocab_size)
        
        if spm is None:
            raise ImportError(
                "sentencepiece is required for UnigramTokenizer. "
                "Install with: pip install sentencepiece"
            )
        
        self.sp = None
        self.model_path = None
    
    def train(
        self,
        texts: Union[List[str], str],
        vocab_size: int,
        character_coverage: float = 0.9995,
        model_type: str = 'unigram',
        **kwargs
    ) -> None:
        """
        Train Unigram tokenizer on corpus.
        
        Args:
            texts: Training texts (list) or path to corpus file
            vocab_size: Target vocabulary size
            character_coverage: Character coverage (0.9995 for multilingual)
            model_type: 'unigram' (recommended) or 'bpe'
            **kwargs: Additional SentencePiece parameters
        """
        self.vocab_size = vocab_size
        
        # Prepare input
        if isinstance(texts, str):
            input_file = texts
        else:
            # Write texts to temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                for text in texts:
                    f.write(text + '\n')
                input_file = f.name
        
        # Create temporary model path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.model') as f:
            model_prefix = f.name.replace('.model', '')
        
        # Training parameters
        train_params = {
            'input': input_file,
            'model_prefix': model_prefix,
            'vocab_size': vocab_size,
            'character_coverage': character_coverage,
            'model_type': model_type,
            'pad_id': 0,
            'unk_id': 1,
            'bos_id': 2,
            'eos_id': 3,
            'pad_piece': '<pad>',
            'unk_piece': '<unk>',
            'bos_piece': '<s>',
            'eos_piece': '</s>',
            'num_threads': kwargs.get('num_threads', 16),
        }
        
        # Add any additional parameters
        train_params.update(kwargs)
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(**train_params)
        
        # Load trained model
        self.model_path = f"{model_prefix}.model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)
        
        # Build vocab dictionaries
        self.vocab = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        self.reverse_vocab = {i: self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())}
        
        # Store special tokens
        self.special_tokens = {
            '<pad>': self.sp.pad_id(),
            '<unk>': self.sp.unk_id(),
            '<s>': self.sp.bos_id(),
            '</s>': self.sp.eos_id(),
        }
        
        self.trained = True
        
        print(f"✅ Trained Unigram tokenizer with {self.sp.get_piece_size()} tokens")
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        if not self.trained or self.sp is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        tokens = self.sp.encode(text, out_type=int)
        
        if add_bos:
            tokens = [self.sp.bos_id()] + tokens
        if add_eos:
            tokens = tokens + [self.sp.eos_id()]
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        if not self.trained or self.sp is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        return self.sp.decode(tokens)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of token ID lists
        """
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """
        Decode multiple token sequences.
        
        Args:
            token_lists: List of token ID lists
            
        Returns:
            List of decoded texts
        """
        return [self.decode(tokens) for tokens in token_lists]
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer to directory.
        
        Args:
            path: Output directory path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save SentencePiece model
        if self.model_path:
            import shutil
            shutil.copy(self.model_path, path / "tokenizer.model")
        
        # Save config
        import json
        config = {
            'vocab_size': self.vocab_size,
            'type': 'unigram',
            'special_tokens': self.special_tokens,
        }
        
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Saved tokenizer to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load tokenizer from directory.
        
        Args:
            path: Input directory path
        """
        path = Path(path)
        
        # Load SentencePiece model
        model_path = path / "tokenizer.model"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
        self.model_path = str(model_path)
        
        # Rebuild vocab
        self.vocab = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        self.reverse_vocab = {i: self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())}
        
        # Load config
        import json
        with open(path / "config.json", 'r') as f:
            config = json.load(f)
        
        self.vocab_size = config['vocab_size']
        self.special_tokens = config['special_tokens']
        self.trained = True
        
        print(f"✅ Loaded tokenizer from {path}")
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size."""
        if self.sp:
            return self.sp.get_piece_size()
        return self.vocab_size
