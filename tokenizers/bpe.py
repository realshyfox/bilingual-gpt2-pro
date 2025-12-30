"""
Byte-Pair Encoding (BPE) Tokenizer
Good for single language or simple bilingual scenarios.
"""

from pathlib import Path
from typing import List, Union, Dict, Set
from collections import Counter, defaultdict
import json
import re

from .base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding tokenizer.
    
    Good for single language training.
    For multilingual (3+ languages), consider UnigramTokenizer instead.
    """
    
    def __init__(self, vocab_size: int = 50257):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
        """
        super().__init__(vocab_size)
        self.merges = {}
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    
    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        """
        Create mapping from bytes to unicode characters.
        Avoids mapping to whitespace/control characters.
        """
        bs = list(range(ord("!"), ord("~")+1)) + \
             list(range(ord("¡"), ord("¬")+1)) + \
             list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def _get_pairs(self, word: tuple) -> Set[tuple]:
        """Get all adjacent pairs of symbols in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def train(
        self,
        texts: Union[List[str], str],
        vocab_size: int,
        min_frequency: int = 2,
        **kwargs
    ) -> None:
        """
        Train BPE tokenizer on corpus.
        
        Args:
            texts: Training texts (list) or path to corpus file
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for merges
            **kwargs: Additional parameters
        """
        self.vocab_size = vocab_size
        
        # Load texts
        if isinstance(texts, str):
            with open(texts, 'r', encoding='utf-8') as f:
                corpus = f.read()
        else:
            corpus = '\n'.join(texts)
        
        # Basic tokenization (split on whitespace and punctuation)
        pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        tokens = re.findall(pattern, corpus)
        
        # Convert to bytes
        word_freqs = Counter(tokens)
        vocab = set()
        
        # Convert words to byte representations
        word_tokens = {}
        for word, freq in word_freqs.items():
            word_bytes = ''.join(self.byte_encoder[b] for b in word.encode('utf-8'))
            word_tokens[word] = list(word_bytes)
            vocab.update(word_bytes)
        
        # Initialize vocab with byte characters
        self.vocab = {char: idx for idx, char in enumerate(sorted(vocab))}
        
        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.special_tokens[token] = self.vocab[token]
        
        num_merges = vocab_size - len(self.vocab)
        
        print(f"Starting BPE training...")
        print(f"Initial vocab size: {len(self.vocab)}")
        print(f"Target vocab size: {vocab_size}")
        print(f"Number of merges: {num_merges}")
        
        # Perform BPE merges
        for merge_idx in range(num_merges):
            # Count pairs
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                word_split = word_tokens[word]
                if len(word_split) < 2:
                    continue
                pairs = self._get_pairs(tuple(word_split))
                for pair in pairs:
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            if pair_freqs[best_pair] < min_frequency:
                break
            
            # Merge the pair
            merged = ''.join(best_pair)
            self.merges[best_pair] = merged
            self.vocab[merged] = len(self.vocab)
            
            # Update word tokens
            for word in word_freqs:
                word_split = word_tokens[word]
                new_word = []
                i = 0
                while i < len(word_split):
                    if i < len(word_split) - 1 and \
                       (word_split[i], word_split[i+1]) == best_pair:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word_split[i])
                        i += 1
                word_tokens[word] = new_word
            
            if (merge_idx + 1) % 1000 == 0:
                print(f"  Merge {merge_idx + 1}/{num_merges}: {best_pair} -> {merged}")
        
        # Build reverse vocab
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.trained = True
        
        print(f"✅ Trained BPE tokenizer with {len(self.vocab)} tokens")
    
    def _bpe(self, token: str) -> List[str]:
        """Apply BPE merges to a token."""
        if not self.merges:
            return [token]
        
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return [token]
        
        while True:
            # Find pair to merge
            bigram = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
            
            if bigram not in self.merges:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i+1] == second:
                    new_word.append(self.merges[bigram])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        
        return list(word)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if not self.trained:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        # Tokenize
        pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        tokens = re.findall(pattern, text)
        
        # Encode
        ids = []
        for token in tokens:
            # Convert to bytes
            token_bytes = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # Apply BPE
            bpe_tokens = self._bpe(token_bytes)
            # Convert to IDs
            for bpe_token in bpe_tokens:
                ids.append(self.vocab.get(bpe_token, self.special_tokens.get('<unk>', 1)))
        
        return ids
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        if not self.trained:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        # Convert IDs to tokens
        text = ''.join(self.reverse_vocab.get(token, '<unk>') for token in tokens)
        
        # Convert from byte encoding
        text_bytes = bytearray([self.byte_decoder[c] for c in text if c in self.byte_decoder])
        
        return text_bytes.decode('utf-8', errors='replace')
    
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
            'merges': {str(k): v for k, v in self.merges.items()},
            'special_tokens': self.special_tokens,
            'type': 'bpe'
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved BPE tokenizer to {path}")
    
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
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = config['special_tokens']
        
        # Restore merges
        self.merges = {}
        for k, v in config['merges'].items():
            pair = eval(k)  # Convert string back to tuple
            self.merges[pair] = v
        
        self.trained = True
        
        print(f"✅ Loaded BPE tokenizer from {path}")
