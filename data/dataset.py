"""
Dataset classes for training.
"""

from pathlib import Path
from typing import Optional, List
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    """Dataset for pre-training on text corpus."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        stride: Optional[int] = None,
        cache_path: Optional[str] = None
    ):
        """
        Initialize text dataset.
        
        Args:
            data_path: Path to text files or directory
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for sliding window (default: max_length)
            cache_path: Path to cache tokenized data
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        
        self.examples = []
        
        # Load data
        print(f"📊 Loading data from {data_path}...")
        data_path = Path(data_path)
        
        if data_path.is_file():
            files = [data_path]
        else:
            # Find all text files
            extensions = ['.txt', '.text']
            files = []
            for ext in extensions:
                files.extend(list(data_path.rglob(f"*{ext}")))
        
        print(f"Found {len(files)} files")
        
        # Process files
        for file_path in tqdm(files, desc="Tokenizing"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # Tokenize
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                # Split into chunks with stride
                for i in range(0, len(tokens) - max_length + 1, self.stride):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) == max_length:
                        self.examples.append(chunk)
                
            except Exception as e:
                print(f"⚠️  Error processing {file_path}: {e}")
                continue
        
        print(f"✅ Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


class QADataset(Dataset):
    """Dataset for fine-tuning on Q&A pairs."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024
    ):
        """
        Initialize Q&A dataset.
        
        Args:
            data_path: Path to JSONL file with Q&A pairs
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        import json
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"📊 Loading Q&A data from {data_path}...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    
                    # Format as conversation
                    text = f"Q: {question}\nA: {answer}"
                    tokens = tokenizer.encode(text)
                    
                    if len(tokens) <= max_length:
                        # Pad to max_length
                        padded = tokens + [tokenizer.token_to_id('<pad>')] * (max_length - len(tokens))
                        self.examples.append(padded)
                
                except Exception:
                    continue
        
        print(f"✅ Loaded {len(self.examples)} Q&A examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    Create DataLoader with proper settings.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batch
    )
