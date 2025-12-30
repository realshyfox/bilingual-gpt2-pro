"""
Data Collators
Prepare batches for training.
"""

import torch
from typing import List, Dict, Any


class DataCollatorForLanguageModeling:
    """Collator for causal language modeling."""
    
    def __init__(self, pad_token_id: int = 0):
        """
        Initialize collator.
        
        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch.
        
        Args:
            examples: List of tokenized sequences
        
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Stack if all same length
        if all(len(ex) == len(examples[0]) for ex in examples):
            input_ids = torch.stack(examples)
            attention_mask = torch.ones_like(input_ids)
        else:
            # Pad to max length in batch
            max_len = max(len(ex) for ex in examples)
            input_ids = []
            attention_mask = []
            
            for ex in examples:
                pad_len = max_len - len(ex)
                input_ids.append(
                    torch.cat([ex, torch.full((pad_len,), self.pad_token_id)])
                )
                attention_mask.append(
                    torch.cat([torch.ones(len(ex)), torch.zeros(pad_len)])
                )
            
            input_ids = torch.stack(input_ids)
            attention_mask = torch.stack(attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
