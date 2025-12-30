"""
Pre-training Trainer
Handles the training loop for pre-training GPT-2 models.
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import GPT2Model


class TextDataset(Dataset):
    """Simple text dataset for pre-training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to text files
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and tokenize data
        print(f"Loading data from {data_path}...")
        self.examples = []
        
        data_path = Path(data_path)
        if data_path.is_file():
            files = [data_path]
        else:
            files = list(data_path.rglob("*.txt"))
        
        for file_path in tqdm(files[:100], desc="Loading files"):  # Limit for demo
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Tokenize
                tokens = tokenizer.encode(text)
                
                # Split into chunks
                for i in range(0, len(tokens) - max_length, max_length):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) == max_length:
                        self.examples.append(chunk)
            
            except Exception as e:
                continue
        
        print(f"Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


class PreTrainer:
    """Pre-training trainer for GPT-2."""
    
    def __init__(
        self,
        model: GPT2Model,
        config: dict,
        output_dir: str
    ):
        """
        Initialize trainer.
        
        Args:
            model: GPT-2 model
            config: Training configuration
            output_dir: Output directory
        """
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training config
        train_config = config['training']
        self.max_steps = train_config.get('max_steps', 100000)
        self.batch_size = train_config.get('batch_size', 16)
        self.grad_accum_steps = train_config.get('gradient_accumulation_steps', 4)
        self.learning_rate = train_config.get('learning_rate', 3e-4)
        self.weight_decay = train_config.get('weight_decay', 0.01)
        self.log_interval = train_config.get('log_interval', 100)
        self.save_interval = train_config.get('save_interval', 10000)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')
        
        # State
        self.global_step = 0
        self.start_time = None
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ):
        """
        Run training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
        """
        print("\n" + "="*70)
        print("  STARTING TRAINING")
        print("="*70 + "\n")
        
        print(f"Device: {self.device}")
        print(f"Model: {self.model.num_parameters():,} parameters")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Effective batch size: {self.batch_size * self.grad_accum_steps}")
        print(f"Max steps: {self.max_steps:,}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Output dir: {self.output_dir}\n")
        
        self.model.train()
        self.start_time = time.time()
        
        running_loss = 0.0
        epoch = 0
        
        while self.global_step < self.max_steps:
            epoch += 1
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch}",
                total=min(len(train_dataloader), self.max_steps - self.global_step)
            )
            
            for step, batch in enumerate(progress_bar):
                if self.global_step >= self.max_steps:
                    break
                
                # Move to device
                batch = batch.to(self.device)
                
                # Forward pass
                logits, loss = self.model(batch, labels=batch)
                loss = loss / self.grad_accum_steps
                
                # Backward pass
                loss.backward()
                
                running_loss += loss.item()
                
                # Update weights
                if (step + 1) % self.grad_accum_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training'].get('gradient_clipping', 1.0)
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.log_interval == 0:
                        avg_loss = running_loss / self.log_interval
                        elapsed = time.time() - self.start_time
                        steps_per_sec = self.global_step / elapsed
                        
                        self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                        self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
                        
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'step/s': f'{steps_per_sec:.2f}'
                        })
                        
                        running_loss = 0.0
                    
                    # Save checkpoint
                    if self.global_step % self.save_interval == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        # Final save
        self.save_checkpoint("final")
        
        print("\n✅ Training complete!")
        print(f"Total time: {time.time() - self.start_time:.0f}s")
        print(f"Output directory: {self.output_dir}\n")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "model.pt"
        )
        
        # Save optimizer
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt"
        )
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
        }, checkpoint_dir / "training_state.pt")
        
        print(f"💾 Saved checkpoint: {checkpoint_dir}")
