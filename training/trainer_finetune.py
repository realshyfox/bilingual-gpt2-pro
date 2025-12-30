"""
Fine-tuning Trainer
Handles the training loop for fine-tuning GPT-2 models on specific tasks.
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import GPT2Model


class FineTuneTrainer:
    """Fine-tuning trainer for GPT-2."""
    
    def __init__(
        self,
        model: GPT2Model,
        config: dict,
        output_dir: str,
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize fine-tuning trainer.
        
        Args:
            model: GPT-2 model
            config: Training configuration
            output_dir: Output directory
            pretrained_path: Path to pretrained checkpoint
        """
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load pretrained weights if provided
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            print("✅ Pretrained weights loaded")
        
        # Training config
        train_config = config['training']
        self.max_steps = train_config.get('max_steps', 10000)
        self.batch_size = train_config.get('batch_size', 8)
        self.grad_accum_steps = train_config.get('gradient_accumulation_steps', 4)
        self.learning_rate = train_config.get('learning_rate', 1e-4)  # Lower LR for fine-tuning
        self.weight_decay = train_config.get('weight_decay', 0.01)
        self.log_interval = train_config.get('log_interval', 50)
        self.eval_interval = train_config.get('eval_interval', 500)
        self.save_interval = train_config.get('save_interval', 1000)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer - often use lower LR for fine-tuning
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)  # Standard Adam betas for fine-tuning
        )
        
        # Learning rate scheduler - cosine with warmup
        self.warmup_steps = train_config.get('warmup_steps', 100)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')
        
        # State
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.start_time = None
    
    def get_lr(self):
        """Get learning rate with warmup and cosine decay."""
        if self.global_step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * (self.global_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.global_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.learning_rate * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ):
        """
        Run fine-tuning loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
        """
        print("\n" + "="*70)
        print("  STARTING FINE-TUNING")
        print("="*70 + "\n")
        
        print(f"Device: {self.device}")
        print(f"Model: {self.model.num_parameters():,} parameters")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Effective batch size: {self.batch_size * self.grad_accum_steps}")
        print(f"Max steps: {self.max_steps:,}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Warmup steps: {self.warmup_steps}")
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
                
                # Update learning rate
                current_lr = self.get_lr()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                
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
                        self.writer.add_scalar('train/lr', current_lr, self.global_step)
                        
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{current_lr:.2e}',
                            'step/s': f'{steps_per_sec:.2f}'
                        })
                        
                        running_loss = 0.0
                    
                    # Evaluation
                    if val_dataloader and self.global_step % self.eval_interval == 0:
                        eval_loss = self.evaluate(val_dataloader)
                        self.writer.add_scalar('val/loss', eval_loss, self.global_step)
                        
                        # Save best model
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint("best")
                            print(f"\n✨ New best model! Val loss: {eval_loss:.4f}")
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.save_interval == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        # Final save
        self.save_checkpoint("final")
        
        print("\n✅ Fine-tuning complete!")
        print(f"Total time: {time.time() - self.start_time:.0f}s")
        print(f"Best validation loss: {self.best_eval_loss:.4f}")
        print(f"Output directory: {self.output_dir}\n")
    
    def evaluate(self, val_dataloader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(self.device)
                logits, loss = self.model(batch, labels=batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
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
            'best_eval_loss': self.best_eval_loss,
        }, checkpoint_dir / "training_state.pt")
        
        if name != "final":
            print(f"💾 Saved checkpoint: {checkpoint_dir}")
