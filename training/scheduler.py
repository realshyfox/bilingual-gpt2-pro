"""
Learning Rate Schedulers
"""

import math
import torch.optim as optim
from typing import Dict, Any


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any]
):
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Training configuration
    
    Returns:
        Scheduler instance
    """
    train_config = config.get('training', {})
    
    scheduler_type = train_config.get('scheduler', 'cosine').lower()
    max_steps = train_config.get('max_steps', 100000)
    warmup_steps = train_config.get('warmup_steps', 2000)
    
    if scheduler_type == 'cosine':
        return CosineAnnealingWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=max_steps,
            min_lr_ratio=train_config.get('min_lr_ratio', 0.1)
        )
    
    elif scheduler_type == 'linear':
        return LinearDecayWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=max_steps
        )
    
    elif scheduler_type == 'constant':
        return ConstantWithWarmup(
            optimizer,
            warmup_steps=warmup_steps
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


class CosineAnnealingWithWarmup:
    """Cosine annealing with linear warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        """Update learning rate."""
        self.step_count += 1
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.step_count < self.warmup_steps:
                # Linear warmup
                lr = base_lr * (self.step_count / self.warmup_steps)
            else:
                # Cosine decay
                progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                progress = min(progress, 1.0)
                lr = self.min_lr_ratio * base_lr + (base_lr - self.min_lr_ratio * base_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class LinearDecayWithWarmup:
    """Linear decay with linear warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        """Update learning rate."""
        self.step_count += 1
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.step_count < self.warmup_steps:
                # Linear warmup
                lr = base_lr * (self.step_count / self.warmup_steps)
            else:
                # Linear decay
                progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                progress = min(progress, 1.0)
                lr = base_lr * (1 - progress)
            
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class ConstantWithWarmup:
    """Constant learning rate with linear warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        """Update learning rate."""
        self.step_count += 1
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.step_count < self.warmup_steps:
                # Linear warmup
                lr = base_lr * (self.step_count / self.warmup_steps)
            else:
                # Constant
                lr = base_lr
            
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
