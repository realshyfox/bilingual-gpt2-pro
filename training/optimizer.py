"""
Optimizer Setup
Factory functions for creating optimizers with proper configurations.
"""

import torch.optim as optim
from typing import Iterator, Dict, Any
import torch.nn as nn


def create_optimizer(
    parameters: Iterator[nn.Parameter],
    config: Dict[str, Any]
) -> optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        parameters: Model parameters
        config: Training configuration dictionary
    
    Returns:
        Optimizer instance
    """
    train_config = config.get('training', {})
    
    optimizer_type = train_config.get('optimizer', 'adamw').lower()
    learning_rate = train_config.get('learning_rate', 3e-4)
    weight_decay = train_config.get('weight_decay', 0.01)
    
    if optimizer_type == 'adamw':
        return optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(
                train_config.get('adam_beta1', 0.9),
                train_config.get('adam_beta2', 0.95)
            ),
            eps=train_config.get('adam_epsilon', 1e-8)
        )
    
    elif optimizer_type == 'adam':
        return optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(
                train_config.get('adam_beta1', 0.9),
                train_config.get('adam_beta2', 0.999)
            ),
            eps=train_config.get('adam_epsilon', 1e-8)
        )
    
    elif optimizer_type == 'sgd':
        return optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=train_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_names: tuple = ('bias', 'LayerNorm.weight', 'ln')
) -> list:
    """
    Create parameter groups with different weight decay settings.
    
    Typically, we don't want to apply weight decay to biases and layer norm parameters.
    
    Args:
        model: The model
        weight_decay: Weight decay value
        no_decay_names: Parameter names that should not have weight decay
    
    Returns:
        List of parameter groups
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
