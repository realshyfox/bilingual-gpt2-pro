#!/usr/bin/env python3
"""
Resource Estimator
Estimates training time and resource requirements.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ConfigBuilder, ParameterValidator, MODEL_PRESETS
from core.utils import format_time


def estimate_training_time(config: dict) -> dict:
    """Estimate training time and resources."""
    training = config['training']
    model = config['model']
    hardware = config['hardware']
    
    # Get model info
    preset = MODEL_PRESETS.get(model['size_preset'], MODEL_PRESETS['mini'])
    params = preset['params_int']
    
    # Training parameters
    max_steps = training['max_steps']
    batch_size = training['batch_size']
    grad_accum = training.get('gradient_accumulation_steps', 1)
    seq_len = model['max_seq_len']
    num_gpus = hardware['num_gpus']
    
    # Effective batch size
    effective_batch = batch_size * grad_accum * num_gpus
    
    # Tokens per step
    tokens_per_step = effective_batch * seq_len
    total_tokens = tokens_per_step * max_steps
    
    # Speed estimation (tokens/second)
    # Base speed for Mini model on 1x 4070Ti with BF16
    base_speed = 2000  # tokens/sec
    
    # Adjust for model size
    size_mult = {
        'tiny': 2.0,
        'mini': 1.0,
        'small': 0.4,
        'medium': 0.2
    }.get(model['size_preset'], 1.0)
    
    # Adjust for precision
    precision_mult = {
        'fp32': 0.5,
        'fp16': 1.0,
        'bf16': 1.0
    }.get(training.get('precision', 'bf16'), 1.0)
    
    # Adjust for GPUs (not perfectly linear)
    gpu_mult = min(num_gpus * 0.85, num_gpus)
    
    # Flash Attention speedup
    flash_mult = 1.5 if model.get('use_flash_attention') else 1.0
    
    # Final speed
    speed = base_speed * size_mult * precision_mult * gpu_mult * flash_mult
    
    # Training time
    training_time_sec = total_tokens / speed
    
    return {
        'total_tokens': total_tokens,
        'tokens_per_step': tokens_per_step,
        'effective_batch_size': effective_batch,
        'estimated_speed': speed,
        'training_time_sec': training_time_sec,
        'training_time_formatted': format_time(training_time_sec),
        'parameters': params,
    }


def main():
    """Estimate resources for training."""
    parser = argparse.ArgumentParser(
        description="Estimate training time and resources"
    )
    
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  RESOURCE ESTIMATOR")
    print("="*70 + "\n")
    
    # Load configuration
    builder = ConfigBuilder()
    try:
        config = builder.from_file(args.config_path)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)
    
    # Estimate resources
    estimates = estimate_training_time(config)
    
    # Print estimates
    print("📊 Training Estimates:\n")
    print(f"Model Parameters: {estimates['parameters']:,}")
    print(f"Effective Batch Size: {estimates['effective_batch_size']}")
    print(f"Tokens per Step: {estimates['tokens_per_step']:,}")
    print(f"Total Tokens: {estimates['total_tokens']:,}")
    print(f"\nEstimated Speed: {estimates['estimated_speed']:.0f} tokens/sec")
    print(f"Training Time: {estimates['training_time_formatted']}")
    
    # Memory estimate
    validator = ParameterValidator(config, verbose=False)
    
    print("\n" + "─"*70)
    print("\n💾 Memory Estimate:\n")
    
    # Run validation to get memory estimate
    success, checks = validator.validate_all()
    
    memory_check = next((c for c in checks if 'Memory' in c.name), None)
    if memory_check:
        print(memory_check.message)
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
