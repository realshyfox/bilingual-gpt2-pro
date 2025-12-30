#!/usr/bin/env python3
"""
Main Training Script
Launches training with specified configuration.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

from core import ConfigBuilder, ParameterValidator, HardwareDetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train bilingual GPT-2 model"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    # Override parameters
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--max-steps", type=int, help="Override max steps")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], help="Override precision")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    
    # Flags
    parser.add_argument("--no-validation", action="store_true", help="Skip parameter validation")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    return parser.parse_args()


def load_config(config_path: str, overrides: dict) -> dict:
    """Load and merge configuration."""
    builder = ConfigBuilder()
    config = builder.from_file(config_path)
    
    # Apply CLI overrides
    if overrides.get('batch_size'):
        builder.set("training.batch_size", overrides['batch_size'])
    if overrides.get('learning_rate'):
        builder.set("training.learning_rate", overrides['learning_rate'])
    if overrides.get('max_steps'):
        builder.set("training.max_steps", overrides['max_steps'])
    if overrides.get('precision'):
        builder.set("training.precision", overrides['precision'])
    if overrides.get('output_dir'):
        builder.set("output.dir", overrides['output_dir'])
    
    return builder.config


def validate_config(config: dict):
    """Validate configuration before training."""
    print("\n" + "="*70)
    print("  VALIDATING CONFIGURATION")
    print("="*70 + "\n")
    
    validator = ParameterValidator(config, verbose=True)
    success, checks = validator.validate_all()
    
    if not success:
        print("\n❌ Configuration validation FAILED!")
        print("Fix errors before training.\n")
        sys.exit(1)
    
    print("\n✅ Configuration validation PASSED!\n")


def main():
    """Main training entry point."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("  BILINGUAL GPT-2 TRAINING")
    print("="*70 + "\n")
    
    # Load configuration
    print(f"📝 Loading configuration from: {args.config}")
    
    overrides = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_steps': args.max_steps,
        'precision': args.precision,
        'output_dir': args.output_dir,
    }
    
    config = load_config(args.config, overrides)
    
    # Validate configuration
    if not args.no_validation:
        validate_config(config)
    
    # Detect hardware
    print("🔍 Detecting hardware...")
    detector = HardwareDetector(verbose=False)
    hardware_config = detector.detect()
    detector._print_config()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available, training will be VERY slow!")
        if not input("Continue? (y/n): ").lower().startswith('y'):
            sys.exit(1)
    
    print("\n" + "="*70)
    print("  STARTING TRAINING")
    print("="*70 + "\n")
    
    print("📋 Training configuration:")
    print(f"  • Model: {config['model']['size_preset'].upper()}")
    print(f"  • Steps: {config['training']['max_steps']:,}")
    print(f"  • Batch size: {config['training']['batch_size']}")
    print(f"  • Precision: {config['training']['precision'].upper()}")
    print(f"  • GPUs: {hardware_config['num_gpus']}")
    print(f"  • Output: {config['output']['dir']}\n")
    
    # Import training modules
    from models import create_model_from_config
    from training import PreTrainer, TextDataset
    from tokenizers import create_tokenizer
    from torch.utils.data import DataLoader
    
    # Create tokenizer
    print("📝 Creating tokenizer...")
    tokenizer = create_tokenizer(
        tokenizer_type=config['tokenizer']['type'],
        vocab_size=config['tokenizer']['vocab_size']
    )
    
    # Check if tokenizer is trained
    tokenizer_path = Path(config['output']['dir']) / "tokenizer"
    if tokenizer_path.exists():
        print(f"✅ Loading tokenizer from {tokenizer_path}")
        tokenizer.load(tokenizer_path)
    else:
        print("⚠️  Tokenizer not trained yet!")
        print(f"   Train with: python tokenizers/train_tokenizer.py {config['dataset']['train_path']}")
        print(f"   Then save to: {tokenizer_path}")
        print("\nFor now, skipping actual training...")
        return
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_model_from_config(config)
    
    # Create dataset
    print("\n📊 Creating dataset...")
    train_dataset = TextDataset(
        data_path=config['dataset']['train_path'],
        tokenizer=tokenizer,
        max_length=config['model']['max_seq_len']
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Create trainer
    print("\n🎯 Initializing trainer...")
    trainer = PreTrainer(
        model=model,
        config=config,
        output_dir=config['output']['dir']
    )
    
    # Start training
    trainer.train(train_dataloader)
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {config['output']['dir']}\n")


if __name__ == "__main__":
    main()
