#!/usr/bin/env python3
"""
Quick Train - One-command training utility
Trains a model with minimal configuration.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import HardwareDetector, SmartDatasetAnalyzer, ConfigBuilder


def main():
    """Quick training with automatic configuration."""
    parser = argparse.ArgumentParser(
        description="Quick training with automatic configuration"
    )
    
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to training data"
    )
    
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "mini", "small", "medium"],
        default="mini",
        help="Model size (default: mini)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/quick_train",
        help="Output directory (default: outputs/quick_train)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum training steps (default: 10000)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  QUICK TRAIN - Automatic Configuration")
    print("="*70 + "\n")
    
    # Step 1: Detect hardware
    print("🔍 Step 1/3: Detecting hardware...")
    detector = HardwareDetector(verbose=False)
    hw_config = detector.detect()
    print(f"✅ Found {hw_config['num_gpus']} GPU(s)\n")
    
    # Step 2: Analyze dataset
    print("📊 Step 2/3: Analyzing dataset...")
    analyzer = SmartDatasetAnalyzer(args.data_path, verbose=True)
    analysis = analyzer.analyze()
    print()
    
    # Step 3: Create config and train
    print("🚀 Step 3/3: Starting training...")
    
    builder = ConfigBuilder()
    builder.set("dataset.train_path", args.data_path)
    builder.set("model.size_preset", args.model_size)
    builder.set("training.max_steps", args.max_steps)
    builder.set("output.dir", args.output_dir)
    
    # Apply recommendations
    if analysis.get('recommendations'):
        recs = analysis['recommendations']
        builder.set("tokenizer.vocab_size", recs.get('recommended_vocab', 50257))
        builder.set("model.vocab_size", recs.get('recommended_vocab', 50257))
    
    # Save config
    config_path = Path(args.output_dir) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    builder.save(config_path)
    
    print(f"\n✅ Configuration saved to: {config_path}")
    print(f"\n🎯 Starting training with:")
    print(f"   Model: {args.model_size.upper()}")
    print(f"   Steps: {args.max_steps:,}")
    print(f"   Output: {args.output_dir}\n")
    
    # Launch training
    import subprocess
    result = subprocess.run([
        sys.executable,
        "train.py",
        "--config", str(config_path)
    ])
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
