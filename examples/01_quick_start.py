#!/usr/bin/env python3
"""
Example 1: Quick Start
Demonstrates the simplest way to get started with bilingual GPT-2 training.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    HardwareDetector,
    SmartDatasetAnalyzer,
    ConfigBuilder,
    ParameterValidator
)


def main():
    """Simple example showing the full workflow."""
    
    print("\n" + "="*70)
    print("  BILINGUAL GPT-2 - QUICK START EXAMPLE")
    print("="*70 + "\n")
    
    # Step 1: Detect hardware
    print("Step 1: Detecting hardware...\n")
    detector = HardwareDetector(verbose=True)
    hardware_config = detector.detect()
    
    # Step 2: Analyze dataset (using example path)
    print("\nStep 2: Analyzing dataset...\n")
    
    # NOTE: Replace with your actual dataset path
    dataset_path = "/path/to/your/dataset"
    
    print(f"📝 Example dataset path: {dataset_path}")
    print("   (Replace with your actual dataset path)\n")
    
    # For demonstration, we'll skip actual analysis
    print("ℹ️  Skipping actual analysis for demo purposes")
    print("   In practice, you would run:")
    print(f"   analyzer = SmartDatasetAnalyzer('{dataset_path}')")
    print("   results = analyzer.analyze()\n")
    
    # Simulate analysis results
    dataset_analysis = {
        "dataset_path": dataset_path,
        "total_tokens": 2_800_000_000,
        "dataset_type": "text_corpus",
        "task_compatible": True,
        "recommendations": {
            "recommended_vocab": 45000,
            "tokenizer_type": "SentencePiece Unigram",
            "training_steps": 500000,
        }
    }
    
    # Step 3: Build configuration
    print("Step 3: Building configuration...\n")
    
    builder = ConfigBuilder()
    
    # Set basic parameters
    builder.set("dataset.train_path", dataset_path)
    builder.set("task.type", "pre-training")
    
    # Apply hardware config
    builder.set("hardware.num_gpus", hardware_config['num_gpus'])
    if hardware_config['gpus']:
        builder.set("hardware.vram_per_gpu", hardware_config['gpus'][0]['total_memory_gb'])
    
    # Apply dataset recommendations
    recs = dataset_analysis['recommendations']
    builder.set("tokenizer.vocab_size", recs['recommended_vocab'])
    builder.set("model.vocab_size", recs['recommended_vocab'])
    builder.set("training.max_steps", recs['training_steps'])
    
    # Set precision based on hardware
    recommended_precision = detector.get_recommended_precision()
    builder.set("training.precision", recommended_precision)
    
    print(f"✅ Configuration built")
    print(f"   • Model: {builder.get('model.size_preset').upper()}")
    print(f"   • Vocab size: {builder.get('tokenizer.vocab_size'):,}")
    print(f"   • Precision: {builder.get('training.precision').upper()}")
    print(f"   • GPUs: {builder.get('hardware.num_gpus')}")
    
    # Step 4: Validate configuration
    print("\nStep 4: Validating configuration...\n")
    
    validator = ParameterValidator(builder.config, verbose=True)
    success, checks = validator.validate_all()
    
    if not success:
        print("\n❌ Validation failed! Applying auto-fixes...\n")
        fixed_config = validator.apply_auto_fixes()
        builder.config = fixed_config
        
        # Re-validate
        validator = ParameterValidator(builder.config, verbose=True)
        success, checks = validator.validate_all()
    
    # Step 5: Save configuration
    print("\nStep 5: Saving configuration...\n")
    
    output_path = Path("configs/example_config.yaml")
    builder.save(output_path)
    
    print(f"✅ Configuration saved to: {output_path}")
    
    # Step 6: Print next steps
    print("\n" + "="*70)
    print("  NEXT STEPS")
    print("="*70 + "\n")
    
    print("1. Edit configuration (if needed):")
    print(f"   nano {output_path}\n")
    
    print("2. Start training:")
    print(f"   python train.py --config {output_path}\n")
    
    print("3. Monitor training:")
    print("   tensorboard --logdir outputs/\n")
    
    print("💡 For interactive setup, run:")
    print("   python cli/setup_wizard.py\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
