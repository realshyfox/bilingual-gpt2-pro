#!/usr/bin/env python3
"""
Example 2: Custom Configuration
Shows how to create a configuration programmatically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ConfigBuilder, MODEL_PRESETS

def main():
    print("Example 2: Custom Configuration\n")
    
    # Create builder
    builder = ConfigBuilder()
    
    # Customize model
    builder.set("model.size_preset", "small")
    builder.set("model.max_seq_len", 2048)
    builder.set("model.use_flash_attention", True)
    
    # Customize training
    builder.set("training.batch_size", 8)
    builder.set("training.max_steps", 100000)
    builder.set("training.precision", "bf16")
    
    # Set paths
    builder.set("dataset.train_path", "~/data/my_corpus")
    builder.set("output.dir", "outputs/custom_model")
    
    # Print summary
    builder.print_summary()
    
    # Save
    output_path = Path("configs/custom_example.yaml")
    builder.save(output_path)
    print(f"✅ Saved to: {output_path}")

if __name__ == "__main__":
    main()
