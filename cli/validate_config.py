#!/usr/bin/env python3
"""
Configuration Validator
Validates training configurations without starting training.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ConfigBuilder, ParameterValidator


def main():
    """Validate configuration file."""
    parser = argparse.ArgumentParser(
        description="Validate training configuration"
    )
    
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply automatic fixes"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  CONFIGURATION VALIDATOR")
    print("="*70 + "\n")
    
    # Load configuration
    print(f"📝 Loading: {args.config_path}\n")
    
    builder = ConfigBuilder()
    try:
        config = builder.from_file(args.config_path)
    except FileNotFoundError:
        print(f"❌ File not found: {args.config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)
    
    # Validate
    validator = ParameterValidator(config, verbose=True)
    success, checks = validator.validate_all()
    
    if not success:
        print("\n" + "="*70)
        
        if args.fix:
            print("🔧 Applying automatic fixes...\n")
            fixed_config = validator.apply_auto_fixes()
            
            # Re-validate
            validator2 = ParameterValidator(fixed_config, verbose=True)
            success2, checks2 = validator2.validate_all()
            
            if success2:
                # Save fixed config
                fixed_path = Path(args.config_path).with_suffix('.fixed.yaml')
                builder.config = fixed_config
                builder.save(fixed_path)
                
                print(f"\n✅ Fixed configuration saved to: {fixed_path}")
                print("   Review changes and rename to use.\n")
                sys.exit(0)
            else:
                print("\n❌ Could not auto-fix all errors")
                sys.exit(1)
        else:
            print("💡 Run with --fix to apply automatic fixes\n")
            sys.exit(1)
    else:
        print("\n" + "="*70)
        print("✅ Configuration is valid and ready for training!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
