#!/usr/bin/env python3
"""
Dataset Analyzer CLI Tool
Analyze datasets without going through the full wizard.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dataset_analyzer import SmartDatasetAnalyzer


def main():
    """Main entry point for dataset analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze dataset for bilingual GPT-2 training"
    )
    
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to dataset directory or file"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["pre-training", "fine-tuning"],
        default="pre-training",
        help="Task type (default: pre-training)"
    )
    
    parser.add_argument(
        "--sampling",
        type=float,
        default=None,
        help="Sampling percentage (0-100, default: auto)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SmartDatasetAnalyzer(
        data_path=args.data_path,
        task_type=args.task,
        cache_enabled=not args.no_cache,
        verbose=not args.quiet
    )
    
    # Run analysis
    try:
        results = analyzer.analyze(sampling_percentage=args.sampling)
        
        # Print report
        if not args.quiet:
            analyzer.print_report()
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
