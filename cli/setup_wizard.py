#!/usr/bin/env python3
"""
Interactive Setup Wizard
THE main entry point for configuring bilingual GPT-2 training.
Guides users through 10 steps to create optimal configuration.
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.hardware_detector import HardwareDetector
from core.dataset_analyzer import SmartDatasetAnalyzer
from core.parameter_validator import ParameterValidator
from core.config_builder import ConfigBuilder
from core.model_registry import MODEL_PRESETS, CONTEXT_OPTIONS

console = Console()


class SetupWizard:
    """Interactive setup wizard for bilingual GPT-2 training."""
    
    def __init__(self):
        """Initialize the wizard."""
        self.console = console
        self.hardware_config = None
        self.dataset_analysis = None
        self.config_builder = ConfigBuilder()
        self.step = 0
        self.total_steps = 10
    
    def run(self) -> Optional[str]:
        """
        Run the complete wizard.
        
        Returns:
            Path to saved config file, or None if cancelled
        """
        self._print_welcome()
        
        try:
            # Step 1: Hardware Detection
            self._step_hardware_detection()
            
            # Step 2: Task Selection
            self._step_task_selection()
            
            # Step 3: Dataset Location
            self._step_dataset_location()
            
            # Step 4: Dataset Analysis (CRITICAL!)
            self._step_dataset_analysis()
            
            # Step 5: Context Window
            self._step_context_window()
            
            # Step 6: Model Size
            self._step_model_size()
            
            # Step 7: Tokenizer
            self._step_tokenizer()
            
            # Step 8: Precision
            self._step_precision()
            
            # Step 9: Advanced Options
            self._step_advanced_options()
            
            # Step 10: Final Validation
            config_path = self._step_final_validation()
            
            return config_path
            
        except KeyboardInterrupt:
            self.console.print("\n\n⚠️  Setup cancelled by user", style="yellow")
            return None
        except Exception as e:
            self.console.print(f"\n\n❌ Error: {e}", style="red")
            return None
    
    def _print_welcome(self):
        """Print welcome message."""
        welcome_text = """
[bold cyan]BILINGUAL GPT-2 - INTELLIGENT SETUP WIZARD[/bold cyan]

This wizard will guide you through setting up your training configuration
in 10 easy steps. Each step includes recommendations based on your hardware
and dataset analysis.

[bold]Features:[/bold]
• 🔍 Automatic hardware detection
• 📊 Intelligent dataset analysis
• ✅ Parameter validation
• 🎯 Smart recommendations
• 📝 Educational explanations

Press Ctrl+C anytime to cancel.
        """
        
        panel = Panel(
            welcome_text,
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()
    
    def _print_step_header(self, title: str):
        """Print step header."""
        self.step += 1
        self.console.print()
        self.console.rule(f"[bold cyan]Step {self.step}/{self.total_steps}: {title}[/bold cyan]")
        self.console.print()
    
    def _step_hardware_detection(self):
        """Step 1: Detect hardware configuration."""
        self._print_step_header("Hardware Detection")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Detecting hardware...", total=None)
            
            detector = HardwareDetector(verbose=False)
            self.hardware_config = detector.detect()
            
            progress.update(task, completed=True)
        
        # Print detected config
        detector._print_config()
        
        # Confirm or override
        if not Confirm.ask("Is this configuration correct?", default=True):
            self.console.print("\n⚙️  Manual configuration:")
            num_gpus = IntPrompt.ask("Number of GPUs", default=self.hardware_config['num_gpus'])
            
            if num_gpus > 0:
                vram = FloatPrompt.ask(
                    "VRAM per GPU (GB)",
                    default=self.hardware_config['gpus'][0]['total_memory_gb'] if self.hardware_config['gpus'] else 16
                )
                
                self.hardware_config['num_gpus'] = num_gpus
                self.hardware_config['gpus'] = [{
                    'total_memory_gb': vram,
                    'supports_bf16': Confirm.ask("Supports BF16?", default=True)
                }]
    
    def _step_task_selection(self):
        """Step 2: Select training task."""
        self._print_step_header("Task Selection")
        
        self.console.print("What do you want to do?")
        self.console.print()
        self.console.print("1. [cyan]Pre-training[/cyan] - Train from scratch on text corpus")
        self.console.print("2. [cyan]Fine-tuning[/cyan] - Adapt existing model for specific task")
        self.console.print("3. [cyan]Both[/cyan] - Complete pipeline (pre-train then fine-tune)")
        self.console.print()
        
        choice = Prompt.ask("Your choice", choices=["1", "2", "3"], default="1")
        
        task_map = {"1": "pre-training", "2": "fine-tuning", "3": "both"}
        task = task_map[choice]
        
        self.config_builder.set("task.type", task)
        
        self.console.print(f"\n✅ Task set to: [bold]{task}[/bold]")
    
    def _step_dataset_location(self):
        """Step 3: Get dataset location."""
        self._print_step_header("Dataset Location")
        
        self.console.print("Where is your training data?")
        self.console.print()
        self.console.print("1. [cyan]Local directory[/cyan] - Path on this machine")
        self.console.print("2. [cyan]HuggingFace[/cyan] - Dataset from HF hub")
        self.console.print("3. [cyan]URL[/cyan] - Download from web")
        self.console.print()
        
        choice = Prompt.ask("Your choice", choices=["1", "2", "3"], default="1")
        
        if choice == "1":
            path = Prompt.ask("Enter local path")
            data_path = Path(path)
            
            if not data_path.exists():
                self.console.print(f"⚠️  Warning: Path does not exist: {path}", style="yellow")
                if not Confirm.ask("Continue anyway?", default=False):
                    return self._step_dataset_location()
            
            self.config_builder.set("dataset.train_path", str(data_path))
            self.dataset_path = data_path
            
        elif choice == "2":
            dataset_name = Prompt.ask("Enter HuggingFace dataset name")
            self.console.print(f"\n💡 Will download: {dataset_name}")
            self.config_builder.set("dataset.train_path", f"hf://{dataset_name}")
            self.dataset_path = None
            
        else:  # URL
            url = Prompt.ask("Enter dataset URL")
            self.console.print(f"\n💡 Will download from: {url}")
            self.config_builder.set("dataset.train_path", url)
            self.dataset_path = None
    
    def _step_dataset_analysis(self):
        """Step 4: Analyze dataset (CRITICAL!)."""
        self._print_step_header("Dataset Analysis")
        
        if self.dataset_path is None or not self.dataset_path.exists():
            self.console.print("⚠️  Cannot analyze remote dataset in wizard", style="yellow")
            self.console.print("Analysis will run when training starts")
            
            # Use dummy analysis
            self.dataset_analysis = {
                "dataset_path": str(self.config_builder.get("dataset.train_path")),
                "task_type": self.config_builder.get("task.type"),
                "dataset_type": "text_corpus",
                "recommendations": {
                    "recommended_vocab": 50257,
                    "tokenizer_type": "SentencePiece Unigram",
                    "training_steps": 500000,
                },
            }
            return
        
        self.console.print(f"📊 Analyzing dataset at: [cyan]{self.dataset_path}[/cyan]\n")
        
        # Create analyzer
        task_type = self.config_builder.get("task.type")
        analyzer = SmartDatasetAnalyzer(
            self.dataset_path,
            task_type=task_type,
            verbose=True
        )
        
        # Run analysis
        self.dataset_analysis = analyzer.analyze()
        
        # Print report
        analyzer.print_report()
        
        # Check compatibility
        if not self.dataset_analysis.get('task_compatible', True):
            self.console.print(
                f"⚠️  [bold red]WARNING:[/bold red] Dataset type "
                f"'{self.dataset_analysis['dataset_type']}' is NOT compatible with "
                f"'{task_type}'",
                style="yellow"
            )
            
            if not Confirm.ask("Continue anyway?", default=False):
                self.console.print("\n💡 Tip: Change task type or use different dataset")
                sys.exit(0)
        
        # Store analysis in config
        self.config_builder.set("dataset.analysis", self.dataset_analysis)
    
    def _step_context_window(self):
        """Step 5: Select context window."""
        self._print_step_header("Context Window")
        
        self.console.print("Select context window size:\n")
        
        # Show options with recommendations
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option", style="cyan", width=8)
        table.add_column("Size", justify="right")
        table.add_column("Best For")
        table.add_column("Speed")
        
        context_sizes = [512, 1024, 2048, 4096]
        
        for i, size in enumerate(context_sizes, 1):
            info = CONTEXT_OPTIONS[size]
            speed_emoji = {"very fast": "🚀", "fast": "⚡", "moderate": "🔄", "slow": "🐌"}
            speed = f"{speed_emoji.get(info['training_speed'], '')} {info['training_speed']}"
            
            default_marker = " ⭐" if size == 1024 else ""
            table.add_row(
                str(i) + default_marker,
                f"{size}",
                info['best_for'],
                speed
            )
        
        self.console.print(table)
        self.console.print()
        
        # Recommend based on dataset analysis
        if self.dataset_analysis:
            avg_length = self.dataset_analysis.get('token_stats', {}).get('avg_doc_length', 1000)
            
            if avg_length < 400:
                recommended = 512
            elif avg_length < 800:
                recommended = 1024
            elif avg_length < 1500:
                recommended = 2048
            else:
                recommended = 4096
            
            self.console.print(f"💡 Recommended: {recommended} (covers ~95% of your documents)")
            self.console.print()
        
        choice = Prompt.ask("Your choice", choices=["1", "2", "3", "4"], default="2")
        context_window = context_sizes[int(choice) - 1]
        
        self.config_builder.set("model.max_seq_len", context_window)
        
        self.console.print(f"\n✅ Context window: [bold]{context_window}[/bold] tokens")
        
        # Auto-select position encoding
        if context_window <= 2048:
            pos_encoding = "rope"
        elif context_window <= 4096:
            pos_encoding = "alibi"
        else:
            pos_encoding = "sliding"
        
        self.config_builder.set("model.position_encoding", pos_encoding)
        self.console.print(f"📍 Position encoding: [bold]{pos_encoding}[/bold]")
    
    def _step_model_size(self):
        """Step 6: Select model size."""
        self._print_step_header("Model Size")
        
        self.console.print("Select model size:\n")
        
        # Show model presets
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Option", style="cyan", width=8)
        table.add_column("Size")
        table.add_column("Parameters")
        table.add_column("Min VRAM")
        table.add_column("Quality")
        
        presets = ["tiny", "mini", "small", "medium"]
        available_vram = self.hardware_config.get('gpus', [{}])[0].get('total_memory_gb', 16)
        
        for i, name in enumerate(presets, 1):
            preset = MODEL_PRESETS[name]
            
            # Check if fits in memory
            fits = preset['min_vram'] <= available_vram
            size_str = name.upper()
            if name == "mini":
                size_str += " ⭐"
            if not fits:
                size_str += " ⚠️"
            
            quality_map = {"tiny": "Basic", "mini": "Good", "small": "Great", "medium": "Excellent"}
            
            table.add_row(
                str(i),
                size_str,
                preset['params'],
                f"{preset['min_vram']}GB",
                quality_map[name]
            )
        
        self.console.print(table)
        self.console.print()
        
        # Recommend based on VRAM
        if available_vram >= 28:
            recommended = "medium"
        elif available_vram >= 14:
            recommended = "small"
        elif available_vram >= 8:
            recommended = "mini"
        else:
            recommended = "tiny"
        
        self.console.print(f"💡 Recommended for {available_vram:.0f}GB VRAM: {recommended.upper()}")
        self.console.print()
        
        default_choice = str(presets.index(recommended) + 1)
        choice = Prompt.ask("Your choice", choices=["1", "2", "3", "4"], default=default_choice)
        
        model_size = presets[int(choice) - 1]
        preset = MODEL_PRESETS[model_size]
        
        # Apply preset to config
        self.config_builder.set("model.size_preset", model_size)
        self.config_builder.set("model.embed_dim", preset['embed_dim'])
        self.config_builder.set("model.num_layers", preset['num_layers'])
        self.config_builder.set("model.num_heads", preset['num_heads'])
        self.config_builder.set("training.batch_size", preset['recommended_batch'])
        
        self.console.print(f"\n✅ Model: [bold]{model_size.upper()}[/bold] ({preset['params']} parameters)")
    
    def _step_tokenizer(self):
        """Step 7: Select tokenizer."""
        self._print_step_header("Tokenizer")
        
        # Get recommendations from dataset analysis
        if self.dataset_analysis:
            recs = self.dataset_analysis.get('recommendations', {})
            recommended_vocab = recs.get('recommended_vocab', 50257)
            recommended_type = recs.get('tokenizer_type', 'SentencePiece Unigram')
        else:
            recommended_vocab = 50257
            recommended_type = 'SentencePiece Unigram'
        
        self.console.print(f"💡 Recommended: [bold]{recommended_type}[/bold]")
        self.console.print(f"💡 Recommended vocab size: [bold]{recommended_vocab:,}[/bold]")
        self.console.print()
        
        if "Unigram" in recommended_type:
            tokenizer_type = "unigram"
        else:
            tokenizer_type = "bpe"
        
        self.config_builder.set("tokenizer.type", tokenizer_type)
        self.config_builder.set("tokenizer.vocab_size", recommended_vocab)
        self.config_builder.set("model.vocab_size", recommended_vocab)
        
        # Ask if user wants to customize
        if Confirm.ask("Use recommended settings?", default=True):
            self.console.print(f"\n✅ Tokenizer: {tokenizer_type}, vocab={recommended_vocab:,}")
        else:
            custom_type = Prompt.ask(
                "Tokenizer type",
                choices=["bpe", "unigram", "wordpiece"],
                default=tokenizer_type
            )
            custom_vocab = IntPrompt.ask(
                "Vocabulary size",
                default=recommended_vocab
            )
            
            self.config_builder.set("tokenizer.type", custom_type)
            self.config_builder.set("tokenizer.vocab_size", custom_vocab)
            self.config_builder.set("model.vocab_size", custom_vocab)
            
            self.console.print(f"\n✅ Tokenizer: {custom_type}, vocab={custom_vocab:,}")
    
    def _step_precision(self):
        """Step 8: Select training precision."""
        self._print_step_header("Training Precision")
        
        # Get hardware capability
        supports_bf16 = self.hardware_config.get('gpus', [{}])[0].get('supports_bf16', False)
        
        self.console.print("Select training precision:\n")
        
        options = []
        if supports_bf16:
            self.console.print("1. [cyan]BF16[/cyan] - Best choice (2x faster, stable) ⭐")
            options.append("1")
        
        self.console.print("2. [cyan]FP16[/cyan] - Fast but less stable")
        options.append("2")
        
        self.console.print("3. [cyan]FP32[/cyan] - Slow but most stable")
        options.append("3")
        
        self.console.print()
        
        if supports_bf16:
            self.console.print("💡 Your GPU supports BF16 - highly recommended!")
            default = "1"
        else:
            self.console.print("💡 Your GPU doesn't support BF16, use FP16")
            default = "2"
        
        self.console.print()
        
        choice = Prompt.ask("Your choice", choices=options, default=default)
        
        precision_map = {"1": "bf16", "2": "fp16", "3": "fp32"}
        precision = precision_map[choice]
        
        self.config_builder.set("training.precision", precision)
        
        self.console.print(f"\n✅ Precision: [bold]{precision.upper()}[/bold]")
    
    def _step_advanced_options(self):
        """Step 9: Configure advanced options."""
        self._print_step_header("Advanced Options")
        
        self.console.print("Configure advanced features (recommended defaults shown):\n")
        
        # Flash Attention
        flash_attn = Confirm.ask(
            "Enable Flash Attention 2? (2-4x faster)",
            default=True
        )
        self.config_builder.set("model.use_flash_attention", flash_attn)
        
        # DeepSpeed ZeRO
        if self.hardware_config.get('num_gpus', 1) > 1:
            zero_stage = IntPrompt.ask(
                "DeepSpeed ZeRO stage (0=off, 2=optimizer shard, 3=full shard)",
                choices=["0", "2", "3"],
                default="2"
            )
            self.config_builder.set("training.zero_stage", int(zero_stage))
        
        # Gradient accumulation
        grad_accum = IntPrompt.ask(
            "Gradient accumulation steps (larger = more stable)",
            default=4
        )
        self.config_builder.set("training.gradient_accumulation_steps", grad_accum)
        
        self.console.print("\n✅ Advanced options configured")
    
    def _step_final_validation(self):
        """Step 10: Validate and save configuration."""
        self._print_step_header("Final Validation")
        
        # Build final config
        config = self.config_builder.config
        
        # Run parameter validator
        self.console.print("🔍 Validating configuration...\n")
        
        validator = ParameterValidator(config, verbose=True)
        success, checks = validator.validate_all()
        
        if not success:
            self.console.print("\n❌ Configuration has critical errors!", style="bold red")
            
            if Confirm.ask("Apply automatic fixes?", default=True):
                fixed_config = validator.apply_auto_fixes()
                self.config_builder.config = fixed_config
                
                self.console.print("\n🔧 Applied fixes, re-validating...\n")
                
                validator = ParameterValidator(fixed_config, verbose=True)
                success, checks = validator.validate_all()
                
                if not success:
                    self.console.print("\n❌ Still have errors - please review manually", style="red")
                    return None
        
        # Print summary
        self.config_builder.print_summary()
        
        # Save config
        output_dir = Path("configs")
        output_dir.mkdir(exist_ok=True)
        
        default_name = "my_config.yaml"
        config_name = Prompt.ask("Config filename", default=default_name)
        
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
        
        config_path = output_dir / config_name
        self.config_builder.save(config_path)
        
        self.console.print(f"\n✅ Configuration saved to: [bold cyan]{config_path}[/bold cyan]")
        
        # Show next steps
        self._print_next_steps(config_path)
        
        return str(config_path)
    
    def _print_next_steps(self, config_path: Path):
        """Print next steps for user."""
        next_steps = f"""
[bold cyan]🎉 Setup Complete![/bold cyan]

[bold]Next steps:[/bold]

1. Review your configuration:
   [cyan]cat {config_path}[/cyan]

2. Start training:
   [cyan]python train.py --config {config_path}[/cyan]

3. Monitor training:
   • TensorBoard: [cyan]tensorboard --logdir outputs/[/cyan]
   • Check logs in outputs/ directory

4. Need help?
   • Check docs/quickstart.md
   • Run: [cyan]python train.py --help[/cyan]

[bold green]Ready to train! 🚀[/bold green]
        """
        
        panel = Panel(
            next_steps,
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(panel)


def main():
    """Main entry point."""
    wizard = SetupWizard()
    config_path = wizard.run()
    
    if config_path:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
