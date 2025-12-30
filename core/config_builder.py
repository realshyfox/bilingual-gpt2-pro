"""
Configuration Builder
Builds and validates complete training configurations.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import yaml

from .model_registry import MODEL_PRESETS, get_model_preset


class ConfigBuilder:
    """Builds training configuration from various inputs."""
    
    def __init__(self):
        """Initialize config builder."""
        self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "dataset": {
                "train_path": None,
                "val_path": None,
                "analysis": {},
            },
            "task": {
                "type": "pre-training",
            },
            "tokenizer": {
                "type": "unigram",
                "vocab_size": 50257,
                "character_coverage": 0.9995,
                "special_tokens": ["<pad>", "<unk>", "<s>", "</s>"],
            },
            "model": {
                "type": "gpt2",
                "size_preset": "mini",
                "embed_dim": 768,
                "num_layers": 12,
                "num_heads": 12,
                "max_seq_len": 1024,
                "vocab_size": 50257,
                "dropout": 0.1,
                "use_flash_attention": True,
                "position_encoding": "rope",
                "activation": "gelu",
                "normalization": "layernorm",
            },
            "training": {
                "max_steps": 500000,
                "batch_size": 16,
                "gradient_accumulation_steps": 4,
                "learning_rate": 3.0e-4,
                "weight_decay": 0.01,
                "warmup_steps": 2000,
                "scheduler": "cosine",
                "precision": "bf16",
                "gradient_clipping": 1.0,
                "optimizer": "adamw",
                "zero_stage": 2,
                "log_interval": 100,
                "eval_interval": 5000,
                "save_interval": 10000,
            },
            "hardware": {
                "num_gpus": 1,
                "vram_per_gpu": 16,
                "supports_bf16": True,
                "supports_fp16": True,
            },
            "output": {
                "dir": "outputs/model",
                "save_total_limit": 3,
                "logging": {
                    "use_wandb": False,
                    "use_tensorboard": True,
                    "project_name": "bilingual-gpt2",
                },
            },
        }
    
    def from_wizard(
        self,
        hardware_config: Dict,
        dataset_analysis: Dict,
        model_size: str,
        context_window: int,
        precision: str,
        task_type: str = "pre-training"
    ) -> Dict:
        """
        Build config from wizard selections.
        
        Args:
            hardware_config: From HardwareDetector
            dataset_analysis: From SmartDatasetAnalyzer
            model_size: Model preset name
            context_window: Context window size
            precision: Training precision
            task_type: Task type
        
        Returns:
            Complete configuration dictionary
        """
        # Start with defaults
        config = self._get_default_config()
        
        # Apply hardware config
        config['hardware'].update({
            "num_gpus": hardware_config.get('num_gpus', 1),
            "vram_per_gpu": hardware_config.get('gpus', [{}])[0].get('total_memory_gb', 16),
            "supports_bf16": hardware_config.get('gpus', [{}])[0].get('supports_bf16', True),
        })
        
        # Apply dataset analysis
        config['dataset']['analysis'] = dataset_analysis
        config['dataset']['train_path'] = dataset_analysis.get('dataset_path')
        
        # Apply model preset
        preset = get_model_preset(model_size)
        config['model'].update({
            "size_preset": model_size,
            "embed_dim": preset['embed_dim'],
            "num_layers": preset['num_layers'],
            "num_heads": preset['num_heads'],
            "max_seq_len": context_window,
        })
        
        # Apply recommendations from dataset analysis
        recs = dataset_analysis.get('recommendations', {})
        if recs:
            config['tokenizer']['vocab_size'] = recs.get('recommended_vocab', 50257)
            config['model']['vocab_size'] = recs.get('recommended_vocab', 50257)
            config['training']['max_steps'] = recs.get('training_steps', 500000)
        
        # Apply precision
        config['training']['precision'] = precision
        
        # Adjust batch size based on model size and VRAM
        config['training']['batch_size'] = preset['recommended_batch']
        
        # Set task type
        config['task']['type'] = task_type
        
        return config
    
    def from_preset(self, preset_name: str) -> Dict:
        """Load config from preset file."""
        preset_path = Path(__file__).parent.parent / "configs" / "presets" / f"{preset_name}.yaml"
        
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_name}")
        
        with open(preset_path, 'r') as f:
            preset_config = yaml.safe_load(f)
        
        # Merge with defaults
        config = self._get_default_config()
        config = self._deep_merge(config, preset_config)
        
        return config
    
    def from_file(self, config_path: Union[str, Path]) -> Dict:
        """Load config from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Merge with defaults
        config = self._get_default_config()
        config = self._deep_merge(config, user_config)
        
        return config
    
    def set(self, key_path: str, value):
        """
        Set a configuration value using dot notation.
        
        Example:
            builder.set("model.embed_dim", 1024)
            builder.set("training.batch_size", 32)
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get(self, key_path: str, default=None):
        """
        Get a configuration value using dot notation.
        
        Example:
            embed_dim = builder.get("model.embed_dim")
        """
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def save(self, output_path: Union[str, Path]):
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def validate(self) -> tuple[bool, list]:
        """
        Validate current configuration.
        
        Returns:
            (is_valid, errors) tuple
        """
        errors = []
        
        # Check required fields
        if not self.config['dataset'].get('train_path'):
            errors.append("Missing dataset.train_path")
        
        if self.config['model']['embed_dim'] % self.config['model']['num_heads'] != 0:
            errors.append(
                f"embed_dim ({self.config['model']['embed_dim']}) must be "
                f"divisible by num_heads ({self.config['model']['num_heads']})"
            )
        
        if self.config['training']['batch_size'] < 1:
            errors.append("batch_size must be at least 1")
        
        if self.config['training']['learning_rate'] <= 0:
            errors.append("learning_rate must be positive")
        
        return len(errors) == 0, errors
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*70)
        print("  CONFIGURATION SUMMARY")
        print("="*70 + "\n")
        
        # Model
        print("📦 Model:")
        print(f"   Size: {self.config['model']['size_preset'].upper()}")
        print(f"   Parameters: ~{MODEL_PRESETS[self.config['model']['size_preset']]['params']}")
        print(f"   Context: {self.config['model']['max_seq_len']} tokens")
        print(f"   Vocab: {self.config['model']['vocab_size']:,}")
        
        # Training
        print("\n🎯 Training:")
        print(f"   Steps: {self.config['training']['max_steps']:,}")
        print(f"   Batch size: {self.config['training']['batch_size']}")
        print(f"   Gradient accumulation: {self.config['training']['gradient_accumulation_steps']}")
        print(f"   Precision: {self.config['training']['precision'].upper()}")
        print(f"   Learning rate: {self.config['training']['learning_rate']}")
        
        # Hardware
        print("\n💻 Hardware:")
        print(f"   GPUs: {self.config['hardware']['num_gpus']}")
        print(f"   VRAM per GPU: {self.config['hardware']['vram_per_gpu']:.1f} GB")
        
        # Output
        print("\n📁 Output:")
        print(f"   Directory: {self.config['output']['dir']}")
        
        print("\n" + "="*70 + "\n")
