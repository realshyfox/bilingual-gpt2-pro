"""
Parameter Validator
Validates ALL parameter combinations BEFORE training to prevent errors.
Fail fast, fail clear with educational error messages.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from .model_registry import MODEL_PRESETS, CONTEXT_OPTIONS, PRECISION_CONFIG


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    passed: bool
    severity: str  # 'critical', 'warning', 'info'
    message: str
    suggestion: Optional[str] = None
    auto_fix_available: bool = False
    auto_fix_value: Optional[any] = None


class ParameterValidator:
    """
    Validates training configuration parameters before GPU time is wasted.
    Provides educational error messages and auto-fix suggestions.
    """
    
    def __init__(self, config: Dict, verbose: bool = True):
        """
        Initialize parameter validator.
        
        Args:
            config: Training configuration dictionary
            verbose: Whether to print validation results
        """
        self.config = config
        self.verbose = verbose
        self.checks: List[ValidationCheck] = []
        self.critical_failures = 0
        self.warnings = 0
    
    def validate_all(self) -> Tuple[bool, List[ValidationCheck]]:
        """
        Run all validation checks.
        
        Returns:
            (success, checks) tuple
        """
        if self.verbose:
            print("\n" + "="*70)
            print("  PARAMETER VALIDATION")
            print("="*70 + "\n")
        
        # Run all validations
        self._validate_embed_dim_divisibility()
        self._validate_memory_requirements()
        self._validate_context_window_compatibility()
        self._validate_tokenizer_language_match()
        self._validate_dataset_model_size()
        self._validate_precision_hardware()
        self._validate_batch_size()
        
        # Count failures
        self.critical_failures = sum(
            1 for c in self.checks if not c.passed and c.severity == 'critical'
        )
        self.warnings = sum(
            1 for c in self.checks if not c.passed and c.severity == 'warning'
        )
        
        success = self.critical_failures == 0
        
        if self.verbose:
            self._print_results()
        
        return success, self.checks
    
    def apply_auto_fixes(self) -> Dict:
        """
        Apply automatic fixes to config where available.
        
        Returns:
            Updated config dictionary
        """
        fixed_config = self.config.copy()
        
        for check in self.checks:
            if not check.passed and check.auto_fix_available:
                # Apply fix based on check name
                if "embed_dim" in check.name.lower():
                    fixed_config['model']['embed_dim'] = check.auto_fix_value
                elif "batch" in check.name.lower():
                    fixed_config['training']['batch_size'] = check.auto_fix_value
                elif "precision" in check.name.lower():
                    fixed_config['training']['precision'] = check.auto_fix_value
        
        return fixed_config
    
    def _validate_embed_dim_divisibility(self):
        """Validate embed_dim is divisible by num_heads."""
        model = self.config.get('model', {})
        embed_dim = model.get('embed_dim', 768)
        num_heads = model.get('num_heads', 12)
        
        if embed_dim % num_heads != 0:
            # Find closest valid embed_dim
            closest_valid = self._find_closest_valid_embed_dim(embed_dim, num_heads)
            
            self.checks.append(ValidationCheck(
                name="Embed Dimension Divisibility",
                passed=False,
                severity="critical",
                message=f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})",
                suggestion=f"Use embed_dim={closest_valid} instead",
                auto_fix_available=True,
                auto_fix_value=closest_valid
            ))
        else:
            head_dim = embed_dim // num_heads
            
            if head_dim not in [64, 128]:
                self.checks.append(ValidationCheck(
                    name="Head Dimension",
                    passed=True,
                    severity="warning",
                    message=f"head_dim={head_dim} is unusual (typically 64 or 128)",
                    suggestion="Consider using 64 or 128 for optimal performance"
                ))
            else:
                self.checks.append(ValidationCheck(
                    name="Embed Dimension Divisibility",
                    passed=True,
                    severity="info",
                    message=f"embed_dim={embed_dim} divisible by num_heads={num_heads} ✓"
                ))
    
    def _validate_memory_requirements(self):
        """Validate memory requirements fit on hardware."""
        model = self.config.get('model', {})
        training = self.config.get('training', {})
        hardware = self.config.get('hardware', {})
        
        # Calculate parameters
        params = self._estimate_parameters(model)
        
        # Estimate memory
        memory_per_gpu = self._estimate_memory_per_gpu(
            params=params,
            batch_size=training.get('batch_size', 16),
            seq_len=model.get('max_seq_len', 1024),
            hidden=model.get('embed_dim', 768),
            layers=model.get('num_layers', 12),
            precision=training.get('precision', 'bf16'),
            num_gpus=hardware.get('num_gpus', 1),
            zero_stage=training.get('zero_stage', 2)
        )
        
        vram_per_gpu = hardware.get('vram_per_gpu', 16)
        
        if memory_per_gpu > vram_per_gpu * 0.9:  # 90% threshold
            # Suggest smaller batch size
            suggested_batch = max(1, training.get('batch_size', 16) // 2)
            
            self.checks.append(ValidationCheck(
                name="Memory Requirements",
                passed=False,
                severity="critical",
                message=f"Estimated {memory_per_gpu:.1f}GB per GPU exceeds available {vram_per_gpu}GB",
                suggestion=f"Reduce batch_size to {suggested_batch} or use gradient accumulation",
                auto_fix_available=True,
                auto_fix_value=suggested_batch
            ))
        elif memory_per_gpu > vram_per_gpu * 0.75:  # 75% warning
            self.checks.append(ValidationCheck(
                name="Memory Requirements",
                passed=True,
                severity="warning",
                message=f"Using {memory_per_gpu:.1f}GB of {vram_per_gpu}GB VRAM (tight fit)",
                suggestion="Consider monitoring memory usage during training"
            ))
        else:
            self.checks.append(ValidationCheck(
                name="Memory Requirements",
                passed=True,
                severity="info",
                message=f"Estimated {memory_per_gpu:.1f}GB per GPU fits comfortably ✓"
            ))
    
    def _validate_context_window_compatibility(self):
        """Validate context window matches position encoding."""
        model = self.config.get('model', {})
        context_window = model.get('max_seq_len', 1024)
        position_encoding = model.get('position_encoding', 'rope')
        
        if position_encoding == 'rope' and context_window > 2048:
            self.checks.append(ValidationCheck(
                name="Context Window Compatibility",
                passed=True,
                severity="warning",
                message=f"RoPE with {context_window} context may be suboptimal",
                suggestion="Consider Sliding Window Attention or ALiBi for longer contexts"
            ))
        elif position_encoding == 'sliding' and context_window < 2048:
            self.checks.append(ValidationCheck(
                name="Context Window Compatibility",
                passed=True,
                severity="info",
                message="Sliding Window is overkill for short context",
                suggestion="RoPE would be faster for contexts ≤2048"
            ))
        else:
            self.checks.append(ValidationCheck(
                name="Context Window Compatibility",
                passed=True,
                severity="info",
                message=f"Position encoding '{position_encoding}' suitable for context={context_window} ✓"
            ))
    
    def _validate_tokenizer_language_match(self):
        """Validate tokenizer type matches number of languages."""
        tokenizer = self.config.get('tokenizer', {})
        dataset = self.config.get('dataset', {})
        
        tokenizer_type = tokenizer.get('type', 'unigram')
        analysis = dataset.get('analysis', {})
        languages = analysis.get('languages', [])
        num_languages = len(languages)
        
        if tokenizer_type == 'bpe' and num_languages > 2:
            self.checks.append(ValidationCheck(
                name="Tokenizer-Language Match",
                passed=True,
                severity="warning",
                message=f"BPE with {num_languages} languages may be suboptimal",
                suggestion="Unigram SentencePiece is better for multilingual (3+ languages)"
            ))
        else:
            self.checks.append(ValidationCheck(
                name="Tokenizer-Language Match",
                passed=True,
                severity="info",
                message=f"Tokenizer '{tokenizer_type}' appropriate for {num_languages} language(s) ✓"
            ))
    
    def _validate_dataset_model_size(self):
        """Validate dataset has enough tokens for model size."""
        model = self.config.get('model', {})
        dataset = self.config.get('dataset', {})
        
        size_preset = model.get('size_preset', 'mini')
        analysis = dataset.get('analysis', {})
        dataset_tokens = analysis.get('total_tokens', 0)
        
        # Minimum tokens per model size
        MIN_TOKENS = {
            'tiny': 1e9,
            'mini': 5e9,
            'small': 20e9,
            'medium': 100e9
        }
        
        min_required = MIN_TOKENS.get(size_preset, 5e9)
        
        if dataset_tokens < min_required:
            self.checks.append(ValidationCheck(
                name="Dataset Size",
                passed=True,
                severity="warning",
                message=f"Dataset has {dataset_tokens/1e9:.1f}B tokens, {size_preset} model needs {min_required/1e9:.0f}B+",
                suggestion="Consider smaller model or more training data"
            ))
        else:
            self.checks.append(ValidationCheck(
                name="Dataset Size",
                passed=True,
                severity="info",
                message=f"Dataset size ({dataset_tokens/1e9:.1f}B tokens) sufficient for {size_preset} model ✓"
            ))
    
    def _validate_precision_hardware(self):
        """Validate precision is supported by hardware."""
        training = self.config.get('training', {})
        hardware = self.config.get('hardware', {})
        
        precision = training.get('precision', 'bf16')
        gpu_supports_bf16 = hardware.get('supports_bf16', True)
        
        if precision == 'bf16' and not gpu_supports_bf16:
            self.checks.append(ValidationCheck(
                name="Precision-Hardware Match",
                passed=False,
                severity="critical",
                message="BF16 not supported on this GPU",
                suggestion="Use FP16 or FP32 instead",
                auto_fix_available=True,
                auto_fix_value='fp16'
            ))
        elif precision == 'fp32':
            self.checks.append(ValidationCheck(
                name="Precision-Hardware Match",
                passed=True,
                severity="warning",
                message="FP32 is 2x slower than FP16/BF16",
                suggestion="Consider BF16 (if supported) or FP16 for faster training"
            ))
        else:
            self.checks.append(ValidationCheck(
                name="Precision-Hardware Match",
                passed=True,
                severity="info",
                message=f"Precision '{precision}' supported by hardware ✓"
            ))
    
    def _validate_batch_size(self):
        """Validate batch size is reasonable."""
        training = self.config.get('training', {})
        model = self.config.get('model', {})
        
        batch_size = training.get('batch_size', 16)
        grad_accum = training.get('gradient_accumulation_steps', 1)
        size_preset = model.get('size_preset', 'mini')
        
        # Recommended batch sizes
        RECOMMENDED_BATCH = {
            'tiny': 32,
            'mini': 16,
            'small': 8,
            'medium': 4
        }
        
        recommended = RECOMMENDED_BATCH.get(size_preset, 16)
        effective_batch = batch_size * grad_accum
        
        if batch_size < 1:
            self.checks.append(ValidationCheck(
                name="Batch Size",
                passed=False,
                severity="critical",
                message="Batch size must be at least 1",
                suggestion=f"Use batch_size={recommended}",
                auto_fix_available=True,
                auto_fix_value=recommended
            ))
        elif effective_batch < 32:
            self.checks.append(ValidationCheck(
                name="Batch Size",
                passed=True,
                severity="warning",
                message=f"Effective batch size ({effective_batch}) is small",
                suggestion="Consider increasing gradient_accumulation_steps for stability"
            ))
        else:
            self.checks.append(ValidationCheck(
                name="Batch Size",
                passed=True,
                severity="info",
                message=f"Batch configuration appropriate (effective={effective_batch}) ✓"
            ))
    
    def _estimate_parameters(self, model: Dict) -> int:
        """Estimate number of model parameters."""
        embed_dim = model.get('embed_dim', 768)
        num_layers = model.get('num_layers', 12)
        vocab_size = model.get('vocab_size', 50257)
        max_seq_len = model.get('max_seq_len', 1024)
        
        # Embedding
        embedding_params = vocab_size * embed_dim
        position_params = max_seq_len * embed_dim
        
        # Transformer layers
        # Attention: 4 * embed_dim^2 (Q, K, V, O projections)
        # MLP: 8 * embed_dim^2 (typical 4x expansion)
        layer_params = (4 + 8) * embed_dim * embed_dim
        transformer_params = num_layers * layer_params
        
        # LM head
        lm_head_params = vocab_size * embed_dim
        
        total_params = embedding_params + position_params + transformer_params + lm_head_params
        
        return int(total_params)
    
    def _estimate_memory_per_gpu(
        self,
        params: int,
        batch_size: int,
        seq_len: int,
        hidden: int,
        layers: int,
        precision: str,
        num_gpus: int,
        zero_stage: int
    ) -> float:
        """
        Estimate memory usage per GPU.
        
        Returns:
            Memory in GB
        """
        bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}.get(precision, 2)
        
        # Model parameters
        model_memory = params * bytes_per_param / 1e9
        
        # Optimizer states (Adam: 2x params in FP32)
        optimizer_memory = params * 2 * 4 / 1e9
        
        # Gradients
        gradient_memory = params * bytes_per_param / 1e9
        
        # Activations
        activation_memory = (
            batch_size * seq_len * hidden * layers * 4 * bytes_per_param / 1e9
        )
        
        # Apply ZeRO optimizations
        if zero_stage == 2:
            # ZeRO-2: Shard optimizer states
            optimizer_per_gpu = optimizer_memory / num_gpus
            total_per_gpu = model_memory + optimizer_per_gpu + gradient_memory + activation_memory
        elif zero_stage == 3:
            # ZeRO-3: Shard model, optimizer, gradients
            model_per_gpu = model_memory / num_gpus
            optimizer_per_gpu = optimizer_memory / num_gpus
            gradient_per_gpu = gradient_memory / num_gpus
            total_per_gpu = model_per_gpu + optimizer_per_gpu + gradient_per_gpu + activation_memory
        else:
            # No ZeRO
            total_per_gpu = model_memory + optimizer_memory + gradient_memory + activation_memory
        
        # Add 20% overhead for misc
        total_per_gpu *= 1.2
        
        return total_per_gpu
    
    def _find_closest_valid_embed_dim(self, embed_dim: int, num_heads: int) -> int:
        """Find closest embed_dim that's divisible by num_heads."""
        # Try nearby values
        for delta in range(0, 128):
            for sign in [1, -1]:
                candidate = embed_dim + (sign * delta)
                if candidate > 0 and candidate % num_heads == 0:
                    head_dim = candidate // num_heads
                    if head_dim in [64, 128]:  # Prefer standard head dims
                        return candidate
        
        # Fallback: just make it divisible
        return ((embed_dim + num_heads - 1) // num_heads) * num_heads
    
    def _print_results(self):
        """Print validation results."""
        # Categorize checks
        critical = [c for c in self.checks if not c.passed and c.severity == 'critical']
        warnings = [c for c in self.checks if not c.passed and c.severity == 'warning']
        info = [c for c in self.checks if c.passed and c.severity == 'info']
        
        # Print critical failures
        if critical:
            print("❌ CRITICAL FAILURES:\n")
            for check in critical:
                self._print_check(check)
        
        # Print warnings
        if warnings:
            print("\n⚠️  WARNINGS:\n")
            for check in warnings:
                self._print_check(check)
        
        # Print successes (abbreviated)
        if info and not critical and not warnings:
            print("✅ ALL CHECKS PASSED\n")
            for check in info:
                print(f"  ✓ {check.name}")
        
        # Summary
        print("\n" + "─"*70)
        if critical:
            print(f"❌ Validation FAILED: {len(critical)} critical error(s)")
            print("   Fix critical errors before training")
        elif warnings:
            print(f"⚠️  Validation passed with {len(warnings)} warning(s)")
            print("   Training can proceed but consider addressing warnings")
        else:
            print("✅ Validation PASSED: Configuration is ready for training!")
        print("─"*70 + "\n")
    
    def _print_check(self, check: ValidationCheck):
        """Print a single check result."""
        print(f"  {check.name}:")
        print(f"    {check.message}")
        if check.suggestion:
            print(f"    💡 {check.suggestion}")
        if check.auto_fix_available:
            print(f"    🔧 Auto-fix available: {check.auto_fix_value}")
        print()
