"""
Model Registry
Defines all model size presets, context options, and precision configurations.
"""

# Model size presets optimized for 2x RTX 4070 Ti Super
MODEL_PRESETS = {
    "tiny": {
        "embed_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "head_dim": 64,
        "params": "40M",
        "params_int": 40_000_000,
        "min_vram": 2,
        "recommended_batch": 32,
        "context_window": 1024,
        "best_for": "Fast prototyping, testing, edge devices",
        "training_speed": "very fast",
    },
    "mini": {  # DEFAULT
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "params": "124M",
        "params_int": 124_000_000,
        "min_vram": 8,
        "recommended_batch": 16,
        "context_window": 1024,
        "best_for": "General purpose, good quality/speed balance",
        "training_speed": "fast",
    },
    "small": {
        "embed_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "head_dim": 64,
        "params": "350M",
        "params_int": 350_000_000,
        "min_vram": 14,
        "recommended_batch": 8,
        "context_window": 1024,
        "best_for": "High quality, production use",
        "training_speed": "moderate",
    },
    "medium": {
        "embed_dim": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "head_dim": 64,
        "params": "760M",
        "params_int": 760_000_000,
        "min_vram": 28,
        "recommended_batch": 4,
        "context_window": 1024,
        "best_for": "Maximum quality, research",
        "training_speed": "slow",
        "requires": "Multiple GPUs or gradient accumulation",
    },
}

# Context window options with position encoding recommendations
CONTEXT_OPTIONS = {
    512: {
        "best_for": "Classification, short text, snippets",
        "position": ["rope", "alibi", "learned"],
        "memory_efficient": True,
        "training_speed": "very fast",
        "typical_use_cases": [
            "Sentiment analysis",
            "Text classification",
            "Short answer generation",
        ],
    },
    1024: {  # DEFAULT
        "best_for": "Standard tasks, paragraphs, articles",
        "position": ["rope", "alibi"],
        "memory_efficient": True,
        "training_speed": "fast",
        "typical_use_cases": [
            "General text generation",
            "Question answering",
            "Summarization",
            "Most NLP tasks",
        ],
    },
    2048: {
        "best_for": "Long text, full articles, stories",
        "position": ["rope", "alibi", "sliding"],
        "memory_efficient": "moderate",
        "training_speed": "moderate",
        "typical_use_cases": [
            "Long-form content generation",
            "Document analysis",
            "Story writing",
        ],
    },
    4096: {
        "best_for": "Documents, books, long context",
        "position": ["alibi", "sliding"],
        "memory_efficient": False,
        "training_speed": "slow",
        "requires": "GQA or Sliding Window Attention",
        "typical_use_cases": [
            "Book analysis",
            "Long document understanding",
            "Research paper processing",
        ],
    },
}

# Precision options with trade-offs
PRECISION_CONFIG = {
    "fp32": {
        "name": "Float32",
        "speed_multiplier": 1.0,
        "memory_multiplier": 2.0,
        "stability": "perfect",
        "numeric_precision": "highest",
        "use_when": [
            "Debugging numerical issues",
            "Detecting NaN/Inf problems",
            "Final verification runs",
        ],
        "pros": [
            "Perfect numerical stability",
            "No gradient scaling needed",
            "Easiest debugging",
        ],
        "cons": [
            "2x slower than FP16/BF16",
            "2x more memory",
            "Rarely necessary",
        ],
    },
    "fp16": {
        "name": "Float16",
        "speed_multiplier": 2.0,
        "memory_multiplier": 1.0,
        "stability": "good_with_loss_scaling",
        "numeric_precision": "good",
        "use_when": [
            "Older GPUs (no BF16 support)",
            "GTX 10 series, RTX 20 series",
            "When BF16 unavailable",
        ],
        "pros": [
            "2x faster than FP32",
            "Half the memory",
            "Wide hardware support",
        ],
        "cons": [
            "Requires loss scaling",
            "Can overflow/underflow",
            "Less stable than BF16",
        ],
        "requires": "gradient_scaling",
    },
    "bf16": {  # RECOMMENDED for RTX 40 series
        "name": "BFloat16",
        "speed_multiplier": 2.0,
        "memory_multiplier": 1.0,
        "stability": "excellent",
        "numeric_precision": "good",
        "use_when": [
            "RTX 30 series (Ampere)",
            "RTX 40 series (Ada) ⭐",
            "A100, H100 (data center)",
        ],
        "pros": [
            "2x faster than FP32",
            "Half the memory",
            "No loss scaling needed",
            "Same range as FP32",
            "Best stability/speed trade-off",
        ],
        "cons": [
            "Requires newer GPUs",
            "Slightly less precise than FP16",
        ],
        "hardware_requirements": {
            "min_compute_capability": 8.0,
            "supported_gpus": [
                "RTX 3090, RTX 3080 Ti",
                "RTX 4090, RTX 4080, RTX 4070 Ti",
                "A100, H100",
            ],
        },
    },
}

# Position encoding options
POSITION_ENCODING_OPTIONS = {
    "learned": {
        "name": "Learned Positional Embedding",
        "best_for": "Short contexts (≤512)",
        "max_context": 512,
        "pros": [
            "Simple implementation",
            "Fast",
            "Works well for fixed lengths",
        ],
        "cons": [
            "Cannot extrapolate beyond training length",
            "Uses extra parameters",
        ],
    },
    "rope": {  # Rotary Position Embedding - DEFAULT
        "name": "RoPE (Rotary Position Embedding)",
        "best_for": "Standard contexts (512-2048)",
        "max_context": 2048,
        "pros": [
            "No extra parameters",
            "Relative position awareness",
            "Good extrapolation",
            "Fast",
        ],
        "cons": [
            "Performance degrades >2048",
            "Not ideal for very long contexts",
        ],
        "used_in": ["LLaMA", "GPT-NeoX", "PaLM"],
    },
    "alibi": {
        "name": "ALiBi (Attention with Linear Biases)",
        "best_for": "Variable/long contexts (up to 8192+)",
        "max_context": 8192,
        "pros": [
            "No extra parameters",
            "Excellent extrapolation",
            "Works for any length",
            "Memory efficient",
        ],
        "cons": [
            "Slightly slower than RoPE",
            "Less common (fewer implementations)",
        ],
        "used_in": ["BLOOM", "MPT"],
    },
    "sliding": {
        "name": "Sliding Window Attention",
        "best_for": "Very long contexts (4096+)",
        "max_context": 32768,
        "pros": [
            "Handles very long contexts",
            "Linear memory scaling",
            "Good for documents",
        ],
        "cons": [
            "More complex implementation",
            "Slower than standard attention",
            "Requires careful tuning",
        ],
        "requires": "window_size parameter",
        "used_in": ["Longformer", "BigBird"],
    },
}

# Activation functions
ACTIVATION_OPTIONS = {
    "gelu": {
        "name": "GELU",
        "default": True,
        "speed": "fast",
        "quality": "good",
        "used_in": ["GPT-2", "BERT", "T5"],
    },
    "swiglu": {
        "name": "SwiGLU",
        "default": False,
        "speed": "moderate",
        "quality": "excellent",
        "requires": "Wider MLP",
        "used_in": ["LLaMA", "PaLM"],
        "note": "Better quality but slower",
    },
    "relu": {
        "name": "ReLU",
        "default": False,
        "speed": "very fast",
        "quality": "acceptable",
        "used_in": ["Early transformers"],
        "note": "Simple but less effective",
    },
}

# Normalization options
NORMALIZATION_OPTIONS = {
    "layernorm": {
        "name": "Layer Normalization",
        "default": True,
        "speed": "fast",
        "stability": "excellent",
        "used_in": ["GPT-2", "BERT", "Most models"],
    },
    "rmsnorm": {
        "name": "RMS Normalization",
        "default": False,
        "speed": "faster",
        "stability": "excellent",
        "params_saved": "~2M for 124M model",
        "used_in": ["LLaMA", "T5"],
        "note": "Slightly faster, fewer parameters",
    },
}

# Optimizer options
OPTIMIZER_OPTIONS = {
    "adamw": {
        "name": "AdamW",
        "default": True,
        "lr_range": [1e-5, 5e-4],
        "recommended_lr": 3e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "pros": [
            "Most widely used",
            "Reliable convergence",
            "Good default choice",
        ],
    },
    "adam": {
        "name": "Adam",
        "default": False,
        "lr_range": [1e-5, 5e-4],
        "recommended_lr": 3e-4,
        "betas": (0.9, 0.999),
        "note": "Similar to AdamW but without decoupled weight decay",
    },
    "sgd": {
        "name": "SGD with Momentum",
        "default": False,
        "lr_range": [1e-3, 1e-1],
        "recommended_lr": 1e-2,
        "momentum": 0.9,
        "note": "Requires careful tuning, not recommended for transformers",
    },
}

# Learning rate scheduler options
SCHEDULER_OPTIONS = {
    "cosine": {
        "name": "Cosine Annealing",
        "default": True,
        "warmup_steps": 2000,
        "min_lr_ratio": 0.1,
        "pros": [
            "Smooth decay",
            "Good final performance",
            "Standard choice",
        ],
    },
    "linear": {
        "name": "Linear Decay",
        "default": False,
        "warmup_steps": 2000,
        "note": "Simple, predictable decay",
    },
    "constant": {
        "name": "Constant (with warmup)",
        "default": False,
        "warmup_steps": 2000,
        "note": "Use for continued pre-training",
    },
}


def get_model_preset(name: str) -> dict:
    """Get model preset configuration by name."""
    if name not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model preset: {name}. "
            f"Available: {', '.join(MODEL_PRESETS.keys())}"
        )
    return MODEL_PRESETS[name].copy()


def get_context_options(size: int) -> dict:
    """Get context window configuration."""
    if size not in CONTEXT_OPTIONS:
        raise ValueError(
            f"Unsupported context size: {size}. "
            f"Available: {', '.join(map(str, CONTEXT_OPTIONS.keys()))}"
        )
    return CONTEXT_OPTIONS[size].copy()


def get_precision_config(precision: str) -> dict:
    """Get precision configuration."""
    if precision not in PRECISION_CONFIG:
        raise ValueError(
            f"Unknown precision: {precision}. "
            f"Available: {', '.join(PRECISION_CONFIG.keys())}"
        )
    return PRECISION_CONFIG[precision].copy()


def list_available_presets() -> None:
    """Print all available model presets."""
    print("\n📊 Available Model Presets:\n")
    for name, config in MODEL_PRESETS.items():
        default = " (DEFAULT)" if name == "mini" else ""
        print(f"  {name.upper()}{default}")
        print(f"    • Parameters: {config['params']}")
        print(f"    • Min VRAM: {config['min_vram']}GB")
        print(f"    • Recommended batch: {config['recommended_batch']}")
        print(f"    • Best for: {config['best_for']}")
        print()
