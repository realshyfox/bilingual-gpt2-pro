# Configuration Reference

Complete reference for all configuration options in bilingual GPT-2 training.

---

## 📋 Configuration File Structure

Configuration files use YAML format:

```yaml
dataset:
  # Dataset settings

task:
  # Task type

tokenizer:
  # Tokenizer settings

model:
  # Model architecture

training:
  # Training parameters

hardware:
  # Hardware configuration

output:
  # Output settings
```

---

## 📊 Dataset Configuration

```yaml
dataset:
  train_path: /path/to/training/data      # Required
  val_path: /path/to/validation/data      # Optional
  
  analysis:                               # Populated by analyzer
    total_tokens: 2800000000
    dataset_type: text_corpus
    languages: [english, spanish]
```

### Parameters:

**train_path** (required)
- Path to training data directory or file
- Supports: local paths, HuggingFace datasets, URLs
- Example: `/data/corpus` or `hf://dataset_name`

**val_path** (optional)
- Path to validation data
- If not provided, no validation during training
- Recommended: 5-10% of training data size

**analysis** (auto-populated)
- Filled by `SmartDatasetAnalyzer`
- Contains metadata about dataset
- Don't edit manually

---

## 🎯 Task Configuration

```yaml
task:
  type: pre-training  # or "fine-tuning"
```

### Options:

**pre-training**
- Training from scratch on text corpus
- Learns general language patterns
- Requires large dataset (1GB+)

**fine-tuning**
- Adapting pretrained model to specific task
- Requires task-specific dataset
- Can use smaller dataset (1,000+ examples)

---

## 🔤 Tokenizer Configuration

```yaml
tokenizer:
  type: unigram                # bpe, unigram, or wordpiece
  vocab_size: 45000            # Target vocabulary size
  character_coverage: 0.9995   # For SentencePiece
  
  special_tokens:
    - "<pad>"
    - "<unk>"
    - "<s>"
    - "</s>"
```

### Parameters:

**type**
- Options: `bpe`, `unigram`, `wordpiece`
- Recommended: `unigram` for multilingual
- Default: `unigram`

**vocab_size**
- Target vocabulary size
- Range: 8,000 - 100,000
- Default: 50,257 (GPT-2 standard)
- Recommended: Use analyzer recommendation

**character_coverage**
- Percentage of characters to cover
- Range: 0.95 - 1.0
- Default: 0.9995
- Lower for languages with many rare characters

**special_tokens**
- Special tokens to add
- Standard: `<pad>`, `<unk>`, `<s>`, `</s>`
- Can add custom tokens

---

## 🧠 Model Configuration

```yaml
model:
  type: gpt2
  size_preset: mini              # tiny, mini, small, medium
  
  # Architecture (from preset)
  embed_dim: 768
  num_layers: 12
  num_heads: 12
  max_seq_len: 1024
  vocab_size: 45000
  
  # Regularization
  dropout: 0.1
  attention_dropout: 0.1
  
  # Advanced features
  use_flash_attention: true
  position_encoding: rope        # rope, alibi, sliding, learned
  activation: gelu               # gelu, swiglu, relu
  normalization: layernorm       # layernorm, rmsnorm
  
  # Optional
  use_gqa: false                 # Grouped Query Attention
  gqa_num_groups: null
```

### Model Size Presets:

**tiny** (40M parameters)
- embed_dim: 512
- num_layers: 6
- num_heads: 8
- Min VRAM: 2GB
- Best for: Testing, edge deployment

**mini** (124M parameters) - **DEFAULT**
- embed_dim: 768
- num_layers: 12
- num_heads: 12
- Min VRAM: 8GB
- Best for: General purpose

**small** (350M parameters)
- embed_dim: 1024
- num_layers: 24
- num_heads: 16
- Min VRAM: 14GB
- Best for: High quality

**medium** (760M parameters)
- embed_dim: 1280
- num_layers: 36
- num_heads: 20
- Min VRAM: 28GB
- Best for: Maximum quality

### Architecture Parameters:

**embed_dim**
- Embedding dimension
- Must be divisible by num_heads
- Typical: 512, 768, 1024, 1280

**num_layers**
- Number of transformer layers
- Range: 6-48
- More = better quality but slower

**num_heads**
- Number of attention heads
- Must divide embed_dim evenly
- Head dimension (embed_dim/num_heads) should be 64 or 128

**max_seq_len**
- Maximum sequence length
- Options: 512, 1024, 2048, 4096
- Longer = more context but more memory

### Advanced Options:

**use_flash_attention**
- Enable Flash Attention 2
- Requires: flash-attn package
- Speedup: 2-4x faster
- Recommended: true (if available)

**position_encoding**
- `learned`: Learned embeddings (simple, max 2048)
- `rope`: Rotary Position Embedding (good for ≤2048)
- `alibi`: Attention with Linear Biases (best for long context)
- `sliding`: Sliding window attention (very long context)

**activation**
- `gelu`: Standard GPT-2 activation
- `swiglu`: Modern activation (better quality, slower)
- `relu`: Simple activation (fast, lower quality)

**normalization**
- `layernorm`: Standard (default)
- `rmsnorm`: Faster, fewer parameters

---

## 🎓 Training Configuration

```yaml
training:
  # Duration
  max_steps: 500000
  max_epochs: null              # Alternative to max_steps
  
  # Batch settings
  batch_size: 16
  gradient_accumulation_steps: 4
  
  # Optimization
  optimizer: adamw              # adamw, adam, sgd
  learning_rate: 3.0e-4
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1.0e-8
  
  # Learning rate schedule
  scheduler: cosine             # cosine, linear, constant
  warmup_steps: 2000
  min_lr_ratio: 0.1
  
  # Precision and stability
  precision: bf16               # fp32, fp16, bf16
  gradient_clipping: 1.0
  gradient_checkpointing: false
  
  # Distributed training
  zero_stage: 2                 # DeepSpeed ZeRO: 0, 1, 2, or 3
  
  # Logging and checkpointing
  log_interval: 100
  eval_interval: 5000
  save_interval: 10000
  
  # Validation
  eval_steps: 500
  
  # Data loading
  num_workers: 4
  prefetch_factor: 2
```

### Key Parameters:

**max_steps**
- Total training steps
- Typical: 100K - 1M
- Recommendation from analyzer

**batch_size**
- Samples per GPU per step
- Adjust based on VRAM
- Typical: 4-32

**gradient_accumulation_steps**
- Accumulate gradients over N steps
- Effective batch = batch_size × grad_accum × num_gpus
- Use to simulate larger batches

**learning_rate**
- Learning rate for optimizer
- Pre-training: 3e-4
- Fine-tuning: 1e-4 to 5e-5
- Adjust based on batch size

**scheduler**
- `cosine`: Cosine annealing (recommended)
- `linear`: Linear decay
- `constant`: Constant LR after warmup

**precision**
- `fp32`: Full precision (slow, stable)
- `fp16`: Half precision (fast, needs scaling)
- `bf16`: Brain float 16 (fast, stable, **recommended**)

**zero_stage**
- `0`: No ZeRO (single GPU)
- `2`: Shard optimizer states (recommended)
- `3`: Shard model + optimizer (max memory savings)

---

## 💻 Hardware Configuration

```yaml
hardware:
  num_gpus: 2
  vram_per_gpu: 16
  supports_bf16: true
  supports_fp16: true
```

### Auto-Detected:
These are automatically filled by `HardwareDetector`.

Manual override only if needed:
- Testing configurations
- Planning for different hardware
- Simulating resource constraints

---

## 📁 Output Configuration

```yaml
output:
  dir: outputs/my_model
  save_total_limit: 3           # Keep only last N checkpoints
  
  logging:
    use_wandb: false
    use_tensorboard: true
    project_name: bilingual-gpt2
    run_name: null              # Auto-generated if null
    log_dir: logs
```

### Parameters:

**dir**
- Output directory for model and logs
- Structure created automatically

**save_total_limit**
- Maximum checkpoints to keep
- Older checkpoints deleted automatically
- 0 = keep all

**logging**
- `use_wandb`: Log to Weights & Biases
- `use_tensorboard`: Log to TensorBoard (recommended)
- `project_name`: Project identifier
- `run_name`: Run identifier (auto-generated if null)

---

## 🔄 Complete Example Configurations

### Example 1: Small English Model

```yaml
dataset:
  train_path: /data/english_corpus
  val_path: /data/english_val

task:
  type: pre-training

tokenizer:
  type: bpe
  vocab_size: 50257

model:
  size_preset: mini
  max_seq_len: 1024
  use_flash_attention: true
  position_encoding: rope

training:
  max_steps: 300000
  batch_size: 16
  learning_rate: 3.0e-4
  precision: bf16

output:
  dir: outputs/english_mini
```

### Example 2: Bilingual Model

```yaml
dataset:
  train_path: /data/en_es_corpus

task:
  type: pre-training

tokenizer:
  type: unigram
  vocab_size: 45000

model:
  size_preset: small
  max_seq_len: 2048
  use_flash_attention: true

training:
  max_steps: 500000
  batch_size: 8
  gradient_accumulation_steps: 8
  precision: bf16
  zero_stage: 2

hardware:
  num_gpus: 2

output:
  dir: outputs/bilingual_small
```

### Example 3: Fine-tuning for Q&A

```yaml
dataset:
  train_path: /data/qa_train.jsonl
  val_path: /data/qa_val.jsonl

task:
  type: fine-tuning

model:
  size_preset: mini
  max_seq_len: 512

training:
  max_steps: 10000
  batch_size: 8
  learning_rate: 5.0e-5         # Lower for fine-tuning
  warmup_steps: 500
  eval_interval: 500

output:
  dir: outputs/qa_model
```

---

## 🛠️ CLI Overrides

Override any parameter from command line:

```bash
python train.py \
  --config config.yaml \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --max-steps 100000 \
  --precision fp16 \
  --output-dir outputs/custom
```

Supported overrides:
- `--batch-size`
- `--learning-rate`
- `--max-steps`
- `--precision`
- `--output-dir`

---

## ✅ Validation

Before training, validate your config:

```bash
python cli/validate_config.py config.yaml
```

This checks:
- Parameter compatibility
- Memory requirements
- Hardware support
- Dataset compatibility

Auto-fix available:
```bash
python cli/validate_config.py config.yaml --fix
```

---

## 💡 Best Practices

### 1. Start with Presets
Use `cli/setup_wizard.py` to generate optimal configs

### 2. Incremental Changes
Change one parameter at a time when experimenting

### 3. Validate Always
Run validator before training

### 4. Monitor Training
Use TensorBoard to track progress

### 5. Save Configs
Keep configs in version control with model checkpoints

### 6. Document Changes
Comment your config files:
```yaml
# Increased batch size for faster training
batch_size: 32  # was: 16
```

---

## 📚 Additional Resources

- **Quick Start:** `docs/quickstart.md`
- **Dataset Guide:** `docs/dataset_guide.md`
- **Troubleshooting:** `docs/troubleshooting.md`
- **Model Registry:** `core/model_registry.py` (all presets)

---

**Need help?** Run the wizard: `python cli/setup_wizard.py`
