# Quick Start Guide

Get up and running with Bilingual GPT-2 Pro in under 10 minutes!

---

## Prerequisites

- **Python 3.10+**
- **CUDA 12.1+** (for GPU training)
- **16GB+ VRAM** recommended
- **Ubuntu 22.04+** or similar Linux distribution

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/bilingual-gpt2-pro.git
cd bilingual-gpt2-pro
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quick Training (3 Methods)

### Method 1: Interactive Wizard (Recommended for Beginners)

The wizard guides you through 10 easy steps:

```bash
python cli/setup_wizard.py
```

**What it does:**
1. Detects your hardware automatically
2. Analyzes your dataset
3. Recommends optimal settings
4. Validates configuration
5. Saves ready-to-use config

**Time:** ~5-10 minutes

---

### Method 2: Use Preset Configuration

If you already have data ready:

```bash
# 1. Copy and edit preset
cp configs/presets/mini_bilingual.yaml configs/my_config.yaml
nano configs/my_config.yaml  # Edit dataset path

# 2. Validate configuration
python cli/validate_config.py configs/my_config.yaml

# 3. Start training
python train.py --config configs/my_config.yaml
```

**Time:** ~2 minutes

---

### Method 3: Analyze Dataset First

If you want to understand your dataset before training:

```bash
# 1. Analyze dataset
python cli/analyze_dataset.py /path/to/data

# This will show:
# - Dataset type (text/QA/code)
# - Language distribution
# - Quality metrics
# - Vocabulary size recommendations
# - Estimated training time

# 2. Run wizard with insights
python cli/setup_wizard.py
```

**Time:** Analysis time + wizard (~10-15 minutes total)

---

## Understanding the Output

### Dataset Analysis Report

```
╔══════════════════════════════════════════════════════════════════╗
║                    DATASET ANALYSIS REPORT                       ║
╚══════════════════════════════════════════════════════════════════╝

📁 Dataset: /mnt/data/corpus
📊 Size: 15.2 GB
📄 Files: 245 files

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. DATASET TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Type: TEXT_CORPUS ✅
✅ COMPATIBLE with pre-training

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. TOKEN ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total tokens: 3.2B
Unique words: 2.8M
Type-Token Ratio: 0.0009 (excellent diversity)

🎯 RECOMMENDED VOCAB SIZE: 45,000
   Unigram: 45,000 ✅ (13% more efficient than BPE)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. LANGUAGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🇬🇧 English: 58.3%
🇪🇸 Spanish: 41.2%
🇫🇷 French: 0.5% (contamination warning)
```

**Key Metrics:**
- **Token count** - NOT file size!
- **Type-Token Ratio** - Lower = better diversity
- **Language detection** - With contamination warnings
- **Quality score** - 0-100 scale

---

## Configuration Options

### Model Sizes

| Size | Parameters | VRAM | Training Speed | Quality |
|------|-----------|------|----------------|---------|
| Tiny | 40M | 2GB | Very Fast | Basic |
| Mini | 124M | 8GB | Fast | Good |
| Small | 350M | 14GB | Moderate | Great |
| Medium | 760M | 28GB | Slow | Excellent |

### Context Windows

| Size | Best For | Speed | Position Encoding |
|------|----------|-------|-------------------|
| 512 | Short text | Very Fast | RoPE, ALiBi |
| 1024 | Standard tasks | Fast | RoPE, ALiBi |
| 2048 | Long text | Moderate | RoPE, ALiBi, Sliding |
| 4096 | Documents | Slow | ALiBi, Sliding |

### Precision

| Type | Speed | Memory | Stability | Requirements |
|------|-------|--------|-----------|--------------|
| FP32 | 1x | 2x | Perfect | Any GPU |
| FP16 | 2x | 1x | Good | Any modern GPU |
| BF16 | 2x | 1x | Excellent | RTX 30/40 series |

---

## Training Workflow

### Complete Pipeline

```bash
# 1. Setup (one time)
python cli/setup_wizard.py
# Output: configs/my_config.yaml

# 2. Train model
python train.py --config configs/my_config.yaml

# 3. Monitor training
tensorboard --logdir outputs/

# 4. Evaluate (coming soon)
python evaluate.py --checkpoint outputs/model/checkpoint-50000

# 5. Export (coming soon)
python scripts/export_model.py --checkpoint outputs/model/final
```

---

## Common Customizations

### Adjust Batch Size for Your GPU

```bash
# Smaller GPU (8GB)
python train.py --config config.yaml --batch-size 8

# Larger GPU (24GB+)
python train.py --config config.yaml --batch-size 32
```

### Use Different Precision

```bash
# Force FP16 (older GPUs)
python train.py --config config.yaml --precision fp16

# Force FP32 (debugging)
python train.py --config config.yaml --precision fp32
```

### Resume Training

```bash
python train.py \
  --config config.yaml \
  --resume outputs/model/checkpoint-50000
```

---

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/

# Open browser to: http://localhost:6006
```

**What to watch:**
- Training loss (should decrease)
- Learning rate schedule
- Gradient norms
- GPU utilization

### Command Line

Training script shows:
- Steps per second
- Estimated time remaining
- Current loss
- GPU memory usage

---

## Next Steps

1. **Read the docs:** Check `docs/` for detailed guides
2. **Try examples:** Look in `examples/` for code samples
3. **Experiment:** Try different model sizes and contexts
4. **Fine-tune:** Train on your own data

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory

**Solutions:**
1. Reduce batch size: `--batch-size 4`
2. Increase gradient accumulation in config
3. Use smaller model: `tiny` or `mini`
4. Enable gradient checkpointing

### Training is Slow

**Optimizations:**
1. Use BF16 precision (if supported)
2. Enable Flash Attention
3. Check GPU utilization with `nvtop`
4. Ensure CUDA is properly installed

### Dataset Analysis Takes Forever

**Solutions:**
1. Use sampling: `--sampling 15` (15% sample)
2. Results are cached - second run is instant
3. For huge datasets (>100GB), max sampling is 50%

---

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/yourusername/bilingual-gpt2-pro/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/bilingual-gpt2-pro/discussions)
- **Documentation:** Check `docs/` directory

---

## Example Session

```bash
# Full example from start to finish
cd bilingual-gpt2-pro

# 1. Activate environment
source venv/bin/activate

# 2. Analyze your dataset
python cli/analyze_dataset.py ~/data/bilingual_corpus
# Takes 5-10 minutes, shows comprehensive report

# 3. Run wizard
python cli/setup_wizard.py
# Answer 10 questions, get optimal config

# 4. Start training
python train.py --config configs/my_config.yaml
# Training begins!

# 5. Monitor (in another terminal)
tensorboard --logdir outputs/
```

---

**Ready to train!** 🚀
