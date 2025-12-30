# Troubleshooting Guide

Solutions to common issues when using the bilingual GPT-2 training system.

---

## 🔧 Installation Issues

### Problem: `pip install` fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement torch>=2.1.0
```

**Solutions:**

1. **Update pip:**
```bash
pip install --upgrade pip
```

2. **Install PyTorch separately:**
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. **Then install other requirements:**
```bash
pip install -r requirements.txt
```

---

### Problem: Flash Attention installation fails

**Error:**
```
ERROR: Failed building wheel for flash-attn
```

**Solution:**

Flash Attention is optional. Remove or comment out in requirements.txt:
```bash
# flash-attn>=2.5.0  # Optional, comment out if install fails
```

The system works without it (just slower). Or install manually:
```bash
pip install flash-attn --no-build-isolation
```

---

### Problem: Import errors after installation

**Error:**
```python
ModuleNotFoundError: No module named 'core'
```

**Solutions:**

1. **Install in editable mode:**
```bash
pip install -e .
```

2. **Or add to PYTHONPATH:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

3. **Or run from project root:**
```bash
cd /path/to/bilingual-gpt2-pro
python cli/setup_wizard.py
```

---

## 🎮 GPU Issues

### Problem: GPU not detected

**Error:**
```
CUDA not available, training will be VERY slow!
```

**Check:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

**Solutions:**

1. **Install NVIDIA drivers:**
```bash
# Ubuntu
sudo ubuntu-drivers autoinstall
sudo reboot
```

2. **Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Check CUDA version match:**
```bash
nvidia-smi  # Shows CUDA version
python -c "import torch; print(torch.version.cuda)"  # Should match
```

---

### Problem: Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce batch size:**
```yaml
training:
  batch_size: 8  # Reduce from 16
```

2. **Increase gradient accumulation:**
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 8  # Effective batch = 64
```

3. **Use smaller model:**
```yaml
model:
  size_preset: tiny  # Instead of mini or small
```

4. **Enable gradient checkpointing:**
```yaml
training:
  gradient_checkpointing: true  # Trades speed for memory
```

5. **Use DeepSpeed ZeRO-3:**
```yaml
training:
  zero_stage: 3  # Shard model across GPUs
```

6. **Clear cache between runs:**
```python
import torch
torch.cuda.empty_cache()
```

---

### Problem: Training is very slow

**Causes and Solutions:**

1. **Not using GPU:**
   - Check `nvidia-smi` shows GPU activity
   - Verify CUDA is available

2. **Not using BF16/FP16:**
```yaml
training:
  precision: bf16  # 2x faster than fp32
```

3. **Flash Attention not enabled:**
```yaml
model:
  use_flash_attention: true
```

4. **Too many data workers:**
```yaml
training:
  num_workers: 4  # Don't set too high
```

5. **Disk bottleneck:**
   - Use SSD instead of HDD
   - Reduce num_workers
   - Cache dataset in RAM

---

## 📊 Dataset Issues

### Problem: "Dataset path not found"

**Error:**
```
FileNotFoundError: Dataset path not found: /path/to/data
```

**Solutions:**

1. **Check path exists:**
```bash
ls -la /path/to/data
```

2. **Use absolute path:**
```yaml
dataset:
  train_path: /home/user/data/corpus  # Not ~/data/corpus
```

3. **Check permissions:**
```bash
chmod -R 755 /path/to/data
```

---

### Problem: "Encoding errors"

**Error:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte...
```

**Solutions:**

1. **Convert files to UTF-8:**
```bash
for file in *.txt; do
    iconv -f ISO-8859-1 -t UTF-8 "$file" > "utf8_$file"
done
```

2. **Use the analyzer to identify bad files:**
```bash
python cli/analyze_dataset.py /path/to/data
# Check "Encoding errors" in report
```

3. **Skip problematic files:**
The system automatically skips files with encoding errors during training.

---

### Problem: "Dataset too small"

**Warning:**
```
⚠️ Dataset too small for mini model
```

**Solutions:**

1. **Use smaller model:**
```yaml
model:
  size_preset: tiny  # Only needs 1B tokens
```

2. **Get more data:**
   - Download Wikipedia dumps
   - Use Common Crawl
   - Combine multiple datasets

3. **Use fine-tuning instead:**
   - Start from pretrained model
   - Fine-tune on your small dataset

---

### Problem: High duplicate rate

**Warning:**
```
⚠️ Duplicate rate: 25.3%
```

**Solution - Deduplicate:**
```python
import hashlib
from pathlib import Path

seen = set()
output_dir = Path('deduplicated/')
output_dir.mkdir(exist_ok=True)

for file in Path('corpus/').glob('*.txt'):
    with open(file, 'r') as f:
        text = f.read()
    
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    if text_hash not in seen:
        seen.add(text_hash)
        with open(output_dir / file.name, 'w') as f:
            f.write(text)
```

---

## ⚙️ Configuration Issues

### Problem: "embed_dim not divisible by num_heads"

**Error:**
```
❌ ERROR: embed_dim (800) must be divisible by num_heads (12)
```

**Solutions:**

1. **Use validator auto-fix:**
```bash
python cli/validate_config.py config.yaml --fix
```

2. **Manual fix:**
```yaml
model:
  embed_dim: 768  # Divisible by 12
  num_heads: 12
```

3. **Use preset:**
```yaml
model:
  size_preset: mini  # Guaranteed valid
```

---

### Problem: "Memory estimate exceeds VRAM"

**Error:**
```
❌ Estimated 18.5GB per GPU exceeds available 16GB
```

**Solutions:**

1. **Reduce batch size:**
```yaml
training:
  batch_size: 8  # Reduce from 16
```

2. **Use gradient accumulation:**
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 16
```

3. **Enable DeepSpeed ZeRO:**
```yaml
training:
  zero_stage: 2  # Or 3 for more savings
```

4. **Use smaller model:**
```yaml
model:
  size_preset: tiny  # Or mini instead of small
```

---

### Problem: "BF16 not supported"

**Error:**
```
❌ BF16 not supported on this GPU. Use FP16 or FP32
```

**Solution:**
```yaml
training:
  precision: fp16  # Instead of bf16
```

Or check GPU:
```python
import torch
print(torch.cuda.get_device_properties(0))
# Need compute capability >= 8.0 for BF16
```

---

## 🏋️ Training Issues

### Problem: Loss is NaN

**Symptoms:**
```
Step 100: loss=nan
```

**Solutions:**

1. **Reduce learning rate:**
```yaml
training:
  learning_rate: 1.0e-4  # Reduce from 3e-4
```

2. **Use FP32:**
```yaml
training:
  precision: fp32  # More stable than fp16
```

3. **Enable gradient clipping:**
```yaml
training:
  gradient_clipping: 1.0
```

4. **Check dataset:**
   - Remove corrupted files
   - Check for extreme values

---

### Problem: Loss not decreasing

**Symptoms:**
```
Step 1000: loss=4.5
Step 2000: loss=4.5
Step 3000: loss=4.5
```

**Causes and Solutions:**

1. **Learning rate too low:**
```yaml
training:
  learning_rate: 3.0e-4  # Increase from 1e-5
```

2. **Learning rate too high:**
```yaml
training:
  learning_rate: 1.0e-4  # Decrease from 1e-3
```

3. **Dataset issues:**
   - Check quality with analyzer
   - Verify labels are correct
   - Check for data leakage

4. **Model too small:**
```yaml
model:
  size_preset: small  # Increase from tiny
```

---

### Problem: Training crashes mid-run

**Error:**
```
Killed
```

**Causes:**

1. **Out of RAM:**
```bash
# Check RAM usage
htop
```

**Solution:**
```yaml
training:
  num_workers: 2  # Reduce from 4
```

2. **Disk full:**
```bash
df -h
```

**Solution:**
- Clear old checkpoints
- Reduce `save_total_limit`

3. **GPU overheating:**
```bash
nvidia-smi  # Check temperature
```

**Solution:**
- Improve cooling
- Reduce batch size

---

### Problem: Checkpoints not saving

**Check:**
```bash
ls outputs/my_model/
```

**Solutions:**

1. **Check permissions:**
```bash
chmod -R 755 outputs/
```

2. **Check disk space:**
```bash
df -h
```

3. **Verify config:**
```yaml
training:
  save_interval: 10000  # Should be > 0
output:
  dir: outputs/my_model  # Should be writable
```

---

## 📈 Performance Issues

### Problem: GPU utilization low

**Check:**
```bash
nvidia-smi
# GPU-Util should be >80%
```

**Solutions:**

1. **Increase batch size:**
```yaml
training:
  batch_size: 32  # If memory allows
```

2. **Reduce num_workers:**
```yaml
training:
  num_workers: 2  # CPU might be bottleneck
```

3. **Use faster storage:**
   - Move data to SSD
   - Use tmpfs/RAM disk

4. **Enable mixed precision:**
```yaml
training:
  precision: bf16
```

---

### Problem: Slow data loading

**Symptoms:**
GPU idle between batches

**Solutions:**

1. **Increase workers:**
```yaml
training:
  num_workers: 8
  prefetch_factor: 4
```

2. **Use faster disk:**
```bash
# Copy dataset to SSD
cp -r /hdd/data /ssd/data
```

3. **Pre-tokenize dataset:**
```python
# Tokenize once, save to disk
# Load tokenized data during training
```

---

## 🔍 Debugging Tips

### Enable verbose logging

```yaml
output:
  logging:
    log_level: DEBUG
```

Or:
```bash
export PYTHONVERBOSE=1
python train.py --config config.yaml
```

---

### Test with tiny dataset

```bash
# Create tiny test dataset
head -n 100 corpus/*.txt > test_corpus.txt

# Quick test run
python cli/quick_train.py test_corpus.txt \
  --model-size tiny \
  --max-steps 100
```

---

### Monitor with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/

# Open browser to localhost:6006
```

Watch for:
- Loss curve (should decrease)
- Learning rate schedule
- Gradient norms

---

### Profile memory usage

```python
import torch

# Before training
torch.cuda.reset_peak_memory_stats()

# After some steps
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_memory:.2f} GB")
```

---

## 🆘 Getting Help

### 1. Check logs
```bash
# Training logs
cat outputs/my_model/logs/events.out.tfevents.*

# Error logs
cat outputs/my_model/train.log
```

### 2. Run validation
```bash
python cli/validate_config.py config.yaml
```

### 3. Test dataset
```bash
python cli/analyze_dataset.py /path/to/data
```

### 4. Check system
```bash
# GPU
nvidia-smi

# RAM
free -h

# Disk
df -h

# CUDA
nvcc --version
```

### 5. Minimal reproduction
```bash
# Test with minimal config
python examples/01_quick_start.py
```

---

## 📚 Common Error Messages

### "CUDA error: device-side assert triggered"

**Cause:** Invalid token IDs or labels

**Solution:**
- Check vocab_size matches tokenizer
- Verify token IDs < vocab_size
- Check dataset format

---

### "RuntimeError: Expected all tensors to be on the same device"

**Cause:** Model and data on different devices

**Solution:**
```python
# Ensure both on GPU
model = model.to('cuda')
batch = batch.to('cuda')
```

---

### "ValueError: Expected input batch_size (X) to match target batch_size (Y)"

**Cause:** Batch size mismatch

**Solution:**
- Use `drop_last=True` in DataLoader
- Check data collator
- Verify dataset length

---

## 💡 Prevention Tips

1. **Always validate config first**
2. **Start with small test run**
3. **Monitor resources during training**
4. **Use version control for configs**
5. **Keep logs of all runs**
6. **Document changes and results**

---

## 📞 Still Need Help?

1. **Re-read documentation:**
   - `README.md`
   - `docs/quickstart.md`
   - `docs/configuration.md`

2. **Check examples:**
   - `examples/01_quick_start.py`
   - `examples/02_custom_config.py`

3. **Use the wizard:**
```bash
python cli/setup_wizard.py
# It handles most issues automatically
```

4. **Test components individually:**
```bash
# Test analyzer
python cli/analyze_dataset.py /path/to/data

# Test validator
python cli/validate_config.py config.yaml

# Test hardware
python -c "from core import HardwareDetector; HardwareDetector().detect()"
```

---

**Most issues are solved by:**
1. Using the wizard (it validates everything)
2. Starting with tiny model and small dataset
3. Checking logs carefully
4. Reading error messages completely

**Good luck with your training!** 🚀
