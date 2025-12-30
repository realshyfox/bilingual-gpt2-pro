"""
Hardware Detector
Auto-detects GPU capabilities and system resources.
"""

import subprocess
import platform
from typing import Dict, List, Optional
import os


class HardwareDetector:
    """Detects and validates hardware configuration."""
    
    def __init__(self, verbose: bool = True):
        """Initialize hardware detector."""
        self.verbose = verbose
        self.config = {}
    
    def detect(self) -> Dict:
        """
        Detect all hardware configuration.
        
        Returns:
            Hardware configuration dictionary
        """
        if self.verbose:
            print("\n" + "="*70)
            print("  HARDWARE DETECTION")
            print("="*70 + "\n")
        
        # Detect GPUs
        gpu_info = self._detect_gpus()
        
        # Detect CPU
        cpu_info = self._detect_cpu()
        
        # Detect RAM
        ram_gb = self._detect_ram()
        
        # Detect CUDA
        cuda_version = self._detect_cuda()
        
        # Compile config
        self.config = {
            "gpus": gpu_info,
            "num_gpus": len(gpu_info),
            "cpu": cpu_info,
            "ram_gb": ram_gb,
            "cuda_version": cuda_version,
            "platform": platform.system(),
            "python_version": platform.python_version(),
        }
        
        if self.verbose:
            self._print_config()
        
        return self.config
    
    def _detect_gpus(self) -> List[Dict]:
        """Detect NVIDIA GPUs using nvidia-smi."""
        gpus = []
        
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    
                    gpu = {
                        "index": i,
                        "name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "total_memory_gb": props.total_memory / (1024**3),
                        "supports_bf16": props.major >= 8,  # Ampere+
                        "supports_fp16": True,
                        "tensor_cores": props.major >= 7,  # Volta+
                    }
                    gpus.append(gpu)
        except ImportError:
            if self.verbose:
                print("⚠️  PyTorch not installed, cannot detect GPUs")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  GPU detection failed: {e}")
        
        return gpus
    
    def _detect_cpu(self) -> Dict:
        """Detect CPU information."""
        cpu_info = {
            "processor": platform.processor(),
            "architecture": platform.machine(),
        }
        
        # Try to get CPU count
        try:
            cpu_info["physical_cores"] = os.cpu_count()
        except Exception:
            cpu_info["physical_cores"] = "unknown"
        
        return cpu_info
    
    def _detect_ram(self) -> float:
        """Detect system RAM in GB."""
        try:
            if platform.system() == "Linux":
                # Read from /proc/meminfo
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return kb / (1024**2)  # Convert to GB
            else:
                # Try psutil as fallback
                try:
                    import psutil
                    return psutil.virtual_memory().total / (1024**3)
                except ImportError:
                    pass
        except Exception:
            pass
        
        return 0.0
    
    def _detect_cuda(self) -> Optional[str]:
        """Detect CUDA version."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda
        except ImportError:
            pass
        
        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        return None
    
    def _print_config(self):
        """Print detected hardware configuration."""
        print("🖥️  System Information:\n")
        print(f"  Platform: {self.config['platform']}")
        print(f"  Python: {self.config['python_version']}")
        
        if self.config['ram_gb'] > 0:
            print(f"  RAM: {self.config['ram_gb']:.1f} GB")
        
        print(f"\n💻 CPU:\n")
        cpu = self.config['cpu']
        print(f"  Architecture: {cpu['architecture']}")
        if cpu.get('physical_cores'):
            print(f"  Cores: {cpu['physical_cores']}")
        
        if self.config['cuda_version']:
            print(f"\n🔷 CUDA: {self.config['cuda_version']}")
        
        print(f"\n🎮 GPUs: {self.config['num_gpus']} detected\n")
        
        if self.config['num_gpus'] == 0:
            print("  ⚠️  No GPUs detected!")
            print("  Training will run on CPU (very slow)")
        else:
            for gpu in self.config['gpus']:
                print(f"  GPU {gpu['index']}: {gpu['name']}")
                print(f"    • VRAM: {gpu['total_memory_gb']:.1f} GB")
                print(f"    • Compute: {gpu['compute_capability']}")
                
                features = []
                if gpu['supports_bf16']:
                    features.append("BF16 ✅")
                if gpu['tensor_cores']:
                    features.append("Tensor Cores ✅")
                
                if features:
                    print(f"    • Features: {', '.join(features)}")
                print()
        
        print("─"*70 + "\n")
    
    def get_recommended_precision(self) -> str:
        """Get recommended precision based on hardware."""
        if self.config['num_gpus'] == 0:
            return "fp32"
        
        # Check if any GPU supports BF16
        for gpu in self.config['gpus']:
            if gpu.get('supports_bf16', False):
                return "bf16"
        
        # Fallback to FP16
        return "fp16"
    
    def get_total_vram(self) -> float:
        """Get total VRAM across all GPUs in GB."""
        return sum(gpu['total_memory_gb'] for gpu in self.config['gpus'])
    
    def get_min_vram(self) -> float:
        """Get minimum VRAM of any single GPU in GB."""
        if not self.config['gpus']:
            return 0.0
        return min(gpu['total_memory_gb'] for gpu in self.config['gpus'])
    
    def validate_for_training(self) -> tuple[bool, List[str]]:
        """
        Validate hardware is suitable for training.
        
        Returns:
            (is_valid, warnings) tuple
        """
        warnings = []
        
        # Check for GPU
        if self.config['num_gpus'] == 0:
            warnings.append("No GPU detected - training will be VERY slow")
        
        # Check VRAM
        min_vram = self.get_min_vram()
        if min_vram < 8:
            warnings.append(f"Low VRAM ({min_vram:.1f}GB) - may limit model size")
        
        # Check CUDA
        if not self.config.get('cuda_version'):
            warnings.append("CUDA not detected - GPU training unavailable")
        
        # Check RAM
        if self.config['ram_gb'] < 16:
            warnings.append("Low system RAM - may cause issues with large datasets")
        
        is_valid = self.config['num_gpus'] > 0 and self.config.get('cuda_version') is not None
        
        return is_valid, warnings
    
    def save_config(self, path: str):
        """Save hardware config to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
