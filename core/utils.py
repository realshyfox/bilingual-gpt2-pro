"""Utility functions for the bilingual GPT-2 training system."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def hash_string(s: str) -> str:
    """Generate SHA256 hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()


def get_file_size_gb(path: Union[str, Path]) -> float:
    """Get total size of file or directory in GB."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / (1024**3)
    
    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size / (1024**3)


def format_bytes(bytes_val: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


def round_to_nearest(value: float, nearest: int = 1000) -> int:
    """Round value to nearest multiple."""
    return int(round(value / nearest) * nearest)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_files(directory: Union[str, Path], extensions: Optional[List[str]] = None) -> int:
    """Count files in directory, optionally filtered by extension."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    files = list(directory.rglob("*"))
    if extensions:
        extensions = [ext.lower() for ext in extensions]
        files = [f for f in files if f.suffix.lower() in extensions]
    
    return len([f for f in files if f.is_file()])


def get_cache_dir() -> Path:
    """Get or create cache directory."""
    cache_dir = Path.home() / ".cache" / "bilingual-gpt2-pro"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
