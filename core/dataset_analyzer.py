"""
Smart Dataset Analyzer - THE CORNERSTONE
Analyzes datasets BEFORE training to prevent wasted GPU time.
KEY PRINCIPLE: Use token count, NOT file size!
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import time

import numpy as np
from tqdm import tqdm

from .utils import (
    get_file_size_gb, format_bytes, format_time,
    round_to_nearest, clamp, ensure_dir, get_cache_dir
)


class SmartDatasetAnalyzer:
    """
    Intelligent dataset analyzer that prevents training errors by analyzing
    datasets BEFORE GPU time is wasted.
    
    Core Innovation: Uses TOKEN COUNT, not file size, for accurate analysis.
    """
    
    # Dataset type validation matrix
    VALIDATION_MATRIX = {
        ("text_corpus", "pre-training"): True,
        ("text_corpus", "fine-tuning"): False,
        ("qa", "pre-training"): False,
        ("qa", "fine-tuning"): True,
        ("instruction", "pre-training"): False,
        ("instruction", "fine-tuning"): True,
        ("conversational", "pre-training"): False,
        ("conversational", "fine-tuning"): True,
        ("code", "pre-training"): True,
        ("code", "fine-tuning"): True,
    }
    
    def __init__(
        self,
        data_path: Union[str, Path],
        task_type: str = "pre-training",
        cache_enabled: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the dataset analyzer.
        
        Args:
            data_path: Path to dataset (directory or file)
            task_type: "pre-training" or "fine-tuning"
            cache_enabled: Whether to cache analysis results
            verbose: Whether to print progress
        """
        self.data_path = Path(data_path)
        self.task_type = task_type
        self.cache_enabled = cache_enabled
        self.verbose = verbose
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        self.cache_dir = get_cache_dir() / "dataset_analysis"
        ensure_dir(self.cache_dir)
        
        self.results = {}
    
    def analyze(
        self,
        sampling_percentage: Optional[float] = None,
        language_detector: Optional[object] = None
    ) -> Dict:
        """
        Run complete dataset analysis.
        
        Args:
            sampling_percentage: Percentage of data to analyze (None = auto-decide)
            language_detector: Optional language detector instance
        
        Returns:
            Complete analysis results dictionary
        """
        if self.verbose:
            print("\n" + "="*70)
            print("  SMART DATASET ANALYZER")
            print("="*70)
        
        # Check cache
        cache_key = self._get_cache_key(sampling_percentage)
        if self.cache_enabled:
            cached = self._load_cache(cache_key)
            if cached:
                if self.verbose:
                    print("✅ Loaded from cache\n")
                self.results = cached
                return cached
        
        start_time = time.time()
        
        # Step 1: Collect files
        files = self._collect_files()
        size_gb = get_file_size_gb(self.data_path)
        
        if self.verbose:
            print(f"\n📁 Dataset: {self.data_path}")
            print(f"📊 Size: {size_gb:.2f} GB")
            print(f"📄 Files: {len(files)}")
        
        # Step 2: Determine sampling strategy
        if sampling_percentage is None:
            sampling_percentage = self._get_sampling_recommendation(size_gb)
        
        if self.verbose and sampling_percentage < 100:
            print(f"🎯 Sampling: {sampling_percentage}%")
        
        # Step 3: Sample files
        sampled_files = self._sample_files(files, sampling_percentage)
        
        # Step 4: Detect dataset type
        dataset_type, type_confidence = self._detect_dataset_type(sampled_files[:10])
        
        if self.verbose:
            print(f"\n🔍 Type: {dataset_type.upper()} (confidence: {type_confidence:.1%})")
        
        # Step 5: Count tokens (CRITICAL!)
        token_stats = self._count_tokens(sampled_files, sampling_percentage)
        
        # Step 6: Detect languages
        language_stats = self._detect_languages(sampled_files, language_detector)
        
        # Step 7: Quality metrics
        quality_metrics = self._calculate_quality_metrics(sampled_files)
        
        # Step 8: Calculate recommendations
        recommendations = self._generate_recommendations(
            token_stats, language_stats, quality_metrics, dataset_type
        )
        
        # Step 9: Task validation
        task_compatible = self._validate_task_compatibility(dataset_type)
        
        elapsed = time.time() - start_time
        
        # Compile results
        self.results = {
            "dataset_path": str(self.data_path),
            "task_type": self.task_type,
            "size_gb": size_gb,
            "num_files": len(files),
            "sampling_percentage": sampling_percentage,
            "dataset_type": dataset_type,
            "type_confidence": type_confidence,
            "task_compatible": task_compatible,
            "token_stats": token_stats,
            "language_stats": language_stats,
            "quality_metrics": quality_metrics,
            "recommendations": recommendations,
            "analysis_time": elapsed,
        }
        
        # Cache results
        if self.cache_enabled:
            self._save_cache(cache_key, self.results)
        
        if self.verbose:
            print(f"\n✅ Analysis complete in {format_time(elapsed)}\n")
        
        return self.results
    
    def print_report(self):
        """Print comprehensive analysis report."""
        if not self.results:
            print("⚠️  No analysis results. Run analyze() first.")
            return
        
        r = self.results
        
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " "*18 + "DATASET ANALYSIS REPORT" + " "*27 + "║")
        print("╚" + "═"*68 + "╝")
        
        print(f"\n📁 Dataset Path: {r['dataset_path']}")
        print(f"📊 Files Found: {r['num_files']} files")
        print(f"💾 Total Size: {r['size_gb']:.2f} GB")
        
        # Dataset Type
        print("\n" + "━"*70)
        print("1. DATASET TYPE")
        print("━"*70)
        print(f"\nType: {r['dataset_type'].upper()} ", end="")
        print(f"(confidence: {r['type_confidence']:.1%})")
        
        if r['task_compatible']:
            print(f"✅ COMPATIBLE with {r['task_type']}")
        else:
            print(f"❌ NOT suitable for {r['task_type']}")
            print(f"   This dataset type is for different tasks.")
        
        # Token Analysis
        print("\n" + "━"*70)
        print("2. TOKEN ANALYSIS")
        print("━"*70)
        
        ts = r['token_stats']
        total_tokens = ts['total_tokens']
        
        if r['sampling_percentage'] < 100:
            print(f"\nTotal tokens: {total_tokens:,} (from {r['sampling_percentage']}% sample)")
            estimated_full = int(total_tokens / (r['sampling_percentage'] / 100))
            print(f"Estimated full: {estimated_full:,}")
        else:
            print(f"\nTotal tokens: {total_tokens:,}")
        
        print(f"Unique words: {ts['unique_words']:,}")
        print(f"Type-Token Ratio: {ts['type_token_ratio']:.4f}", end="")
        
        if ts['type_token_ratio'] > 0.01:
            print(" (low diversity)")
        elif ts['type_token_ratio'] > 0.001:
            print(" (good diversity)")
        else:
            print(" (excellent diversity)")
        
        # Recommendations
        recs = r['recommendations']
        print(f"\n🎯 RECOMMENDED VOCAB SIZE: {recs['recommended_vocab']:,}")
        print(f"   BPE: ~{recs['bpe_vocab']:,}")
        print(f"   Unigram: ~{recs['unigram_vocab']:,} ✅ ({recs['efficiency_gain']:.0%} more efficient)")
        
        # Languages
        print("\n" + "━"*70)
        print("3. LANGUAGES")
        print("━"*70)
        
        langs = r['language_stats']
        for lang_info in langs['languages'][:5]:  # Top 5
            flag = self._get_flag(lang_info['language'])
            pct = lang_info['percentage']
            print(f"\n{flag} {lang_info['language'].title()}: {pct:.1f}%")
            
            if pct < 1.0 and pct > 0.5:
                print(f"   ⚠️  Possible contamination")
        
        if langs['num_languages'] > 5:
            print(f"\n... and {langs['num_languages'] - 5} more languages")
        
        # Quality
        print("\n" + "━"*70)
        print("4. QUALITY")
        print("━"*70)
        
        qm = r['quality_metrics']
        
        dup_status = "✅" if qm['duplicate_rate'] < 5 else "⚠️"
        print(f"\n{dup_status} Duplicate rate: {qm['duplicate_rate']:.1f}%")
        
        enc_status = "✅" if qm['encoding_errors'] == 0 else "❌"
        print(f"{enc_status} Encoding: UTF-8, {qm['encoding_errors']} errors")
        
        quality_score = qm['quality_score']
        if quality_score >= 80:
            score_emoji = "✅"
            score_label = "excellent"
        elif quality_score >= 60:
            score_emoji = "⚠️"
            score_label = "good"
        else:
            score_emoji = "❌"
            score_label = "needs improvement"
        
        print(f"{score_emoji} Quality score: {quality_score:.0f}/100 ({score_label})")
        
        # Recommendations
        print("\n" + "━"*70)
        print("5. RECOMMENDATIONS")
        print("━"*70)
        
        print(f"\nTokenizer: {recs['tokenizer_type']}")
        print(f"Vocab size: {recs['recommended_vocab']:,}")
        print(f"Model: {recs['recommended_model']}")
        print(f"Training steps: {recs['training_steps']:,}")
        
        if recs.get('warnings'):
            print("\n⚠️  Warnings:")
            for warning in recs['warnings']:
                print(f"   • {warning}")
        
        print("\n" + "═"*70 + "\n")
    
    def _collect_files(self) -> List[Path]:
        """Collect all text files from dataset path."""
        if self.data_path.is_file():
            return [self.data_path]
        
        text_extensions = {'.txt', '.text', '.json', '.jsonl', '.csv', '.tsv'}
        files = []
        
        for file_path in self.data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                files.append(file_path)
        
        return sorted(files)
    
    def _get_sampling_recommendation(self, size_gb: float) -> float:
        """Get recommended sampling percentage based on dataset size."""
        if size_gb < 1:
            return 100.0
        elif size_gb < 10:
            return 50.0
        elif size_gb < 50:
            return 25.0
        elif size_gb < 100:
            return 15.0
        else:
            return 10.0
    
    def _sample_files(self, files: List[Path], percentage: float) -> List[Path]:
        """Sample files based on percentage."""
        if percentage >= 100:
            return files
        
        num_to_sample = max(1, int(len(files) * percentage / 100))
        return random.sample(files, num_to_sample)
    
    def _detect_dataset_type(self, sample_files: List[Path]) -> Tuple[str, float]:
        """
        Detect dataset type from sample files.
        
        Returns:
            (type, confidence) tuple
        """
        type_scores = defaultdict(float)
        
        for file_path in sample_files[:5]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(10000)  # First 10KB
                
                # Check for JSON structure
                if file_path.suffix in {'.json', '.jsonl'}:
                    try:
                        if file_path.suffix == '.jsonl':
                            sample = json.loads(content.split('\n')[0])
                        else:
                            sample = json.loads(content)
                        
                        # Check keys
                        if isinstance(sample, dict):
                            keys = set(sample.keys())
                            
                            if 'question' in keys and 'answer' in keys:
                                type_scores['qa'] += 2
                            elif 'instruction' in keys or 'input' in keys:
                                type_scores['instruction'] += 2
                            elif 'messages' in keys or 'conversation' in keys:
                                type_scores['conversational'] += 2
                            else:
                                type_scores['text_corpus'] += 1
                    except json.JSONDecodeError:
                        type_scores['text_corpus'] += 1
                
                # Check for code patterns
                code_patterns = [
                    r'def\s+\w+\s*\(',
                    r'class\s+\w+',
                    r'import\s+\w+',
                    r'function\s+\w+\s*\(',
                    r'=>',
                ]
                code_matches = sum(1 for p in code_patterns if re.search(p, content))
                if code_matches >= 2:
                    type_scores['code'] += code_matches
                
                # Default to text corpus
                type_scores['text_corpus'] += 0.5
                
            except Exception:
                type_scores['text_corpus'] += 0.5
        
        if not type_scores:
            return 'text_corpus', 0.5
        
        best_type = max(type_scores, key=type_scores.get)
        total_score = sum(type_scores.values())
        confidence = type_scores[best_type] / total_score if total_score > 0 else 0.5
        
        return best_type, confidence
    
    def _count_tokens(
        self,
        files: List[Path],
        sampling_pct: float
    ) -> Dict:
        """
        Count tokens - THE CRITICAL FUNCTION!
        Uses token count, NOT file size.
        """
        if self.verbose:
            print("\n🔢 Counting tokens...")
        
        total_chars = 0
        total_tokens = 0
        unique_words = set()
        unique_bigrams = set()
        unique_trigrams = set()
        
        doc_lengths = []
        
        iterator = tqdm(files, desc="Processing files") if self.verbose else files
        
        for file_path in iterator:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # Character count
                total_chars += len(text)
                
                # Word tokenization (simple split)
                words = text.lower().split()
                total_tokens += len(words)
                doc_lengths.append(len(words))
                
                # Unique words
                unique_words.update(words)
                
                # Bigrams
                for i in range(len(words) - 1):
                    unique_bigrams.add((words[i], words[i+1]))
                
                # Trigrams
                for i in range(len(words) - 2):
                    unique_trigrams.add((words[i], words[i+1], words[i+2]))
                
            except Exception:
                continue
        
        # Estimate subword tokens (more realistic)
        estimated_subwords = len(unique_bigrams) + len(unique_words) // 2
        
        # Adjust for sampling
        if sampling_pct < 100:
            scale_factor = 100 / sampling_pct
            total_tokens = int(total_tokens * scale_factor)
            total_chars = int(total_chars * scale_factor)
        
        # Calculate type-token ratio
        type_token_ratio = len(unique_words) / total_tokens if total_tokens > 0 else 0
        
        # Document length statistics
        if doc_lengths:
            avg_doc_length = np.mean(doc_lengths)
            std_doc_length = np.std(doc_lengths)
            median_doc_length = np.median(doc_lengths)
        else:
            avg_doc_length = std_doc_length = median_doc_length = 0
        
        return {
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "unique_words": len(unique_words),
            "unique_bigrams": len(unique_bigrams),
            "unique_trigrams": len(unique_trigrams),
            "estimated_subwords": estimated_subwords,
            "type_token_ratio": type_token_ratio,
            "avg_doc_length": avg_doc_length,
            "std_doc_length": std_doc_length,
            "median_doc_length": median_doc_length,
        }
    
    def _detect_languages(
        self,
        files: List[Path],
        language_detector: Optional[object] = None
    ) -> Dict:
        """Detect languages in dataset."""
        if language_detector is None:
            # Try to import lingua
            try:
                from lingua import Language, LanguageDetectorBuilder
                detector = LanguageDetectorBuilder.from_all_languages().build()
            except ImportError:
                if self.verbose:
                    print("⚠️  lingua-language-detector not installed, skipping language detection")
                return {
                    "languages": [{"language": "unknown", "percentage": 100.0}],
                    "num_languages": 1,
                }
        else:
            detector = language_detector
        
        if self.verbose:
            print("\n🌍 Detecting languages...")
        
        language_tokens = Counter()
        total_tokens = 0
        
        # Sample up to 100 files for language detection
        sample_files = random.sample(files, min(100, len(files)))
        
        for file_path in sample_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read(50000)  # First 50KB
                
                # Detect language
                try:
                    detected = detector.detect_language_of(text)
                    if detected:
                        lang_name = detected.name.lower()
                        tokens = len(text.split())
                        language_tokens[lang_name] += tokens
                        total_tokens += tokens
                except Exception:
                    pass
                    
            except Exception:
                continue
        
        # Calculate percentages
        languages = []
        for lang, tokens in language_tokens.most_common():
            pct = (tokens / total_tokens * 100) if total_tokens > 0 else 0
            languages.append({
                "language": lang,
                "tokens": tokens,
                "percentage": pct,
            })
        
        return {
            "languages": languages,
            "num_languages": len(languages),
        }
    
    def _calculate_quality_metrics(self, files: List[Path]) -> Dict:
        """Calculate dataset quality metrics."""
        if self.verbose:
            print("\n✨ Calculating quality...")
        
        # Hash-based duplicate detection
        hashes = set()
        duplicates = 0
        total_docs = 0
        encoding_errors = 0
        
        lengths = []
        
        for file_path in files[:1000]:  # Sample for quality
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Hash
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in hashes:
                    duplicates += 1
                else:
                    hashes.add(text_hash)
                
                total_docs += 1
                lengths.append(len(text))
                
            except UnicodeDecodeError:
                encoding_errors += 1
            except Exception:
                continue
        
        duplicate_rate = (duplicates / total_docs * 100) if total_docs > 0 else 0
        
        # Length distribution (detect outliers)
        if lengths:
            q1, q3 = np.percentile(lengths, [25, 75])
            iqr = q3 - q1
            outliers = sum(1 for l in lengths if l < q1 - 1.5*iqr or l > q3 + 1.5*iqr)
            outlier_rate = outliers / len(lengths) * 100
        else:
            outlier_rate = 0
        
        # Content diversity (Shannon entropy)
        if lengths:
            entropy = -sum((l/sum(lengths)) * np.log2(l/sum(lengths)) for l in lengths if l > 0)
        else:
            entropy = 0
        
        # Quality score (0-100)
        quality_score = 100
        quality_score -= min(duplicate_rate * 6, 30)  # Penalize duplicates
        quality_score -= min(encoding_errors * 20, 20)  # Penalize errors
        quality_score -= min(outlier_rate * 2, 10)  # Penalize outliers
        quality_score += min(entropy / 15 * 30, 30)  # Reward diversity
        quality_score = max(0, min(100, quality_score))
        
        return {
            "duplicate_rate": duplicate_rate,
            "encoding_errors": encoding_errors,
            "outlier_rate": outlier_rate,
            "entropy": entropy,
            "quality_score": quality_score,
        }
    
    def _generate_recommendations(
        self,
        token_stats: Dict,
        language_stats: Dict,
        quality_metrics: Dict,
        dataset_type: str
    ) -> Dict:
        """Generate intelligent recommendations."""
        # Optimal vocab size based on tokens
        estimated_subwords = token_stats['estimated_subwords']
        
        bpe_vocab = int(estimated_subwords * 0.65)
        unigram_vocab = int(estimated_subwords * 0.55)
        
        # Clamp and round
        bpe_vocab = round_to_nearest(clamp(bpe_vocab, 8000, 100000), 1000)
        unigram_vocab = round_to_nearest(clamp(unigram_vocab, 8000, 100000), 1000)
        
        # Recommend Unigram for multilingual
        num_languages = language_stats['num_languages']
        if num_languages >= 2:
            recommended_vocab = unigram_vocab
            tokenizer_type = "SentencePiece Unigram"
        else:
            recommended_vocab = min(bpe_vocab, unigram_vocab)
            tokenizer_type = "BPE or Unigram"
        
        efficiency_gain = (bpe_vocab - unigram_vocab) / bpe_vocab if bpe_vocab > 0 else 0
        
        # Model recommendation
        total_tokens = token_stats['total_tokens']
        if total_tokens < 1e9:
            recommended_model = "Tiny (40M)"
        elif total_tokens < 5e9:
            recommended_model = "Mini (124M)"
        elif total_tokens < 20e9:
            recommended_model = "Small (350M)"
        else:
            recommended_model = "Medium (760M)"
        
        # Training steps
        # Rule: ~20 tokens per parameter
        if "Tiny" in recommended_model:
            target_tokens = 40e6 * 20
        elif "Mini" in recommended_model:
            target_tokens = 124e6 * 20
        elif "Small" in recommended_model:
            target_tokens = 350e6 * 20
        else:
            target_tokens = 760e6 * 20
        
        training_steps = int(target_tokens / total_tokens * 500000) if total_tokens > 0 else 100000
        training_steps = round_to_nearest(training_steps, 10000)
        
        # Warnings
        warnings = []
        if quality_metrics['quality_score'] < 60:
            warnings.append("Low quality score - consider data cleaning")
        if quality_metrics['duplicate_rate'] > 10:
            warnings.append("High duplicate rate - deduplication recommended")
        if num_languages > 5:
            warnings.append("Many languages detected - consider focused training")
        
        return {
            "tokenizer_type": tokenizer_type,
            "recommended_vocab": recommended_vocab,
            "bpe_vocab": bpe_vocab,
            "unigram_vocab": unigram_vocab,
            "efficiency_gain": efficiency_gain,
            "recommended_model": recommended_model,
            "training_steps": training_steps,
            "warnings": warnings,
        }
    
    def _validate_task_compatibility(self, dataset_type: str) -> bool:
        """Validate if dataset type is compatible with task."""
        return self.VALIDATION_MATRIX.get((dataset_type, self.task_type), False)
    
    def _get_cache_key(self, sampling_pct: Optional[float]) -> str:
        """Generate cache key for analysis results."""
        # Include file modification times in hash
        files = self._collect_files()
        mtimes = [f.stat().st_mtime for f in files[:100]]  # Sample
        mtime_sum = sum(mtimes)
        
        key_str = f"{self.data_path}_{sampling_pct}_{mtime_sum}_{self.task_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached analysis results."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def _save_cache(self, cache_key: str, results: Dict):
        """Save analysis results to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception:
            pass
    
    def _get_flag(self, language: str) -> str:
        """Get flag emoji for language."""
        flags = {
            'english': '🇬🇧',
            'spanish': '🇪🇸',
            'french': '🇫🇷',
            'german': '🇩🇪',
            'italian': '🇮🇹',
            'portuguese': '🇵🇹',
            'russian': '🇷🇺',
            'chinese': '🇨🇳',
            'japanese': '🇯🇵',
            'korean': '🇰🇷',
            'arabic': '🇸🇦',
            'hindi': '🇮🇳',
        }
        return flags.get(language.lower(), '🌍')
