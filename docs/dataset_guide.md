# Dataset Preparation Guide

Complete guide for preparing datasets for bilingual GPT-2 training.

---

## 📊 Supported Dataset Types

### 1. Text Corpus (Pre-training)
**Best for:** Language modeling, general knowledge

**Format:**
```
corpus/
├── file1.txt
├── file2.txt
└── file3.txt
```

**Requirements:**
- Plain text files (.txt)
- UTF-8 encoding
- One document per file OR continuous text
- Minimum: 1GB (100M tokens)
- Recommended: 10GB+ (1B+ tokens)

**Example:**
```text
This is a document in English about machine learning.
It can contain multiple paragraphs and sentences.

Machine learning is a subset of artificial intelligence...
```

---

### 2. Question-Answer Pairs (Fine-tuning)
**Best for:** Q&A systems, chatbots

**Format:** JSONL (one JSON object per line)
```jsonl
{"question": "What is machine learning?", "answer": "Machine learning is..."}
{"question": "How does GPT work?", "answer": "GPT uses transformer architecture..."}
```

**Requirements:**
- `.jsonl` extension
- Each line is valid JSON
- Must have "question" and "answer" keys
- Minimum: 1,000 pairs
- Recommended: 10,000+ pairs

---

### 3. Instruction Following (Fine-tuning)
**Best for:** Task completion, instruction following

**Format:** JSONL
```jsonl
{"instruction": "Translate to Spanish:", "input": "Hello world", "output": "Hola mundo"}
{"instruction": "Summarize:", "input": "Long text...", "output": "Summary..."}
```

**Requirements:**
- `.jsonl` extension
- Keys: "instruction", "input" (optional), "output"
- Minimum: 1,000 examples
- Recommended: 10,000+ examples

---

### 4. Conversational (Fine-tuning)
**Best for:** Dialog systems, chatbots

**Format:** JSONL
```jsonl
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"messages": [{"role": "user", "content": "Help me"}, {"role": "assistant", "content": "Sure!"}]}
```

---

## 🌍 Bilingual Dataset Preparation

### Option 1: Mixed Documents
**Structure:**
```
corpus/
├── english/
│   ├── doc1_en.txt
│   └── doc2_en.txt
└── spanish/
    ├── doc1_es.txt
    └── doc2_es.txt
```

**Pros:**
- Easy to organize
- Language-specific analysis possible
- Can balance languages easily

**Cons:**
- Requires manual organization

---

### Option 2: Parallel Corpus
**Best for:** Translation tasks

**Format:**
```
data/
├── source_en.txt
└── target_es.txt
```

Each line corresponds:
```
# source_en.txt
Hello world
How are you?

# target_es.txt
Hola mundo
¿Cómo estás?
```

---

### Option 3: Mixed Single Files
**Structure:**
```
corpus/
└── mixed.txt
```

**Content:**
```text
This is English text about AI.
Este es texto en español sobre IA.
More English content here.
Más contenido en español aquí.
```

**Pros:**
- Natural language mixing
- Single file simplicity

**Cons:**
- Harder to balance languages

---

## ✅ Dataset Quality Checklist

### Encoding
- [ ] All files are UTF-8 encoded
- [ ] No binary data mixed in
- [ ] Special characters handled correctly

### Content
- [ ] No duplicate documents (>10% is bad)
- [ ] Diverse topics and styles
- [ ] Natural language (not machine-generated spam)
- [ ] Appropriate for your use case

### Size
- [ ] Pre-training: 1GB+ (minimum), 10GB+ (recommended)
- [ ] Fine-tuning: 1,000+ examples (minimum), 10,000+ (recommended)

### Languages
- [ ] Language ratio matches your needs
- [ ] No unexpected language contamination (>1%)
- [ ] Each language has sufficient examples

---

## 🔧 Data Cleaning

### Remove Duplicates
```bash
# Using sort and uniq
sort corpus/*.txt | uniq > cleaned.txt
```

### Fix Encoding
```python
import chardet

def fix_encoding(input_file, output_file):
    with open(input_file, 'rb') as f:
        raw = f.read()
        encoding = chardet.detect(raw)['encoding']
    
    with open(input_file, 'r', encoding=encoding) as f:
        text = f.read()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
```

### Remove Short Documents
```python
def filter_short_docs(input_dir, output_dir, min_words=50):
    for file in input_dir.glob('*.txt'):
        with open(file, 'r') as f:
            text = f.read()
        
        if len(text.split()) >= min_words:
            output_path = output_dir / file.name
            with open(output_path, 'w') as f:
                f.write(text)
```

---

## 📈 Dataset Analysis

### Use the Built-in Analyzer
```bash
python cli/analyze_dataset.py /path/to/corpus
```

**This shows:**
- Total tokens (accurate count!)
- Language distribution
- Quality score (0-100)
- Duplicate rate
- Recommended vocab size
- Optimal model size

### Manual Verification
```python
from core import SmartDatasetAnalyzer

analyzer = SmartDatasetAnalyzer('/path/to/corpus')
results = analyzer.analyze()
analyzer.print_report()

# Access specific metrics
print(f"Total tokens: {results['token_stats']['total_tokens']:,}")
print(f"Languages: {results['language_stats']['num_languages']}")
print(f"Quality: {results['quality_metrics']['quality_score']:.0f}/100")
```

---

## 🎯 Recommendations by Use Case

### General Language Model (Bilingual)
- **Size:** 20GB+ (2B+ tokens)
- **Languages:** 50/50 split
- **Type:** Text corpus
- **Quality:** >80 quality score

### Domain-Specific Model (e.g., Medical)
- **Size:** 5GB+ (500M+ tokens)
- **Languages:** Match your domain needs
- **Type:** Text corpus from domain
- **Quality:** >90 quality score (domain-specific)

### Chatbot (Fine-tuning)
- **Size:** 10,000+ conversations
- **Type:** Conversational format
- **Quality:** Natural, diverse dialogs

### Q&A System (Fine-tuning)
- **Size:** 5,000+ Q&A pairs
- **Type:** Question-answer format
- **Quality:** Accurate, diverse questions

---

## 📥 Data Sources

### Pre-training Corpora
- **Wikipedia:** Clean, multilingual (dumps.wikimedia.org)
- **Common Crawl:** Massive web data (commoncrawl.org)
- **OSCAR:** Filtered web data (oscar-corpus.com)
- **CC-100:** Curated Common Crawl subset

### Fine-tuning Datasets
- **HuggingFace Datasets:** hub.huggingface.co/datasets
- **SQuAD:** Question answering (rajpurkar.github.io/SQuAD-explorer)
- **Natural Questions:** Google's Q&A dataset
- **XNLI:** Cross-lingual inference

### Bilingual Resources
- **Tatoeba:** Parallel sentences (tatoeba.org)
- **OpenSubtitles:** Movie subtitles (opus.nlpl.eu)
- **Europarl:** European Parliament proceedings
- **UN Corpus:** United Nations documents

---

## ⚠️ Common Issues

### Issue: "Encoding errors"
**Solution:**
```bash
# Convert all files to UTF-8
for file in *.txt; do
    iconv -f ISO-8859-1 -t UTF-8 "$file" > "utf8_$file"
done
```

### Issue: "High duplicate rate"
**Solution:**
```python
# Deduplicate using hashing
seen = set()
with open('deduplicated.txt', 'w') as out:
    for file in glob('*.txt'):
        with open(file) as f:
            text = f.read()
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                out.write(text + '\n\n')
```

### Issue: "Language contamination"
**Solution:**
```python
from lingua import LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_languages().build()

# Filter by language
for file in glob('*.txt'):
    with open(file) as f:
        text = f.read()
    
    language = detector.detect_language_of(text)
    if language and language.name.lower() in ['english', 'spanish']:
        # Keep file
        pass
    else:
        # Remove or move to separate folder
        print(f"Unexpected language in {file}: {language}")
```

### Issue: "Dataset too small"
**Solutions:**
1. Download more data from public sources
2. Use data augmentation (paraphrasing, back-translation)
3. Start with smaller model (Tiny or Mini)
4. Use transfer learning from pretrained model

---

## 🚀 Quick Start Examples

### Example 1: Wikipedia Dump
```bash
# Download Wikipedia dump
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Extract text (using wikiextractor)
pip install wikiextractor
wikiextractor enwiki-latest-pages-articles.xml.bz2 -o corpus/

# Analyze
python cli/analyze_dataset.py corpus/
```

### Example 2: Custom Text Files
```bash
# Organize your files
mkdir -p data/corpus
cp /path/to/your/*.txt data/corpus/

# Analyze
python cli/analyze_dataset.py data/corpus/

# Review report and start training
python cli/setup_wizard.py
```

### Example 3: HuggingFace Dataset
```python
from datasets import load_dataset

# Download dataset
dataset = load_dataset('oscar', 'unshuffled_deduplicated_en')

# Save to text files
with open('corpus/oscar_en.txt', 'w') as f:
    for item in dataset['train']:
        f.write(item['text'] + '\n\n')

# Analyze
python cli/analyze_dataset.py corpus/
```

---

## 📊 Expected Results

After running the analyzer, you should see:

**Good Dataset:**
- Quality score: 80-100
- Duplicate rate: <5%
- Encoding errors: 0
- Balanced languages (if bilingual)
- Diverse content

**Needs Improvement:**
- Quality score: <60
- Duplicate rate: >10%
- Encoding errors: >0
- Unbalanced languages
- Repetitive content

---

## 💡 Pro Tips

1. **Start small:** Test with 1GB before processing 100GB
2. **Use sampling:** Analyzer can sample large datasets (faster)
3. **Cache results:** Analysis is cached for reuse
4. **Balance languages:** Aim for 40-60% split for bilingual
5. **Quality over quantity:** 10GB of good data > 100GB of spam
6. **Verify manually:** Spot-check random samples
7. **Document source:** Keep track of where data came from

---

## 📚 Additional Resources

- **Dataset analyzer docs:** See `core/dataset_analyzer.py` docstrings
- **Configuration guide:** See `docs/configuration.md`
- **Quick start:** See `docs/quickstart.md`

---

**Need help?** Run `python cli/analyze_dataset.py --help` for options.
