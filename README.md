# Cross-Domain QA Dataset

A multi-domain question-answering dataset combining legal, medical, financial, and reading comprehension tasks for evaluating cross-domain performance of language models and knowledge systems.

## Overview

This repository provides a script to generate a cross-domain QA dataset from four distinct domains:

- **Legal**: Merger Agreement Understanding Dataset (MAUD) from LegalBench
- **Medical**: PubMedQA labeled dataset  
- **Financial**: FinQA dataset with generated multiple-choice answers
- **Reading Comprehension**: MCTest dataset (MC500)

The resulting dataset is designed for evaluating how well models and ontology learning approaches preserve information across different specialized domains.

## Installation

### Prerequisites

- Python 3.13+
- Git with submodule support

### Setup

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/albsadowski/qa-cross-domain-dataset.git
cd qa-cross-domain-dataset
```

2. Install Python dependencies:
```bash
uv sync
```

3. Initialize submodules (if not cloned recursively):
```bash
git submodule init
git submodule update
```

## Usage

### Basic Usage

Generate a balanced dataset with 600 examples per domain:

```bash
uv run main.py --output cross_domain_dataset.csv
```

### Configuration Options

```bash
uv run main.py \
  --output dataset.csv \
  --target-size 500 \
  --random-seed 42
```

**Parameters:**
- `--output`: Output CSV filename (required)
- `--target-size`: Number of examples per domain (default: 600)
- `--random-seed`: Random seed for reproducibility (default: 42)

## Output Format

The generated CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `domain` | Domain category: `legal`, `medical`, `finance`, `reading_comprehension` |
| `task_id` | Specific task identifier (e.g., `maud:t1`, `pubmedqa`, `finqa`, `mc500`) |
| `text` | Source text/context for the question |
| `question` | Question to be answered |
| `answers` | JSON array of multiple-choice options in format `[["A", "option1"], ["B", "option2"], ...]` |
| `answer` | Correct answer letter (A, B, C, or D) |

## Data Sources

### Legal Domain (MAUD)
- **Source**: [LegalBench](https://github.com/HazyResearch/legalbench)
- **Tasks**: 34 merger agreement analysis tasks
- **Format**: Multiple choice questions about legal contract provisions
- **License**: Check LegalBench repository for licensing terms

### Medical Domain (PubMedQA)  
- **Source**: [PubMedQA](https://pubmedqa.github.io/)
- **Format**: Yes/No/Maybe questions based on PubMed abstracts
- **License**: Check original dataset for licensing terms

### Financial Domain (FinQA)
- **Source**: [FinQA](https://github.com/czyssrs/FinQA)
- **Format**: Numerical reasoning questions with generated multiple-choice answers
- **Processing**: Automatic generation of plausible distractors for multiple-choice format
- **License**: Check FinQA repository for licensing terms

### Reading Comprehension (MCTest)
- **Source**: [MCTest](https://github.com/mcobzarenco/mctest)
- **Format**: Multiple-choice reading comprehension questions
- **Subset**: MC500 test set
- **License**: Check MCTest repository for licensing terms

## Technical Details

### Answer Generation (FinQA)

For financial questions, the script automatically generates multiple-choice options when not provided:

- **Numerical questions**: Creates plausible variations using percentage and absolute offsets
- **Yes/No questions**: Maps to binary choice format
- **Fallback**: Generates random variations within reasonable ranges

### Balancing Strategy

**Legal domain**: Samples evenly across all 34 MAUD tasks to ensure coverage of different legal reasoning types.

**Other domains**: Random sampling to reach target size while maintaining data quality.

## License

This repository contains code for dataset generation. Please check the individual data source repositories for their respective licensing terms:

- LegalBench: [License](https://github.com/HazyResearch/legalbench)
- PubMedQA: [License](https://pubmedqa.github.io/)  
- FinQA: [License](https://github.com/czyssrs/FinQA)
- MCTest: [License](https://github.com/mcobzarenco/mctest)
