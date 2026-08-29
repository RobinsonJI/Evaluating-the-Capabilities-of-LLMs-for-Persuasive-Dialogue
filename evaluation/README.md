# Evaluation

Scripts and data for evaluating the persuasiveness of LLM-generated dialogue in debates.

## Overview

This module provides a pipeline for assessing both **logical** and **subjective** persuasiveness of debates between humans and LLMs. The evaluation workflow is split across four Jupyter notebooks, designed to be run in sequence.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [1-saving-human-model-debates.ipynb](1-saving-human-model-debates.ipynb) | Extracts and saves model--model, human--model and human--human debate transcripts for evaluation. |
| 2 | [2-evaluating-logical-persuasiveness.ipynb](2-evaluating-logical-persuasiveness.ipynb) | Evaluates the logical persuasiveness of debate arguments (e.g. from debate adjudication via Persuasio). |
| 3 | [3-evaluating-subjective-persuasiveness.ipynb](3-evaluating-subjective-persuasiveness.ipynb) | Evaluates subjective persuasiveness using human annotations (e.g. from Prolific). |
| 4 | [4-comparing-logical-and-subjective-persuasiveness.ipynb](4-comparing-logical-and-subjective-persuasiveness.ipynb) | Compares logical and subjective persuasiveness scores using agreement and distance metrics. |

## Data

The [`data/`](data/) directory contains debate transcripts, annotation results, and demographic breakdowns of annotators:

- **`all-debates.csv`** / **`human-model-debates.csv`** / **`model2model-debates.csv`** --- Debate transcripts.
- **`subjective-persuasiveness-prolific-results.csv`** --- Subjective persuasiveness annotations collected via Prolific.
- **`prolific_demographic_export_*.csv`** --- Prolific participant demographics.
- **`self-declared-*-annotators.csv`** --- Annotator subsets by self-declared demographics (age, gender, political leaning).

## Setup

1. Install [uv](https://github.com/astral-sh/uv) if not already installed.
2. Install dependencies and create the virtual environment:
   ```bash copy
   uv sync
   ```
3. Run the notebooks in order (1 -> 4) using the created virtual environment as the Jupyter kernel.

## Usage

1. **Prepare debate transcripts** --- Run notebook 1 to extract and serialise debate transcripts into the `data/` directory.
2. **Logical evaluation** --- Run notebook 2 to compute logical persuasiveness scores.
3. **Subjective evaluation** --- Run notebook 3 to analyse human annotations collected from Prolific, including demographic breakdowns by age, gender, and political leaning.
4. **Comparison** --- Run notebook 4 to compare logical and subjective persuasiveness results.

## Requirements

- Python 3.12+ (see [`.python-version`](.python-version))
- Dependencies are managed via [`pyproject.toml`](pyproject.toml) and installed with `uv sync`.

## Project Structure

```
evaluation/
├── data/                          # Debate transcripts, annotations & demographics
│   ├── all-debates.csv
│   ├── human-model-debates.csv
│   ├── model2model-debates.csv
│   ├── subjective-persuasiveness-prolific-results.csv
│   ├── prolific_demographic_export_*.csv
│   └── self-declared-*-annotators.csv
├── 1-saving-human-model-debates.ipynb
├── 2-evaluating-logical-persuasiveness.ipynb
├── 3-evaluating-subjective-persuasiveness.ipynb
├── 4-comparing-logical-and-subjective-persuasiveness.ipynb
├── __init__.py
├── pyproject.toml
├── .python-version
└── README.md
```

## License

This project is licensed under the [MIT License](../LICENSE).