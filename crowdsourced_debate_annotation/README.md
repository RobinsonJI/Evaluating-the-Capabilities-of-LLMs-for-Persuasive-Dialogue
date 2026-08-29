# Crowdsourced Debate Annotation

This directory contains the full pipeline for crowdsourcing subjective persuasiveness annotations of debate transcripts via [Prolific](https://www.prolific.com/). Annotators compare pairs of debate transcripts and judge which speaker presented the more persuasive arguments.

## Installation

## Pipeline Overview

The annotation pipeline is split across a series of Jupyter notebooks, intended to be run in order:

### 1. Data Preparation

- **`1-saving-human-model-debates.ipynb`** --- Loads and saves human-vs-model and model-vs-model debate transcripts into structured CSV files (`human-model-debates.csv`, `model2model-debates.csv`).
- **`2-sampling-and-costing-comparisons.ipynb`** --- Samples pairwise comparisons from the debate data and estimates the cost of running the annotation study on Prolific.

### 2. Prolific Study Setup & Processing

- **`3-downloading-prolific-pilot-results.ipynb`** --- Downloads and inspects results from the initial pilot study on Prolific.
- **`3.1-processing-dataset-to-add-attention-checks-and-custom-group-tasks.ipynb`** --- Processes the annotation dataset to inject attention check items and organise tasks into custom groups for Prolific deployment.
- **`3.2-checking-final-results.ipynb`** --- Downloads final study results, checks attention check pass/fail rates, removes rejected annotators, approves valid submissions, and exports cleaned results.
- **`3.3-krippendorffs-alpha.ipynb`** --- Computes Krippendorff's alpha for inter-annotator agreement on both persuasiveness judgements (nominal) and confidence ratings (nominal).

### 3. Demographic Analysis

- **`4-subjective-persuasiveness-across-demographic-data.ipynb`** --- Analyses how persuasiveness judgements vary across annotator demographics (age, gender, political leaning) using the Prolific demographic export.

## Key Files

| File | Description |
|------|-------------|
| `prolific.py` | Lightweight Python client wrapper for the Prolific API. |
| `annotation-guidelines.md` | Annotation guidelines provided to crowdworkers. |
| `pyproject.toml` | Python project/dependency configuration. |
| `.python-version` | Specifies the Python version for the project. |

## Data

All data files are stored in `data/`. Key files include:

| File | Description |
|------|-------------|
| `all-debates.csv` | Combined dataset of all debate transcripts. |
| `human-model-debates.csv` | Debate transcripts between human and model speakers. |
| `model2model-debates.csv` | Debate transcripts between two model speakers. |
| `019c5684-e8e6-737c-9442-07692f856379.csv` | Raw annotation results exported from Prolific. |
| `prolific_demographic_export_*.csv` | Annotator demographic data from Prolific. |
| `subjective-persuasiveness-prolific-results.csv` | Final cleaned persuasiveness results with win counts and confidence scores. |
| `final_full_3_annotations_per_human_with_2_atten_checks.csv` | Final dataset with 3 annotations per comparison and 2 attention checks. |
| `self-declared-*-annotators.csv` | Demographic subgroup splits (age, gender, political leaning). |
| `failed_tests/` | Data from earlier experimental runs that were rejected or superseded. |

## Annotation Design

- **Task**: Annotators are shown two debate transcripts side-by-side and asked:
  1. *Which Speaker A presented the more persuasive arguments?* (3 options: Speaker A from Debate 1 / Both equally persuasive / Speaker A from Debate 2)
  2. *How confident are you in your choice?* (5-point Likert scale: Very unsure -> Very confident)
- **Replication**: Each comparison receives 3 independent annotations.
- **Quality control**: Attention check items with known correct answers are embedded in the task set. Annotators who fail multiple attention checks are flagged and their submissions rejected.
- **Inter-annotator agreement**: Measured using Krippendorff's alpha at the nominal level.

## Requirements

Dependencies are managed via `pyproject.toml` and installed with `uv sync`. The project requires a Prolific API key set as the environment variable `prolific_key`.
