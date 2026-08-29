# Utterance Classification Evaluation

Scripts and resources for evaluating the performance of utterance prediction by Large Language Models. 

# Setup


1. Install [uv](https://github.com/astral-sh/uv) if not already installed
2. Install dependencies and create virtual environment:
   ```bash
   uv sync
   ```

## Available Commands

The following uv scripts are defined in `pyproject.toml`:

### `uv run model_utt_class`
Run utterance classification for each model that we are interested in.

```bash copy
uv run model_utt_class --experiments-file utterance_types_testset.csv --model-config models.yaml --batch-size 100
```

### `uv run bootstrap`
Generate experiment to run bootstrap tests. 

*Example run command:*

```bash copy
uv run bootstrap --experiments-file classification_results.csv --bootstrap-config bootstrap.yaml
```

Make sure to change the number of bootstrap datasets to the integer you require. We chose 10k samples.


## Files:

- `data_structures.py`: Pydantic data structures for utterance prediction evaluation.
- `data.py`: Methods for constructing sampled datasets (see `dataset.md` for details).
- `llm_client.py`: Wrapper for classifying utterances using OpenAI API.
- `evaluator.py`: Script for loading data, running classification, and evaluating results.

## Dataset

See `dataset.md` for details on the dataset used. 

## Evaluation Methodology (TO BE IMPLEMENTED)

- Pull max N samples for each utterance type. 
- For each model M:
    - For each utterance type U:
        - Make R classifications at temperature=0 with different random seeds.
        - Store aggregated results (i.e. majority decision for each sample) and raw results (i.e. all R decisions for each sample).
- Bootstrap evaluation:
    - Do 10,000 bootstrap samples, with size equal to the original sample size, sampling with replacement.
    - For each bootstrap sample:
        - Using aggregated results:
            - Compute confusion matrix for each model.
            - Compute per-class precision, recall, F1 for each model.
            - Compute overall precision, recall, macro-averaged F1 for each model.
- From bootstrap:
    - Compute mean and 95% confidence intervals for:
        - Overall precision, recall, macro-averaged F1 for each model.
        - Per-class precision, recall, F1 for each model.
