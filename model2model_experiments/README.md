# Model-to-Model Experiments

Experiment runner for generating and executing persuasion dialogues between pairs of language models using the [Persuasio](../persuasio) backend.

## Setup

1. Install [uv](https://github.com/astral-sh/uv) if not already installed.
2. Install dependencies and create the virtual environment:
   ```bash copy
   uv sync
   ```

## Usage

Three commands are available as uv scripts (defined in `pyproject.toml`):

| Command | Description |
|---------|-------------|
| `uv run make_m2m` | Generate experiment configurations -- creates matchups between models and speaker types. |
| `uv run count_m2m` | Print statistics on the number of generated experiment configurations. |
| `uv run run_m2m` | Execute the configured model-to-model dialogue sessions via the Persuasio API. |

Run them in the order above: generate configs -> verify counts -> execute experiments.

## Project Structure

```
model2model_experiments/
├── model2model/                # Main package
│   ├── run_experiments.py      # Experiment execution engine
│   ├── make_experiments.py     # Experiment configuration generator
│   └── data/                   # Created during execution
│       ├── config/             # Experiment configuration files
│       ├── dialogues/          # Generated dialogue transcripts
│       ├── experiments/        # Experiment definitions
│       └── logs/               # Execution logs
├── pyproject.toml
└── README.md
```

## Dependencies

This package depends on the local [`persuasio`](../persuasio) package, which provides the core dialogue system and API backend.

> [!NOTE]
>

## License

This project is licensed under the [MIT License](../LICENSE).