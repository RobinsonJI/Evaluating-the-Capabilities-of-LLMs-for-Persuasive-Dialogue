# Evaluating the Capabilities of LLMs for Persuasive Dialogue

This repository contains the code, data, and experimental infrastructure to accompany our paper (see [below](#citation) for the citation).

## Overview

This project investigated the persuasive capabilities of LLMs in structured dialogue settings. It includes:

- **Persuasio** — A formal persuasion dialogue system built on [Prakken's (2006)](https://www.cambridge.org/core/journals/knowledge-engineering-review/article/abs/formal-systems-for-persuasion-dialogue/6308104ED846C3B1A25996A9E0C8A5AD) framework, implemented as a multi-agent system using [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) with retrieval-augmented generation (RAG) over a Neo4j knowledge base.
- **Human--Model Debate UI** — A web-based platform for conducting and recording debates between human participants and LLM agents.
- **Model-to-Model Experiments** — An experiment runner for generating persuasion dialogues between pairs of language models.
- **Evaluation** — Scripts and notebooks for evaluating both logical and subjective persuasiveness of generated dialogues, including crowdsourced annotation via Prolific.
- **Utterance Classification Evaluation** — Tools for evaluating the accuracy of LLM-based utterance classification within the dialogue system.

## Repository Structure

```
.
├── persuasio/                          # Core dialogue system (FastAPI + LangGraph)
├── ui/                                 # Human--model debate platform
│   ├── client/                         #   Gradio-based frontend
│   └── session_manager/                #   FastAPI session management backend
├── model2model_experiments/            # Model-vs-model experiment runner
├── evaluation/                         # Persuasiveness evaluation scripts & data
├── utterance_classification_eval/      # Utterance classification evaluation
├── crowdsourced_debate_annotation/     # Prolific crowdsourcing tools & notebooks
├── human-model-debate-workshop-data/   # Workshop debate data
└── Code/                               # Legacy/shared code and knowledge base notebooks
```

### Component Details

| Directory | Description | README |
|-----------|-------------|--------|
| [`persuasio/`](persuasio/) | Formal persuasion dialogue system with RAG-augmented multi-agent architecture | [README](persuasio/README.md) |
| [`ui/`](ui/) | Human--model debate UI (Gradio client + FastAPI session manager) | [README](ui/README.md) |
| [`ui/client/`](ui/client/) | Gradio-based debate interface | [README](ui/client/README.md) |
| [`ui/session_manager/`](ui/session_manager/) | Session management API and business logic | [README](ui/session_manager/README.md) |
| [`model2model_experiments/`](model2model_experiments/) | Experiment runner for model-vs-model debates | [README](model2model_experiments/README.md) |
| [`evaluation/`](evaluation/) | Logical and subjective persuasiveness evaluation | [README](evaluation/README.md) |
| [`utterance_classification_eval/`](utterance_classification_eval/) | Evaluation of utterance classification accuracy | — |
| [`crowdsourced_debate_annotation/`](crowdsourced_debate_annotation/) | Prolific integration for crowdsourced debate annotation | — |

## Getting Started

### Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- **PostgreSQL 17.6** (for production deployments)
- **Neo4j Community Edition** (for RAG knowledge base — see [RAG setup guide](persuasio/persuasio/rag/README.md))
- **Git LFS** (for downloading the Neo4j knowledge base dump)

### Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<owner>/Evaluating-the-Capabilities-of-LLMs-for-Persuasive-Dialogue.git
   cd Evaluating-the-Capabilities-of-LLMs-for-Persuasive-Dialogue
   git lfs pull
   ```

2. **Set up Persuasio (core dialogue system):**

   ```bash
   cd persuasio
   uv sync
   ```

   Configure the required environment variables in `persuasio/.env` (see the [Persuasio README](persuasio/README.md) for details including LLM API keys, PostgreSQL, and Neo4j credentials).

   ```bash
   uv run uvicorn persuasio.app:app --reload
   ```

3. **Set up the Human--Model Debate UI:**

   ```bash
   cd ui
   uv sync
   ```

   Configure environment variables for both the client and session manager (see the [UI README](ui/README.md)).

   Run all services together using [Honcho](https://github.com/nickstenning/honcho):

   ```bash
   honcho start
   ```

   Or run individually:

   ```bash
   # Session Manager
   cd ui/session_manager
   uv run uvicorn session_manager.app:app --reload

   # Client
   cd ui/client
   uv run gradio client/app.py
   ```

4. **Run Model-to-Model Experiments:**

   ```bash
   cd model2model_experiments
   uv sync
   make_m2m        # Generate experiment configurations
   run_m2m         # Execute experiments
   ```

   See the [Model-to-Model README](model2model_experiments/README.md) for full usage instructions.

5. **Run Evaluation:**

   ```bash
   cd evaluation
   uv sync
   ```

   Open the Jupyter notebooks in order:
   - `1-saving-human-model-debates.ipynb`
   - `2-evaluating-logical-persuasiveness.ipynb`
   - `3-evaluating-subjective-persuasiveness.ipynb`
   - `4-comparing-logical-and-subjective-persuasiveness.ipynb`

### Neo4j Knowledge Base Setup

The RAG system requires a Neo4j knowledge base built on the argumentation graph from [Robinson et al. (2026)](https://arxiv.org/pdf/2602.18351). Three installation methods are supported:

1. **Docker** (recommended)
2. **Neo4j Desktop** (Windows, Mac, Linux)
3. **Linux Server** (Ubuntu Server 24.04 LTS)

See the [RAG README](persuasio/persuasio/rag/README.md) for detailed setup instructions.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────┐
│   Gradio    │────▶│ Session Manager  │────▶│  Persuasio │
│   Client    │◀────│    (FastAPI)     │◀────│  (FastAPI)  │
└─────────────┘     └──────────────────┘     └─────┬──────┘
                                                   │
                                            ┌──────┴──────┐
                                            │  LangGraph  │
                                            │ Multi-Agent │
                                            └──────┬──────┘
                                                   │
                                     ┌─────────────┼─────────────┐
                                     │             │             │
                                ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
                                │   LLM   │  │  Neo4j  │  │ PostgreSQL│
                                │   API   │  │  (RAG)  │  │  (State)  │
                                └─────────┘  └─────────┘  └──────────┘
```

- **Persuasio** implements Prakken's formal persuasion dialogue framework as a LangGraph multi-agent system, with sub-agents for utterance classification, argument generation, commitment tracking, and disambiguation.
- **The Session Manager** handles debate session lifecycle, participant management, and data persistence.
- **The Client** provides a Gradio-based web interface for human participants to engage in debates with LLM agents.

## Evaluation

The evaluation framework assesses persuasiveness along two dimensions:

- **Logical Persuasiveness** — Automated evaluation based on formal argumentation properties.
- **Subjective Persuasiveness** — Crowdsourced human judgments collected via [Prolific](https://www.prolific.com/), using pairwise comparison methodology.

Results and analysis are provided in the [evaluation notebooks](evaluation/).

## Citation

The citiation for this repository will be made available upon acceptance.

## License

This project is licensed under the [MIT License](LICENSE).