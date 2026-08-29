# Persuasio

Persuasio is a conversational, agentic microservice built on [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) for simulating, analysing, and testing persuasive dialogues.

Dialogues are implemented using [Prakken's (2006)](https://www.cambridge.org/core/journals/knowledge-engineering-review/article/abs/formal-systems-for-persuasion-dialogue/6308104ED846C3B1A25996A9E0C8A5AD) formal system for persuasion dialogue, enabling systematic experimentation with persuasion strategies, commitment tracking, and dialogue protocols.

At its core, Persuasio provides an API backend for debating LLMs within an argumentation-based setting. It is designed for researchers and developers exploring persuasion, argumentation, and dialogue systems to evaluate the persuasive capabilities of LLMs.

> [!NOTE]
> This software is distributed under the MIT License. It comes with ABSOLUTELY NO WARRANTY.

## Features

- Agent-based dialogue framework with LangGraph.
- Support for multiple LLMs, graph- and vector-based RAG ([Neo4j](https://neo4j.com/)) and SQL persistence ([PostgreSQL](https://www.postgresql.org/)).
- Configurable development and production modes.
- Modular design with agents, prompts, and dialogue graphs.
- Logging, visualisation, and export of dialogue sessions.

## Setup

1. Install [uv](https://github.com/astral-sh/uv) if not already installed.
2. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
3. Install dependencies and create the virtual environment:
   ```bash
   uv sync --extra cpu
   ```
   Alternatively, to run PyTorch on an NVIDIA GPU:
   ```bash
   uv sync --extra cu130
   ```

## Running the Server

Start the FastAPI development server:

```bash
uv run fastapi dev persuasio/app.py
```

Specify a custom port with the `--port` flag:

```bash
uv run fastapi dev persuasio/app.py --port 8080
```

### Development vs Production Mode

Set the `PERSUASIO_MODE` variable in your `.env` file to `dev` or `production`, or pass it inline:

```bash
PERSUASIO_MODE=production uv run fastapi dev persuasio/app.py
```

> [!IMPORTANT]
> Production mode requires a running **PostgreSQL** instance.

> [!IMPORTANT]
> Both production and dev modes require a running instance of the Neo4j knowledge base.

> [!NOTE]
> An example `.env` file can be found [here](.env.example)

On Unix-based systems, you may need to increase the file descriptor limit before running the production server:

```bash
ulimit -n 4096
```

## Database Setup

### PostgreSQL

PostgreSQL is required for production mode. This software uses PostgreSQL 17.6.

1. Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/).
2. Create a database (e.g. `persuasio`) --- see the [Neon PostgreSQL tutorial](https://neon.com/postgresql/tutorial) for a quickstart.
3. Configure the following variables in your `.env` file:
   - `PSQL_USERNAME`, `PSQL_PASSWORD`, `PSQL_HOST`, `PSQL_PORT`, `PSQL_SSLMODE`, `PSQL_DB_NAME`

### Neo4j (for RAG)

Persuasio supports retrieval-augmented generation (RAG) with [Neo4j Community Edition](https://neo4j.com/product/community-edition/).

See the [RAG README](persuasio/rag/README.md) for setup instructions.

## Project Structure

```
persuasio/
├── persuasio/
│   ├── app.py                      # FastAPI application entry point
│   ├── api.py                      # API route definitions
│   ├── agents/                     # Dialogue agents
│   │   ├── subgraph_calls.py
│   │   └── subgraph_agents/
│   │       ├── base_model.py
│   │       ├── commitments.py
│   │       ├── disambiguation.py
│   │       ├── generation_agents.py
│   │       ├── human_in_the_loop.py
│   │       ├── persuasiveness_choice.py
│   │       ├── typical_responses.py
│   │       └── utterance_classification.py
│   ├── config/                     # Configuration files
│   │   ├── config.yml
│   │   ├── logger.py
│   │   └── rag_locution_proposition_return_type.py
│   ├── datatypes/                  # API models & enums
│   │   ├── api.py
│   │   ├── enums.py
│   │   └── pydantic_basemodels.py
│   ├── graphs/                     # LangGraph definitions
│   │   ├── parent_graph.py
│   │   ├── sub_graphs/
│   │   │   ├── base_graph.py
│   │   │   ├── human_graph.py
│   │   │   ├── mas_graph.py
│   │   │   └── mas_rag_graph.py
│   │   └── figures/                # Graph visualisations
│   ├── models/                     # LLM & embedding configs
│   │   ├── models.py
│   │   └── sentence_transformers.py
│   ├── prompts/                    # Prompt templates
│   │   ├── generators/
│   │   └── system/
│   ├── rag/                        # Graph & vector RAG
│   ├── routers/                    # Conditional graph edges
│   ├── states/                     # LangGraph state definitions
│   ├── tools/                      # Post-processing tools
│   ├── utils/                      # Helpers & logging
│   └── outputs/                    # Dialogue logs & results
├── .env.example                    # Environment variable template
├── Dockerfile.persuasio            # Docker image for Persuasio
├── Dockerfile.neo4j                # Docker image for Neo4j
├── pyproject.toml
├── LICENSE
└── README.md
```

## References

- Prakken, H. (2006). *Formal systems for persuasion dialogue*. The Knowledge Engineering Review, 21(2), 163--188. [DOI](https://www.cambridge.org/core/journals/knowledge-engineering-review/article/abs/formal-systems-for-persuasion-dialogue/6308104ED846C3B1A25996A9E0C8A5AD)
- LangGraph --- [Documentation](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)

## License

This project is licensed under the [MIT License](LICENSE).
