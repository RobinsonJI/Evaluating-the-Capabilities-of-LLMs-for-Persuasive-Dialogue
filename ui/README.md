# UI for Human--Model Experiments

UI Platform for evaluating persuasive dialogue in LLMs. This is a nested workspace containing two microservices: a Gradio-based client UI and a FastAPI session manager.

## Structure

```
ui/                           
├── session_manager/                      # Session Manager Service
│   ├── data/                             # Local file storage (for dev)
│   │   ├── sessions.json                 # Session data for initialising in dev
│   │   ├── participants.json             # Participant data for initialising in dev
│   │   └── dialogues/                    # Dialogue logs
│   └── session_manager/
│   │   └── models/                       # Custom datatypes
│   │   │   ├── __init__.py
│   │   │   ├── api.py                    # _Request and Response models
│   │   │   ├── entities.py               # Entity models (Session, Participant, etc)
│   │   │   ├── enums.py                  # Various enums
│   │   │   ├── persuasio.py              # Models for matching persuasio API
│   │   │   └── results.py                # Dataclasses for internal logic
│   │   ├── __init__.py
│   │   ├── api.py                        # FastAPI routes (accessed by client)
│   │   ├── app.py                        # FastAPI app
│   │   ├── config.yml                    # Config for FastAPI CORS
│   │   ├── data.py                       # Storage interfaces for local/PSQL
│   │   ├── persuasio_client.py           # Wrapper for persuasio API
│   │   └── session_manager.py            # SessionManager class (core logic)
│   ├── pyproject.toml                    # Service-specific UV config
│   ├── Dockerfile                        # Dockerfile for containerisation
│   └── .env                              # Environment variables
│
├── client/                               # Frontend UI Service
│   ├── client/
│   │   └── assets/
│   │   │   └── style.css                 # Custom CSS for Gradio
│   │   ├── __init__.py
│   │   ├── app.py                        # App initialisation / entry point
│   │   ├── client.py                     # Generic client class (framework-agnostic)
│   │   ├── gradio_client.py              # Gradio-specific client implementation
│   │   ├── models.py                     # Pydantic models for internal data structures
│   │   └── session_manager_client.py     # Wrapper for session manager API
│   ├── pyproject.toml                    # Service-specific UV config
│   ├── Dockerfile                        # Dockerfile for containerisation
│   └── .env                              # Environment variables
│
├── pyproject.toml                        # Workspace root UV config
├── Procfile                              # File for honcho to run all services
├── docker-compose.yml                    # Docker Compose config for local testing
└── README.md                             
```

## Requirements

- Python 3.10+ with [uv](https://docs.astral.sh/uv/getting-started/installation/).
- A running instance of [Persuasio](../persuasio) (see the [Persuasio README](../persuasio/README.md) for setup).
- [Optional] [PostgreSQL](https://www.postgresql.org/download/) for production mode.
- [Optional] [Honcho](https://github.com/nickstenning/honcho) for running all services together locally.
- [Optional] [Docker](https://docs.docker.com/desktop/setup/install) for containerised deployment.

## Setup

A shared virtual environment with dependencies for both services can be created by running:

```bash
uv sync --all-packages
```

Each service also has its own `pyproject.toml`. All dependencies are stored in these files --- no dependencies are stored in the root `pyproject.toml`.

Separate virtual environments can also be created for each service if desired:

```bash
cd session_manager
uv sync
```

```bash
cd client
uv sync
```

### Environment Variables

Each service has its own `.env` file for environment variables needed to run. These are loaded at runtime. They can be changed in the individual `.env` files, set in the system environment, or overridden during execution (e.g. `SESSION_MANAGER_MODE=prod uv run ...`).

#### Session Manager Service

| Name | Description | Required |
|------|-------------|----------|
| `SESSION_MANAGER_MODE` | Operating mode: `"dev"` or `"prod"` | No (defaults to `"dev"`) |
| `SESSION_MANAGER_API_KEY` | API key to check incoming requests against | Yes |
| `PERSUASIO_API_KEY` | API key for authenticating with the Persuasio service | Yes |
| `PERSUASIO_BASE_URL` | Base URL of the Persuasio service | Yes |
| `PSQL_USERNAME` | PostgreSQL database username | Prod only |
| `PSQL_PASSWORD` | PostgreSQL database password | Prod only |
| `PSQL_HOST` | PostgreSQL database host | Prod only |
| `PSQL_PORT` | PostgreSQL database port | Prod only |
| `PSQL_SSLMODE` | PostgreSQL SSL mode (`"disable"`, `"prefer"`, etc.) | Prod only |
| `PSQL_DB_NAME` | PostgreSQL database name | Prod only |

#### Client Service

| Name | Description | Required |
|------|-------------|----------|
| `CLIENT_MODE` | Operating mode: `"dev"` or `"prod"` | No (defaults to `"dev"`) |
| `SESSION_MANAGER_API_KEY` | API key for authenticating with the Session Manager | Yes |
| `SESSION_MANAGER_BASE_URL` | Base URL of the Session Manager service | Yes |

## Running Services Individually

**Session Manager:**

```bash
cd session_manager
uv run fastapi dev session_manager/app.py --port 8000
```

**Client:**

```bash
cd client
uv run gradio client/app.py
```

## Running All Services with Honcho

[Honcho](https://github.com/nickstenning/honcho) can run the client, session manager, and Persuasio backend together using the included [`Procfile`](Procfile).

1. Install Honcho (if not already installed):
   ```bash
   uv tool install honcho
   ```

2. Start all services from the `ui/` directory:
   ```bash
   honcho start
   ```

This starts:

| Service | Default Port |
|---------|-------------|
| Persuasio | `8080` |
| Session Manager | `8000` |
| Client | `7860` |

Press `Ctrl+C` to stop all services.

## Docker Compose

All services can also be run together using **Docker Compose** for local testing. This uses the included [`docker-compose.yml`](docker-compose.yml) and the individual Dockerfiles.

```bash
docker-compose up --build
```

Press `Ctrl+C` to stop all services.

> [!WARNING]
> Docker Compose may require changing `psycopg2` to `psycopg2-binary` in the `pyproject.toml` files.

## License

This project is licensed under the [MIT License](../LICENSE).