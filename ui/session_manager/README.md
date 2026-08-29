# Session Manager

Business logic and API layer for managing human–model debate sessions, built with [FastAPI](https://fastapi.tiangolo.com/).

## Architecture Overview

| File | Description |
|------|-------------|
| `app.py` | Main entry point --- loads configuration and launches the FastAPI application. |
| `api.py` | FastAPI endpoints for use by the [Client](../client). |
| `session_manager.py` | Core `SessionManager` class containing business logic for managing sessions, participants, and dialogues. |
| `persuasio_client.py` | HTTP client wrapper for the [Persuasio](../../persuasio) API. |
| `data.py` | Generic `TemplateDB` interface for session/participant storage, with implementations for local JSON (`LocalDB`) and PostgreSQL (`SQLDB`). |
| `config.yml` | Configuration file for FastAPI CORS settings. |

### Models

| File | Description |
|------|-------------|
| `models/api.py` | Pydantic `Request` and `Response` models for the API endpoints. |
| `models/entities.py` | Pydantic models for core entities such as `Session` and `Participant`. |
| `models/enums.py` | Enums for speaker and status values. |
| `models/persuasio.py` | Models for interacting with the Persuasio API. |
| `models/results.py` | Dataclasses for internal logic and results representation. |

## Setup

See the [root README](../README.md) for overall project setup. Instructions below apply to the session manager only.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Install dependencies and create the virtual environment:
   ```bash
   uv sync
   ```
3. Configure the required environment variables (see below).

### Environment Variables

| Name | Description | Required |
|------|-------------|----------|
| `SESSION_MANAGER_MODE` | Operating mode: `"dev"` or `"prod"` | No (defaults to `"dev"`) |
| `SESSION_MANAGER_API_KEY` | API key to authenticate incoming requests | Yes |
| `PERSUASIO_API_KEY` | API key for authenticating with the Persuasio service | Yes |
| `PERSUASIO_BASE_URL` | Base URL of the Persuasio service | Yes |
| `PSQL_USERNAME` | PostgreSQL database username | Prod only |
| `PSQL_PASSWORD` | PostgreSQL database password | Prod only |
| `PSQL_HOST` | PostgreSQL database host | Prod only |
| `PSQL_PORT` | PostgreSQL database port | Prod only |
| `PSQL_SSLMODE` | PostgreSQL SSL mode (`"disable"`, `"prefer"`, etc.) | Prod only |
| `PSQL_DB_NAME` | PostgreSQL database name | Prod only |

> [!IMPORTANT]
> All `PSQL_*` variables are required when `SESSION_MANAGER_MODE=prod`.

## Running

Start the FastAPI development server:

```bash
uv run fastapi dev session_manager/app.py --port 8000
```

Or using uvicorn directly:

```bash
uv run uvicorn session_manager.app:app --reload --port 8000
```

## License

This project is licensed under the [MIT License](../../LICENSE).