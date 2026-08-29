# Client

Frontend client for the human--model debate interface, built with [Gradio](https://www.gradio.app/).

## Architecture Overview

| File | Description |
|------|-------------|
| `app.py` | Main entry point --- loads configuration, instantiates the client, and launches the app. |
| `client.py` | `BasePersuasuiClient` --- abstract base class with framework-agnostic core logic. |
| `gradio_client.py` | `GradioPersuasuiClient` --- Gradio implementation of the base client. |
| `session_manager_client.py` | HTTP client for the [Session Manager](../session_manager) API. |
| `models.py` | Pydantic models for framework-agnostic internal data structures. |

## Extending the Client

The `BasePersuasuiClient` is designed to be portable. To create a UI with a different framework, inherit from it and implement:

- `create_interface(self)` --- Build the UI layout.
- `launch(self)` --- Start the UI application.

## Setup

See the [root README](../README.md) for overall project setup. Instructions below apply to the client only.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Install dependencies and create the virtual environment:
   ```bash copy
   uv sync
   ```
3. Configure the required environment variables (see below).

### Environment Variables

| Name | Description | Required |
|------|-------------|----------|
| `CLIENT_MODE` | Operating mode: `"dev"` or `"prod"` | No (defaults to `"dev"`) |
| `SESSION_MANAGER_API_KEY` | API key for authenticating with the Session Manager | Yes |
| `SESSION_MANAGER_BASE_URL` | Base URL of the Session Manager service | Yes |

## Running

```bash
uv run gradio client/app.py
```

Or using plain Python:

```bash
uv run python -m client.app
```

## License

This project is licensed under the [MIT License](../../LICENSE).