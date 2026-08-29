import gradio as gr
from dotenv import load_dotenv
import os
import logging
from pathlib import Path

from client.logs import LocalLogs, SQLLogs, SQLParams

# --- LOAD ENVIRONMENT VARIABLES ---

load_dotenv()

# --- SETUP LOGGING ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

MODE = os.getenv("CLIENT_MODE", "dev").lower()
FRAMEWORK = os.getenv("FRAMEWORK", "gradio").lower()

SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")
UI_PORT = int(os.getenv("UI_PORT", "7860"))
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "2"))
AI_RESPONSE_DELAY = int(os.getenv("AI_RESPONSE_DELAY", "15"))  # Simulated delay for AI responses

SESSION_MANAGER_API_KEY = os.getenv("SESSION_MANAGER_API_KEY")
if not SESSION_MANAGER_API_KEY:
    raise ValueError("SESSION_MANAGER_API_KEY environment variable is required")
SESSION_MANAGER_BASE_URL = os.getenv("SESSION_MANAGER_BASE_URL")
if not SESSION_MANAGER_BASE_URL:
    raise ValueError("SESSION_MANAGER_BASE_URL environment variable is required")

# Log DB env variables
LOGS_PSQL_USERNAME = os.getenv("LOGS_PSQL_USERNAME")
LOGS_PSQL_PASSWORD = os.getenv("LOGS_PSQL_PASSWORD")
LOGS_PSQL_HOST = os.getenv("LOGS_PSQL_HOST")
LOGS_PSQL_PORT = os.getenv("LOGS_PSQL_PORT")
LOGS_PSQL_SSLMODE = os.getenv("LOGS_PSQL_SSLMODE")
LOGS_PSQL_DB_NAME = os.getenv("LOGS_PSQL_DB_NAME")
if MODE=="production" and any(cred is None for cred in [LOGS_PSQL_USERNAME, LOGS_PSQL_PASSWORD, LOGS_PSQL_HOST, LOGS_PSQL_PORT, LOGS_PSQL_SSLMODE, LOGS_PSQL_DB_NAME]):
    raise ValueError("PostgreSQL credentials must be set in environment variables for production mode")
# Create log config dict
LOG_DB_CONFIG = {
    'dbname': LOGS_PSQL_DB_NAME,
    'user': LOGS_PSQL_USERNAME,
    'password': LOGS_PSQL_PASSWORD,
    'host': LOGS_PSQL_HOST,
    'port': LOGS_PSQL_PORT,
    "sslmode" : LOGS_PSQL_SSLMODE
}

# --- SETUP LOGS ---

if MODE == "production":
    logs = SQLLogs(SQLParams(**LOG_DB_CONFIG))

    # Perform health check on startup
    logs_healthy, logs_error = logs.check_connection_health()

    if not logs_healthy:
        logging.error(f"Logs database health check failed: {logs_error}")
        raise RuntimeError(f"Logs database connection unhealthy: {logs_error}")

    logging.info("Logs database health check passed")
else:
    logs = LocalLogs(data_dir=Path(__file__).parent.parent / "data")

# --- APP ---
match FRAMEWORK:
    case "gradio":
        from client.gradio_client import GradioPersuasuiClient as framework_client
    case _:
        raise ValueError(f"Unsupported UI framework: {FRAMEWORK}")

app = framework_client(
    mode=MODE,
    refresh_interval=REFRESH_INTERVAL,
    server_name=SERVER_NAME,
    server_port=UI_PORT,
    api_base_url=SESSION_MANAGER_BASE_URL,
    api_key=SESSION_MANAGER_API_KEY,
    logs=logs,
    ai_response_delay=AI_RESPONSE_DELAY
)

app.launch()