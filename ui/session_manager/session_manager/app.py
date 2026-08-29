import json
import logging
import os
from pathlib import Path
from datetime import datetime as dt, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
from collections import deque
from dotenv import load_dotenv
import httpx
import yaml
from contextlib import asynccontextmanager
from importlib.metadata import version, PackageNotFoundError

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- FASTAPI ---
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# --- SESSION MANAGER API ---
from session_manager.api import router

# --- DATABASE ---
from session_manager.data import LocalDB, SQLDB, SQLParams, TemplateDB, LocalLogs, TemplateLogs, SQLLogs
from session_manager.session_manager import SessionManager


def get_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "0.0.0-dev"

load_dotenv()

# --- APP CONFIGURATION ---

TITLE = "persuasui-session-manager"
MODE = os.getenv("SESSION_MANAGER_MODE", "dev")

# Load environment variables
PERSUASIO_API_KEY = os.getenv("PERSUASIO_API_KEY")
if not PERSUASIO_API_KEY:
    raise ValueError("PERSUASIO_API_KEY environment variable is required")
PERSUASIO_BASE_URL = os.getenv("PERSUASIO_BASE_URL")
if not PERSUASIO_BASE_URL:
    raise ValueError("PERSUASIO_BASE_URL environment variable is required")
SESSION_MANAGER_API_KEY = os.getenv("SESSION_MANAGER_API_KEY")
if not SESSION_MANAGER_API_KEY:
    raise ValueError("SESSION_MANAGER_API_KEY environment variable is required")

SQL_PARAMS = SQLParams(
    dbname=os.getenv("PSQL_DB_NAME"),
    user=os.getenv("PSQL_USERNAME"),
    password=os.getenv("PSQL_PASSWORD"),
    host=os.getenv("PSQL_HOST"),
    port=os.getenv("PSQL_PORT"),
    sslmode=os.getenv("PSQL_SSLMODE")
)

if MODE=="production" and not SQL_PARAMS.exists:
    raise ValueError("PostgreSQL credentials must be set in environment variables for production mode")

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
LOG_DB_CONFIG = SQLParams(
    dbname=LOGS_PSQL_DB_NAME,
    user=LOGS_PSQL_USERNAME,
    password=LOGS_PSQL_PASSWORD,
    host=LOGS_PSQL_HOST,
    port=LOGS_PSQL_PORT,
    sslmode=LOGS_PSQL_SSLMODE
)


# Load yaml config (non-sensitive)
config_path = os.path.join(os.path.dirname(__file__), "config.yml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    CORS_CONFIG = config["cors"]
    
    
# --- LIFESPAN CONTEXT MANAGER ---

_db: Optional[TemplateDB] = None
_logs: Optional[TemplateLogs] = None
_session_manager: Optional[SessionManager] = None
    
def get_session_manager() -> SessionManager:
    """Returns the SessionManager created in lifespan"""
    global _session_manager
    if _session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return _session_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global _db, _session_manager
    
    if MODE=="production":
        logging.info("Starting Session Manager in PRODUCTION mode with SQLDB")
        _db = SQLDB(sql_params=SQL_PARAMS)
        _logs = SQLLogs(sql_params=LOG_DB_CONFIG)

        # Perform initial health check
        db_healthy, db_error = _db.check_connection_health()
        logs_healthy, logs_error = _logs.check_connection_health()

        if not db_healthy:
            logging.error(f"Database health check failed: {db_error}")
            raise RuntimeError(f"Database connection unhealthy: {db_error}")

        if not logs_healthy:
            logging.error(f"Logs database health check failed: {logs_error}")
            raise RuntimeError(f"Logs database connection unhealthy: {logs_error}")

        logging.info("Database health checks passed")
    else:
        logging.info("Starting Session Manager in DEVELOPMENT mode with LocalDB")
        _db = (LocalDB(data_dir=Path(__file__).parent.parent / "data"))
        _logs = LocalLogs(data_dir=Path(__file__).parent.parent / "data")

    _session_manager = SessionManager(
        db=_db,
        logs=_logs,
        persuasio_base_url=PERSUASIO_BASE_URL,
        persuasio_api_key=PERSUASIO_API_KEY
    )

    # Log successful startup with separate health confirmations
    if MODE=="production":
        _session_manager._log_activity(
            event_type="system",
            participant_id="SYSTEM",
            message="Main database connection established and healthy",
            severity="info"
        )
        _session_manager._log_activity(
            event_type="system",
            participant_id="SYSTEM",
            message="Logs database connection established and healthy",
            severity="info"
        )

    yield
    
    await _session_manager.persuasio_client.close()
    if isinstance(_db, SQLDB):
        _db.sql.close()
    if isinstance(_logs, SQLLogs):
        _logs.sql.close()

    logging.info("Session Manager shutdown complete")
        
    
# --- FASTAPI APP ---

app = FastAPI(
    title=TITLE, 
    version=get_package_version(TITLE),
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_CONFIG["allow_origins"],
    allow_credentials=CORS_CONFIG["allow_credentials"],
    allow_methods=CORS_CONFIG["allow_methods"],
    allow_headers=CORS_CONFIG["allow_headers"],
)

app.include_router(router)
