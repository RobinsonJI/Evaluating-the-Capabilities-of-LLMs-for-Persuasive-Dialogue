# --- ENV VARIABLES ---
from dotenv import load_dotenv
import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

TITLE = "persuasio"

# # .env path
# env_file = Path(__file__)
# print(env_file)
# Load envr
load_dotenv()

# Read mode (dev or production) from environment variable
MODE = os.getenv("PERSUASIO_MODE", "dev")

# Load secrets from enviroment
API_KEY = os.getenv("PERSUASIO_API_KEY")
if not API_KEY:
    raise ValueError("PERSUASIO_API_KEY environment variable is required")

# State DB env variables
PSQL_USERNAME = os.getenv("PSQL_USERNAME")
PSQL_PASSWORD = os.getenv("PSQL_PASSWORD")
PSQL_HOST = os.getenv("PSQL_HOST")
PSQL_PORT = os.getenv("PSQL_PORT")
PSQL_SSLMODE = os.getenv("PSQL_SSLMODE")
PSQL_DB_NAME = os.getenv("PSQL_DB_NAME")
if MODE=="production" and any(cred is None for cred in [PSQL_USERNAME, PSQL_PASSWORD, PSQL_HOST, PSQL_PORT, PSQL_DB_NAME]):
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
LOG_DB_CONFIG = {
                'dbname': LOGS_PSQL_DB_NAME,
                'user': LOGS_PSQL_USERNAME,
                'password': LOGS_PSQL_PASSWORD,
                'host': LOGS_PSQL_HOST,
                'port': LOGS_PSQL_PORT,
                "sslmode" : LOGS_PSQL_SSLMODE
            }

# Neo4j env variables (for graph- and vector-based RAG)
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# Load yaml config (non-sensitive)
config_path = os.path.join(os.path.dirname(__file__), "config", "config.yml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    CORS_CONFIG = config["cors"]



# =================== IMPORTS ===================

import asyncio
import sys
from importlib.metadata import version, PackageNotFoundError

if sys.platform == "win32":
    # Use SelectorEventLoop for Windows compatibility with psycopg
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- FASTAPI ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- MEMORY MANAGEMENT / SQL  ---
from contextlib import asynccontextmanager
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import psycopg2

# --- PERSUASIO GRAPH ---
from persuasio.graphs.parent_graph import workflow
from persuasio.utils.graph_compiler import compile_parent_graph
from persuasio.utils.draw_langgraphs import draw_graph

# --- PERSUASIO API ---
from persuasio.api import router

# --- HELPER FUNCTIONS ---
def get_package_version(package_name: str) -> str:
    """Get package version from installed metadata"""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "0.0.0-dev"
    

# init global variables
compiled_graph = None
checkpointer = None
session_db = {}
cur = None
db_connection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global compiled_graph, checkpointer, session_db, cur, db_connection

    # Startup
    if MODE == "production":
        # --- PRODUCTION MODE: Use PostgreSQL ---
        logging.info("Starting Persuasio in PRODUCTION mode...")

        checkpoint_uri = f"postgres://{PSQL_USERNAME}:{PSQL_PASSWORD}@{PSQL_HOST}:{PSQL_PORT}/{PSQL_DB_NAME}?sslmode={PSQL_SSLMODE}"

        logging.info("Connecting to AsyncPostgresSaver...")
        async with AsyncPostgresSaver.from_conn_string(checkpoint_uri) as checkpointer:

            logging.info("Setting up checkpointer...")
            await checkpointer.setup()

            logging.info("Compiling parent graph...")
            compiled_graph = compile_parent_graph(workflow=workflow, checkpointer=checkpointer)

            logging.info("Drawing graph...")
            draw_graph(graph=compiled_graph, graph_name="parent_graph")

            # --- LangGraph thread_id and state storage ---
            # Connect to PostgreSQL database
            logging.info(f"Connecting to PSQL(State) at {PSQL_HOST}:{PSQL_PORT}...")
            db_connection = psycopg2.connect(
                dbname=PSQL_DB_NAME,
                user=PSQL_USERNAME,
                password=PSQL_PASSWORD,
                host=PSQL_HOST,
                port=PSQL_PORT
            )
            cur = db_connection.cursor()

            # Create tables if they don't exist
            logging.info("Creating PSQL(State) tables...")
            cur.execute("""CREATE TABLE IF NOT EXISTS session_data (
                        id VARCHAR PRIMARY KEY,
                        status VARCHAR
                        );
            """)
            cur.execute("""CREATE TABLE IF NOT EXISTS runtime_states (
                        id VARCHAR PRIMARY KEY,
                        state JSONB
                        );
            """)
            cur.execute("""CREATE TABLE IF NOT EXISTS final_states (
                        id VARCHAR PRIMARY KEY,
                        state JSONB
                        );
            """)
            cur.execute("""CREATE TABLE IF NOT EXISTS terminated_states (
                        id VARCHAR PRIMARY KEY,
                        state JSONB
                        );
            """)
            db_connection.commit()

            # --- Logging storage ---
            # Connect to PostgreSQL database
            logging.info(f"Connecting to PSQL(Logs) at {LOGS_PSQL_HOST}:{LOGS_PSQL_PORT}...")
            log_connection = psycopg2.connect(**LOG_DB_CONFIG)
            log_cur = log_connection.cursor()
            # Create table called "persuasio" to log our system on sql
            logging.info("Creating PSQL(Logs) table if not exists...")
            log_cur.execute("""CREATE TABLE IF NOT EXISTS persuasio (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                level VARCHAR,
                session_id VARCHAR,
                service VARCHAR,
                status_code CHAR(3),
                message TEXT,
                context JSONB);"""
            )
            log_connection.commit()
            log_cur.close()

            logging.info("Production mode initialised with PostgreSQL!")

            # Application is running
            yield

            # Shutdown
            logging.info("Shutdown initiated")
            cur.close()
            db_connection.close()

            logging.info("PostgreSQL connection closed")

    else:
        # --- DEVELOPMENT MODE: Use in-memory storage ---
        logging.info("Starting Persuasio in DEVELOPMENT mode...")

        session_db = {}
        checkpointer = InMemorySaver()
        logging.info("Compiling parent graph...")
        compiled_graph = compile_parent_graph(workflow=workflow, checkpointer=checkpointer)
        logging.info("Drawing graph...")
        draw_graph(graph=compiled_graph, graph_name="parent_graph")

        logging.info("Development mode initialized with in-memory storage")

        # Application is running
        yield

        # Shutdown
        logging.info("Development mode shutting down")


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
