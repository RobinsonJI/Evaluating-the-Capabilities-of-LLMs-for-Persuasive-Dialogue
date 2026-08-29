"""
Activity logging infrastructure for the client.

Provides local (JSONL) and SQL (PostgreSQL) logging implementations.
"""

import os
import json
from pathlib import Path
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Optional
import psycopg2 as pg
from functools import wraps


class SQLParams(BaseModel):
    """PostgreSQL connection parameters"""
    dbname: str = None
    user: str = None
    password: str = None
    host: str = None
    port: str = None
    sslmode: str = None

    @property
    def exists(self):
        return all([
            self.dbname is not None,
            self.user is not None,
            self.password is not None,
            self.host is not None,
            self.port is not None
        ])


def reconnect_on_failure(method):
    """Decorator to reconnect and retry once if a connection error occurs"""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (pg.OperationalError, pg.InterfaceError):
            # Only reconnect on connection-related errors
            self._init_sql_client()
            return method(self, *args, **kwargs)
        # Any other exception just raises immediately
    return wrapper


class TemplateLogs(ABC):
    """
    Template interface for logs management
    save_log
    load_log
    """

    @abstractmethod
    def save_log(self, event):
        pass

    @abstractmethod
    def load_log(self) -> List:
        pass

    def render_logs(self) -> List[str]:
        """Render logs as formatted strings"""
        events = self.load_log()
        rendered = []
        for event in events:
            timestamp_str = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            session_info = f" [session={event.session_id}]" if event.session_id else ""
            rendered.append(
                f"[{timestamp_str}] [{event.severity.upper()}] [{event.event_type}] {event.participant_id}: {event.message}{session_info}"
            )
        return rendered


class LocalLogs(TemplateLogs):
    """Local JSONL file logging implementation"""

    def __init__(self, data_dir: Path = Path("./data")):
        self.data_dir = data_dir

        # check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir.absolute()}")

        # make logs dir if it doesn't exist
        self.logs_dir = data_dir / "logs"
        if not self.logs_dir.exists():
            self.logs_dir.mkdir(parents=True)

    def save_log(self, event):
        """Save a log entry to logs/logs.jsonl (append single line)"""
        logs_file = self.logs_dir / "logs.jsonl"

        # Convert event to dict and serialize timestamp to ISO format
        event_dict = event.model_dump(mode='json')

        # Append as a single JSON line
        with open(logs_file, "a") as f:
            json.dump(event_dict, f)
            f.write("\n")

    def load_log(self) -> List:
        """Load all log entries from logs/logs.jsonl"""
        from client.models import ActivityEvent

        logs_file = self.logs_dir / "logs.jsonl"
        if not logs_file.exists():
            return []

        events = []
        with open(logs_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event_dict = json.loads(line)
                # Pydantic will handle datetime deserialization automatically
                events.append(ActivityEvent(**event_dict))

        return events


class SQLLogs(TemplateLogs):
    """PostgreSQL logging implementation"""

    def __init__(self, sql_params: SQLParams):
        self.sql_params = sql_params
        self._init_sql_client()
        self._create_tables()

    def _init_sql_client(self):
        """Initialize SQL client connection"""
        self.sql = pg.connect(**self.sql_params.model_dump(mode='json'))
        self.sql.autocommit = True

    @reconnect_on_failure
    def _create_tables(self):
        """
        Create the client table to store ActivityEvent data.
        """
        cursor = self.sql.cursor()

        # Activity logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                message TEXT NOT NULL,
                session_id TEXT
            );
        """)

    @reconnect_on_failure
    def save_log(self, event):
        """Save a log entry to the SQL database."""
        cursor = self.sql.cursor()
        sql = """
            INSERT INTO client (timestamp, event_type, severity, participant_id, message, session_id)
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        cursor.execute(sql, (
            event.timestamp,
            event.event_type,
            event.severity,
            event.participant_id,
            event.message,
            event.session_id
        ))

    @reconnect_on_failure
    def load_log(self) -> List:
        """Load all log entries from the SQL database."""
        from client.models import ActivityEvent

        cursor = self.sql.cursor()
        cursor.execute("""
            SELECT timestamp, event_type, severity, participant_id, message, session_id
            FROM client
            ORDER BY timestamp
        """)

        events = []
        for row in cursor.fetchall():
            events.append(ActivityEvent(
                timestamp=row[0],
                event_type=row[1],
                severity=row[2],
                participant_id=row[3],
                message=row[4],
                session_id=row[5]
            ))

        return events

    def check_connection_health(self):
        """
        Perform a simple health check on the PostgreSQL connection.

        Returns:
            Tuple of (is_healthy: bool, error_message: Optional[str])
        """
        try:
            cursor = self.sql.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()

            if result and result[0] == 1:
                return True, None
            else:
                return False, "Unexpected result from health check query"

        except pg.OperationalError as e:
            return False, f"Connection error: {str(e)}"
        except pg.InterfaceError as e:
            return False, f"Interface error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {type(e).__name__}: {str(e)}"
