import os
from pathlib import Path
from abc import ABC, abstractmethod
from pydantic import BaseModel
import json
from typing import List

import psycopg2 as pg
from functools import wraps

from session_manager.models import Session, Participant, DialogueTurn, ActivityEvent


class SQLParams(BaseModel):
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
    
class TemplateDB(ABC):
    """
    Template interface for data management
    load_sessions
    load_participants
    update_session
    load_dialogue
    save_dialogue
    """
    
    @abstractmethod
    def load_sessions_raw(self) -> List[dict]:
        pass    

    @abstractmethod
    def load_participants_raw(self) -> List[dict]:
        pass
    
    @abstractmethod
    def update_session(self, session: Session):
        pass

    @abstractmethod
    def update_participant(self, participant: Participant):
        pass
    
    @abstractmethod
    def load_dialogue_raw(self, session_id: str) -> List[dict]:
        pass
    
    @abstractmethod
    def save_dialogue(self, session_id: str, dialogue: List[DialogueTurn]):
        pass

    def _parse_sessions(self, sessions: list) -> List[Session]:
        parsed = []
        for session_data in sessions:
            if isinstance(session_data, str):
                # Data is a JSON string from SQLDB
                parsed.append(Session.model_validate_json(session_data))
            elif isinstance(session_data, dict):
                # Data is a dict from LocalDB
                parsed.append(Session(**session_data))
        return parsed

    def load_sessions(self) -> List[Session]:
        return self._parse_sessions(self.load_sessions_raw())
    
    def _parse_participants(self, participants: list) -> List[Participant]:
        parsed = []
        for participant_data in participants:
            if isinstance(participant_data, str):
                # Data is a JSON string from SQLDB
                parsed.append(Participant.model_validate_json(participant_data))
            elif isinstance(participant_data, dict):
                # Data is a dict from LocalDB
                parsed.append(Participant(**participant_data))
        return parsed
    
    def load_participants(self) -> List[Participant]:
        return self._parse_participants(self.load_participants_raw())

    def _parse_dialogue(self, dialogue: list) -> List[DialogueTurn]:
        parsed = []
        for turn_data in dialogue:
            if isinstance(turn_data, str):
                # Data is a JSON string from SQLDB
                parsed.append(DialogueTurn.model_validate_json(turn_data))
            elif isinstance(turn_data, dict):
                # Data is a dict from LocalDB
                parsed.append(DialogueTurn(**turn_data))
        return parsed
    
    def load_dialogue(self, session_id: str) -> List[DialogueTurn]:
        raw_dialogue = self.load_dialogue_raw(session_id)
        return self._parse_dialogue(raw_dialogue)
    

# --- LOCAL DATA IMPLEMENTATION ---

class LocalDB(TemplateDB):
    """
    Local data management implementation
    participant and session info stored in json files in data/
    save dialogue history to data/
    """
    
    def __init__(self, data_dir: Path = Path("./data")):
        self.data_dir = data_dir
        
        # check if data directory exists and files exist
        if any([
            not os.path.exists(self.data_dir),
            not os.path.exists(os.path.join(self.data_dir, "sessions.json")),
            not os.path.exists(os.path.join(self.data_dir, "participants.json"))
        ]):
            raise FileNotFoundError(f"Data directory or required files not found: {self.data_dir.absolute()}")
        
        # make dialogues dir if it doesn't exist
        self.dialogues_dir = data_dir / "dialogues"
        if not self.dialogues_dir.exists():
            self.dialogues_dir.mkdir(parents=True)
    
    def load_sessions_raw(self) -> List[dict]:
        """Read and return list of sessions from sessions.json"""
        sessions_file = self.data_dir / "sessions.json"
        with open(sessions_file, "r") as f:
            return json.load(f)
    
    def load_participants_raw(self) -> List[dict]:
        """Read and return list of participants from participants.json"""
        participants_file = self.data_dir / "participants.json"
        with open(participants_file, "r") as f:
            return json.load(f)

    def update_session(self, session: Session):
        """Update session info in sessions.json by replacing or appending."""
        try:
            sessions = self.load_sessions()
        except (FileNotFoundError, json.JSONDecodeError):
            sessions = []

        session_found = False
        for i, s in enumerate(sessions):
            if s.session_id == session.session_id:
                sessions[i] = session
                session_found = True
                break
        
        if not session_found:
            sessions.append(session)

        sessions_file = self.data_dir / "sessions.json"
        with open(sessions_file, "w") as f:
            json.dump([s.model_dump(mode='json') for s in sessions], f, indent=2)

    def update_participant(self, participant: Participant):
        """Update participant info in participants.json by replacing or appending."""
        try:
            participants = self.load_participants()
        except (FileNotFoundError, json.JSONDecodeError):
            participants = []

        participant_found = False
        for i, p in enumerate(participants):
            if p.participant_id == participant.participant_id:
                participants[i] = participant
                participant_found = True
                break
        
        if not participant_found:
            participants.append(participant)

        participants_file = self.data_dir / "participants.json"
        with open(participants_file, "w") as f:
            json.dump([p.model_dump(mode='json') for p in participants], f, indent=2)

    def load_dialogue_raw(self, session_id) -> List[dict]:
        """Load dialogue history for a session from data/dialogues/{session_id}.json"""
        dialogue_file = self.dialogues_dir / f"{session_id}.json"
        if not dialogue_file.exists():
            return []
        with open(dialogue_file, "r") as f:
            return json.load(f)
        
        with open(dialogue_file, "r") as f:
            data = json.load(f)
            
        dialogue = [DialogueTurn(**turn) for turn in data]
        return dialogue
    
    def save_dialogue(self, session_id: str, dialogue: List[DialogueTurn]):
        """Save dialogue history for a session to data/dialogues/{session_id}.json"""
        dialogue_file = self.dialogues_dir / f"{session_id}.json"
        with open(dialogue_file, "w") as f:
            json.dump([turn.model_dump(mode="json") for turn in dialogue], f, indent=2)

# --- SQL DATA IMPLEMENTATION ---

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

class SQLDB(TemplateDB):
    """
    SQL data management implementation
    participant and session info stored in a SQL database
    save dialogue history to a SQL database
    """

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
        Create the necessary tables (sessions, participants, dialogues) if they do not already exist.
        The schema uses a single JSONB column to store the Pydantic model data.
        """
        cursor = self.sql.cursor()
        
        # Session table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_data JSONB
            );
        """)
        
        # Participant table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                participant_data JSONB
            );
        """)

        # Dialogue history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dialogues (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                turn_data JSONB
            );
        """)
        cursor.close()

    @reconnect_on_failure
    def load_sessions_raw(self) -> List[str]:
        """Load session JSON blobs from the SQL database."""
        cursor = self.sql.cursor()
        cursor.execute("SELECT session_data FROM sessions")
        rows = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return rows

    @reconnect_on_failure
    def load_participants_raw(self) -> List[str]:
        """Load participant JSON blobs from the SQL database."""
        cursor = self.sql.cursor()
        cursor.execute("SELECT participant_data FROM participants")
        rows = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return rows

    @reconnect_on_failure
    def update_session(self, session: Session):
        """Update/insert session info in the SQL database."""
        cursor = self.sql.cursor()
        session_json = session.model_dump_json()
        sql = """
            INSERT INTO sessions (session_id, session_data) VALUES (%s, %s)
            ON CONFLICT (session_id) DO UPDATE SET session_data = EXCLUDED.session_data;
        """
        cursor.execute(sql, (session.session_id, session_json))
        cursor.close()

    @reconnect_on_failure
    def update_participant(self, participant: Participant):
        """Update/insert participant info in the SQL database."""
        cursor = self.sql.cursor()
        participant_json = participant.model_dump_json()
        sql = """
            INSERT INTO participants (participant_id, participant_data) VALUES (%s, %s)
            ON CONFLICT (participant_id) DO UPDATE SET participant_data = EXCLUDED.participant_data;
        """
        cursor.execute(sql, (participant.participant_id, participant_json))
        cursor.close()

    @reconnect_on_failure
    def load_dialogue_raw(self, session_id: str) -> List[str]:
        """Load dialogue history for a session from the SQL database"""
        cursor = self.sql.cursor()
        cursor.execute("SELECT turn_data FROM dialogues WHERE session_id = %s", (session_id,))
        rows = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return rows

    @reconnect_on_failure
    def save_dialogue(self, session_id: str, dialogue: list[DialogueTurn]):
        """Save dialogue history for a session to the SQL database.

        This implementation deletes all existing turns for the session and inserts the new ones,
        mirroring the overwrite behavior of the LocalDB implementation.
        """
        cursor = self.sql.cursor()
        cursor.execute("DELETE FROM dialogues WHERE session_id = %s", (session_id,))
        if not dialogue:
            cursor.close()
            return
        
        data = [
            (session_id, turn.model_dump_json())
            for turn in dialogue
        ]
        cursor.executemany("INSERT INTO dialogues (session_id, turn_data) VALUES (%s, %s)", data)
        cursor.close()

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


# --- LOGS ---

class TemplateLogs(ABC):
    """
    Template interface for logs management
    save_log
    load_log
    """
    
    @abstractmethod
    def save_log(self, event: ActivityEvent):
        pass

    @abstractmethod
    def load_log(self) -> List[ActivityEvent]:
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
    def __init__(self, data_dir: Path = Path("./data")):
        self.data_dir = data_dir
        
        # check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir.absolute()}")
        
        # make logs dir if it doesn't exist
        self.logs_dir = data_dir / "logs"
        if not self.logs_dir.exists():
            self.logs_dir.mkdir(parents=True)
            
    def save_log(self, event: ActivityEvent):
        """Save a log entry to logs/logs.jsonl (append single line)"""
        logs_file = self.logs_dir / "logs.jsonl"

        # Convert event to dict and serialize timestamp to ISO format
        event_dict = event.model_dump(mode='json')

        # Append as a single JSON line
        with open(logs_file, "a") as f:
            json.dump(event_dict, f)
            f.write("\n")
            
    def load_log(self) -> List[ActivityEvent]:
        """Load all log entries from logs/logs.jsonl"""
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
        Create the session_manager table to store ActivityEvent data.
        """
        cursor = self.sql.cursor()

        # Activity logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_manager (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                message TEXT NOT NULL,
                session_id TEXT
            );
        """)
        cursor.close()
    
        
    @reconnect_on_failure
    def save_log(self, event: ActivityEvent):
        """Save a log entry to the SQL database."""
        cursor = self.sql.cursor()
        sql = """
            INSERT INTO session_manager (timestamp, event_type, severity, participant_id, message, session_id)
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
        cursor.close()
        
    @reconnect_on_failure
    def load_log(self) -> List[ActivityEvent]:
        """Load all log entries from the SQL database."""
        cursor = self.sql.cursor()
        cursor.execute("""
            SELECT timestamp, event_type, severity, participant_id, message, session_id
            FROM session_manager
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
        cursor.close()

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