from typing import List, Dict, Any
import psycopg2

import persuasio.app as app_module
from persuasio.datatypes.enums import Mode, SessionStatus

from persuasio.utils.logs import log_and_raise

def check_session_exists(mode : str):
    pass


def get_logs(session_id : str, func_name : str) -> List[Dict[str, Any]] | List[str]:

    """
    rtype: List[Dict[str, Any]] in production
    rtype: List[str] in dev
    """

    if (app_module.MODE == Mode.PROD.value) or (app_module.MODE == Mode.PROD):
        # Get logs from PostgreSQL db
        log_conn = psycopg2.connect(**app_module.LOG_DB_CONFIG)
        log_cur = log_conn.cursor()
        log_cur.execute(
            "SELECT * FROM persuasio WHERE session_id = %s",
            (session_id,)
        )
        logs = log_cur.fetchall()
        log_conn.close()

        if logs is None:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=func_name.__name__,
                message=f"You tried to access the log for Session {session_id} but that session has not been created yet.",
                mode=app_module.MODE
            )

        logs = [
            {
                "index" : x[0], 
                "timestamp" : x[1].isoformat(), 
                "log_level" : x[2],
                "session_id" : x[3], 
                "callable" : x[4], 
                "status_code" : x[5], 
                "message" : x[6], 
                "context" : x[7], 
            } 
            for x in logs]
        
    else:
        if session_id not in app_module.session_db:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=func_name.__name__,
                message=f"You tried to access the log for Session {session_id} but that session has not been created yet.",
                mode=app_module.MODE
            )

        # Read log file (same for both modes)
        with open(f"persuasio/outputs/logs/{session_id}.log", "r") as file:
            log_lines = file.readlines()

        logs = [line.strip() for line in log_lines]

    return logs



def get_state(session_id : str, func_name) -> Dict[str, str]:

    if (app_module.MODE == Mode.PROD.value) or (app_module.MODE == Mode.PROD):
        # Production: Query from runtime_states table
        app_module.cur.execute(
            "SELECT * FROM runtime_states WHERE id = %s",
            (session_id,)
        )
        db_response = app_module.cur.fetchone()
        if db_response is None:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=func_name.__name__,
                message=f"You tried to access the the LangGraph state of Session {session_id} but that session has not been created yet.",
                mode=app_module.MODE
            )
        state = db_response[1]
    else:
        # Development: Get from in-memory dict
        if session_id not in app_module.session_db:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=func_name.__name__,
                message=f"You tried to access the the LangGraph state of Session {session_id} but that session has not been created yet.",
                mode=app_module.MODE
            )

        state = app_module.session_db[session_id]["state"]

    return state


def get_ongoing() -> List[str]:

    if app_module.MODE == "production":
        # Production: Query PostgreSQL
        app_module.cur.execute(
            "SELECT * FROM session_data WHERE status = %s OR status = %s",
            (SessionStatus.RUNNING.value, SessionStatus.STARTED.value)
        )
        db_response = app_module.cur.fetchall()

        ongoing = [id for (id, _status_) in db_response]

    else:
        ongoing = [
            session_id for session_id, value in app_module.session_db.items()
            if (value["status"] == SessionStatus.RUNNING) or (value["status"] == SessionStatus.STARTED)
        ]

    return ongoing


def get_terminated() -> List[str]:
    if app_module.MODE == "production":
        # Production: get session data from PostgreSQL
        app_module.cur.execute(
            "SELECT * FROM terminated_states"
        )
        db_response = app_module.cur.fetchall()

        terminated = [id for (id, state) in db_response]

    else:
        # Development: Get from in-memory dict
        terminated = [
                session_id for session_id, value in app_module.session_db.get("terminated_states", {}).items()
            ]
        
    return terminated


def get_finished() -> List[str]:

    if app_module.MODE == "production":
        # Production: Query PostgreSQL
        app_module.cur.execute(
            "SELECT * FROM session_data WHERE status = %s",
            (SessionStatus.FINISHED.value,)
        )
        db_response = app_module.cur.fetchall()

        finished = [id for (id, _status_) in db_response]
    else:
        finished =[
            session_id for session_id, value in app_module.session_db.items()
            if (value.get("status") == SessionStatus.FINISHED)
            ]
        
    return finished


def get_ended() -> List[str]:

    finished = get_finished()
    terminated = get_terminated()

    return list(set(finished).union(terminated))