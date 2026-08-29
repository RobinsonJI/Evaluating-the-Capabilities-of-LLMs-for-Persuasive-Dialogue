from fastapi import HTTPException
import logging
import functools
import time
from datetime import datetime
import psycopg2
from psycopg2.extras import Json

from persuasio.datatypes.enums import Mode, LogLevels

import persuasio.app as app_module



def log(session_id : str, level : LogLevels, service: str, message: str, mode : Mode, context = None) -> None:
    """
    Log an informational message to the appropriate logging destination.
    
    Logs the message to either a PostgreSQL database (in production mode) or to the 
    standard Python logger (in non-production mode). Unlike log_and_raise, this 
    function only logs and does not raise an exception.
    
    Args:
        session_id (str): Unique identifier for the session. Used both as a database
            field in production mode and as the logger name in non-production mode.
        service (str): Name of the service generating the log entry. Used for 
            categorization in the database log.
        message (str): Informational message to be logged.
        mode (Mode): Operating mode determining logging behavior. When set to 
            Mode.PROD.value or Mode.PROD, logs to PostgreSQL database; otherwise 
            logs via Python's logging module.
        context (dict, optional): Additional contextual information about the log entry.
            Stored as JSON in the database. Defaults to None (empty dict in database).
    
    Returns:
        None
    
    Notes:
        - In production mode, requires app_module.LOG_DB_CONFIG to be defined with PostgreSQL 
          connection parameters.
        - Database logs are stored in the 'persuasio' table with level set to 'INFO'
          and status_code set to 200.
        - Timestamp is automatically generated at the time of logging.
        - In non-production mode, logs at WARNING level despite being an info log
          (consider using logger.info() instead for consistency).
    """
    if (mode == Mode.PROD.value) or (mode == Mode.PROD):
        conn = psycopg2.connect(**app_module.LOG_DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO persuasio (timestamp, level, session_id, service, status_code, message, context)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            level.value.upper(),
            session_id,
            service,
            200 if level == LogLevels.INFO else 500,
            message,
            Json(context or {})
        ))
        conn.commit()
        cur.close()
    else:
        logger = logging.getLogger(session_id)
        if (level == LogLevels.ERROR) or (level == LogLevels.ERROR.value):
            logger.error(message)
        if (level == LogLevels.WARN) or (level == LogLevels.WARN.value):
            logger.warning(message)
        else:
            logger.info(message)


def log_and_raise(session_id : str, status_code: int, service: str, message: str, mode : Mode, context = None) -> HTTPException:
    """
    Log a warning message and raise an HTTP exception.
    
    Logs the error to either a PostgreSQL database (in production mode) or to the 
    standard Python logger (in non-production mode), then raises an HTTPException 
    with the provided status code and message.
    
    Args:
        session_id (str): Unique identifier for the session. Used as the logger name
            in non-production mode.
        status_code (int): HTTP status code to use when raising the exception.
        service (str): Name of the service where the error occurred. Used for 
            categorization in the database log.
        message (str): Error message describing what went wrong. Used both for 
            logging and as the HTTPException detail.
        mode (Mode): Operating mode determining logging behavior. When set to 
            Mode.PROD.value, logs to PostgreSQL database; otherwise logs via 
            Python's logging module.
        context (dict, optional): Additional contextual information about the error.
            Stored as JSON in the database. Defaults to None (empty dict in database).
    
    Raises:
        HTTPException: Always raised after logging, with the provided status_code 
            and message as the detail.
    
    Notes:
        - In production mode, requires app_module.LOG_DB_CONFIG to be defined with PostgreSQL 
          connection parameters.
        - Database logs are stored in the 'persuasio' table with level set to 'WARN'.
        - Timestamp is automatically generated at the time of logging.
    """
    if (mode == Mode.PROD.value) or (mode == Mode.PROD):
        conn = psycopg2.connect(**app_module.LOG_DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO persuasio (timestamp, level, session_id, service, status_code, message, context)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "WARN",
            session_id,
            service,
            status_code,
            message,
            Json(context or {})
        ))
        conn.commit()
        cur.close()
    else:
        logger = logging.getLogger(session_id)
        logger.warning(message)
        

    raise HTTPException(status_code, detail=message)






# def log_class(cls):
#     """
#     Class decorator that adds logging to all methods of a class.
#     Logs method entry, arguments, and return values.
#     """

#     for attr_name, attr_value in cls.__dict__.items():
#         if callable(attr_value) and not attr_name.startswith("__"):           
#             @functools.wraps(attr_value)
#             def wrapper(self, *args, _method=attr_value, _name=attr_name, **kwargs):
#                 if self.state["mode"] == "production":
#                     pass
#                 else:
#                     logger = logging.getLogger(self.state["session_id"])
#                     logger.info(f"ENTER: '{self.__class__.__name__}' calling '{_name}'; CURRENT SPEAKER: {self.state['speaker']}, DIALOGUE TURN NUM = {len(self.state['dialogue_history'])}.")
#                     start_time = time.time()
#                     result = _method(self, *args, **kwargs)
#                     duration = time.time() - start_time
#                     logger.info(f"EXIT: '{self.__class__.__name__}' calling '{_name}' ({duration} secs).")
#                 return result

#             setattr(cls, attr_name, wrapper)

#     return cls

def log_class(cls):
    """
    Class decorator that adds logging to all methods of a class.
    Logs method entry, arguments, and return values.
    """

    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith("__"):           
            @functools.wraps(attr_value)
            def wrapper(self, *args, _method=attr_value, _name=attr_name, **kwargs):
                if (self.state["mode"] == Mode.PROD.value) or (self.state["mode"] == Mode.PROD):
                    conn = psycopg2.connect(**app_module.LOG_DB_CONFIG)
                    cur = conn.cursor()

                    # Log entry into class
                    cur.execute("""
                        INSERT INTO persuasio (timestamp, level, session_id, service, status_code, message, context)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "INFO",
                        self.state["session_id"],
                        self.__class__.__name__ + "." +  _name,
                        200,
                        f"ENTER: '{self.__class__.__name__}' calling '{_name}'; CURRENT SPEAKER: {self.state['speaker']}, DIALOGUE TURN NUM = {len(self.state['dialogue_history'])}.",
                        Json({})
                    ))
                    conn.commit()

                    # Start timer for class run time
                    start_time = time.time()
                    # Run class methods
                    result = _method(self, *args, **kwargs)
                    # Compute duration
                    duration = time.time() - start_time

                    # Log exit of class with times
                    cur.execute("""
                        INSERT INTO persuasio (timestamp, level, session_id, service, status_code, message, context)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "INFO",
                        self.state["session_id"],
                        self.__class__.__name__ + "." +  _name,
                        200,
                        f"EXIT: '{self.__class__.__name__}' calling '{_name}' ({duration} secs).",
                        Json({"duration" : duration})
                    ))
                    conn.commit()

                    cur.close()
                    
                else:
                    logger = logging.getLogger(self.state["session_id"])
                    logger.info(f"ENTER: '{self.__class__.__name__}' calling '{_name}'; CURRENT SPEAKER: {self.state['speaker']}, DIALOGUE TURN NUM = {len(self.state['dialogue_history'])}.")
                    start_time = time.time()
                    result = _method(self, *args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(f"EXIT: '{self.__class__.__name__}' calling '{_name}' ({duration} secs).")
                return result

            setattr(cls, attr_name, wrapper)

    return cls


# def log_function(func):
#     """
#     Function decorator that logs entry, arguments, and return values.
#     """
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         if args and hasattr(args[0], "session_id"):
#             state = args[0]
#             print(state)
#             logger = logging.getLogger(state["session_id"])  
#             logger.info(f"ENTER: Calling '{func.__name__}'; CURRENT SPEAKER: {state.get('speaker', state[state['current_speaker'].value])}, DIALOGUE TURN NUM = {len(state['dialogue_history'])}.")
#             start_time = time.time()
#             result = func(*args, **kwargs)
#             duration = time.time() - start_time
#             logger.info(f"EXIT: '{func.__name__}' returned {result!r} ({duration:.4f} secs).")
#         else:
#             result = func(*args, **kwargs)
#         return result
#     return wrapper

def log_function(func):
    """
    Function decorator that logs entry, arguments, and return values.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args and hasattr(args[0], "session_id"):
            state = args[0]
            if (state["mode"] == Mode.PROD.value) or (state["mode"] == Mode.PROD):
                conn = psycopg2.connect(**app_module.LOG_DB_CONFIG)
                cur = conn.cursor()

                # Log entry into class
                cur.execute("""
                    INSERT INTO persuasio (timestamp, level, session_id, service, status_code, message, context)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "INFO",
                    state["session_id"],
                    func.__name__,
                    200,
                    f"ENTER: Calling '{func.__name__}'; CURRENT SPEAKER: {state.get('speaker', state[state['current_speaker'].value])}, DIALOGUE TURN NUM = {len(state['dialogue_history'])}.",
                    Json({})
                ))
                conn.commit()

                # Start func timer
                start_time = time.time()
                # run func
                result = func(*args, **kwargs)
                # Compute run time duration
                duration = time.time() - start_time

                # Log exit from class
                cur.execute("""
                    INSERT INTO persuasio (timestamp, level, session_id, service, status_code, message, context)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "INFO",
                    state["session_id"],
                    func.__name__,
                    200,
                    f"EXIT: '{func.__name__}' returned {result!r} ({duration:.4f} secs).",
                    Json({"duration" : duration})
                ))
                conn.commit()
                cur.close()

            else:
                logger = logging.getLogger(state["session_id"])  
                logger.info(f"ENTER: Calling '{func.__name__}'; CURRENT SPEAKER: {state.get('speaker', state[state['current_speaker'].value])}, DIALOGUE TURN NUM = {len(state['dialogue_history'])}.")
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"EXIT: '{func.__name__}' returned {result!r} ({duration:.4f} secs).")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper
