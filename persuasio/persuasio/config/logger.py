import os

def logging_config(session_id):

    LOG_DIR = "persuasio/outputs/logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, f"{session_id.replace(":","_")}.log")

    LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] %(message)s"
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": LOG_FILE,
            "formatter": "standard",
            "level": "INFO",   
        },
    },
    "loggers": {
        session_id: {  # root logger
            "handlers": ["file"],
            "level": "INFO",   
            "propagate": False
        },
    },
}
    return LOGGING_CONFIG

