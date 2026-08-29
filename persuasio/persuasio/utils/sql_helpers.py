import json
from pydantic import BaseModel

from persuasio.utils.parsers import enum_encoder
from persuasio.datatypes.enums import SaveState

def create_session_data_sql_entry() -> str:

    update_string = f"""INSERT INTO session_data (id, status) VALUES (%s, %s);"""

    return update_string

def update_session_data_sql_entry() -> str:

    update_string = f"""INSERT INTO session_data (id, status) 
    VALUES (%s, %s)
    ON CONFLICT (id)
    DO UPDATE SET 
        status = EXCLUDED.status;"""

    return update_string

def save_state_to_db(session_state : dict, session_state_status : SaveState) -> str:

    state = session_state.copy()

    state["dialogue_history"] = [x.model_dump(mode="json") if isinstance(x,BaseModel) else x for x in state["dialogue_history"]]

    if (session_state_status == SaveState.FINAL_STATE):
        update_string = f"""INSERT INTO {session_state_status.value} (id, state) 
        VALUES (%s, %s);"""
    else:
        if "__interrupt__" in state:
            temp_state = state["__interrupt__"][0].value
            temp_state["dialogue_history"] = [x.model_dump(mode="json") if isinstance(x,BaseModel) else x for x in state["dialogue_history"]]

            # make sure interrupt state has all the data from original state
            for k, v in state.items():
                if (k not in temp_state.keys()) and (k != "__interrupt__"):
                    temp_state[k] = v

            state = temp_state.copy()

        update_string = f"""INSERT INTO {session_state_status.value} (id, state) 
        VALUES (%s, %s)
        ON CONFLICT (id)
        DO UPDATE SET 
            state = EXCLUDED.state;"""

    return update_string, json.dumps(state, default=enum_encoder)