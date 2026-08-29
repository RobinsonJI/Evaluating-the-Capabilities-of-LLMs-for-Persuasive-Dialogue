import logging
import logging.config
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader

from langgraph.types import Command

# --- PERSUASIO DATATYPES ---
from persuasio.states.state import ParentState
from persuasio.datatypes.enums import SessionStatus, SpeakerOrder, SpeakerType, PoliticalPositionEnsembleOrModelName, SaveState, LogLevels
from persuasio.datatypes.api import ClientResponse, SessionParameters, PersuasioResponse

# --- PERSUASIO METHODS ---
from persuasio.utils.parsers import parse_state_to_persuasio_basemodel_object
from persuasio.utils.graph_compiler import get_compiled_graph
from persuasio.utils.api_dependencies import session_parameters, response
from persuasio.config.logger import logging_config
from persuasio.utils.logs import log_and_raise, log
from persuasio.utils.sql_helpers import (
    create_session_data_sql_entry,
    update_session_data_sql_entry,
    save_state_to_db,
)

from persuasio.utils.db_helpers import get_logs, get_state, get_ongoing, get_ended, get_finished, get_terminated

import persuasio.app as app_module

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key from the request header"""
    if api_key != app_module.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# --- HEALTH CHECK ENDPOINT ---
@router.get("/")
@router.get("/health")
async def health_check():
    """Health check endpoint for Azure Container Apps probes.

    Returns 200 OK if the application is running and ready to serve requests.
    """
    return {"status": "healthy"}

# --- CREATE A LANGGRAPH THREAD (I.E. A SESSION FOR OUR USE-CASE) ---
@router.get("/sessions/create", dependencies=[Depends(verify_api_key)])
async def create_session(
    response: ClientResponse = Depends(response),
    params: SessionParameters = Depends(session_parameters)
) -> PersuasioResponse:
    """
    Create a new Persuasio debate session and execute the initial LangGraph step.

    Behavior differs depending on environment mode:
    - **Production Mode**: Uses PostgreSQL for all session lookups and storage.
   - **Development Mode**: Uses in-memory session storage.

    Workflow
    --------
    1. Validate session ID uniqueness.
    2. Validate speaker metadata and constraints.
    3. Initialise the full `ParentState` dictionary.
    4. Persist the new session.
    5. Run the LangGraph parent graph for the first time.
    6. Save updated runtime state and determine whether:
         - The dialogue FINISHED,
         - The dialogue TERMINATED,
         - Or the system is WAITING for human input.

    Parameters
    ----------
    response : ClientResponse
        Initial user utterance or empty (depending on speaker type).
    params : SessionParameters
        Configuration for participants, models, topic, limits, etc.

    Returns
    -------
    PersuasioResponse
        The state after the initial LangGraph invocation, formatted for API output.

    Raises
    ------
    HTTPException
        If the session already exists in an incompatible state or parameters are invalid.
    """

    # --- MODE-AWARE SESSION EXISTENCE CHECK ---
    if app_module.MODE == "production":
        log(
            session_id=params.session_id,
            level=LogLevels.INFO,
            service=create_session.__name__,
            message=f"Trying to create session {params.session_id}...",
            mode=app_module.MODE
        )

        # Production: Check PostgreSQL database
        app_module.cur.execute(
            "SELECT * FROM session_data WHERE id = %s",
            (params.session_id,)
        )
        db_response = app_module.cur.fetchone()

        if db_response is not None:
            id = db_response[0]
            status = db_response[1]

            if params.session_id == id:
                if status == SessionStatus.FINISHED.value:
                    log_and_raise(
                        session_id=params.session_id,
                        status_code=400,
                        service=create_session.__name__,
                        message=f"Session {params.session_id} has finished.",
                        mode=app_module.MODE
                    )
                elif status == SessionStatus.TERMINATED.value:
                    log_and_raise(
                        session_id=params.session_id,
                        status_code=400,
                        service=create_session.__name__,
                        message=f"Session {params.session_id} has been terminated.",
                        mode=app_module.MODE
                    )
                else:
                    log_and_raise(
                        session_id=params.session_id,
                        status_code=400,
                        service=create_session.__name__,
                        message=f"Session {params.session_id} is ongoing",
                        mode=app_module.MODE
                    )

        session_exists = db_response is not None

    else:
        # Create logging file for session ID / LangGraph thread
        logger_config = logging_config(params.session_id)
        logging.config.dictConfig(logger_config)
        logger = logging.getLogger(params.session_id)
        log(
            session_id=params.session_id,
            level=LogLevels.INFO,
            service=create_session.__name__,
            message=f"Trying to create session {params.session_id}...",
            mode=app_module.MODE
        )

        # Development: Check in-memory dictionary
        if params.session_id in app_module.session_db:
            if app_module.session_db[params.session_id]["status"] == SessionStatus.FINISHED:
                log_and_raise(
                    session_id=params.session_id,
                    status_code=400,
                    service=create_session.__name__,
                    message=f"Session {params.session_id} has finished.",
                    mode=app_module.MODE
                )
            elif app_module.session_db[params.session_id]["status"] == SessionStatus.TERMINATED:
                log_and_raise(
                    session_id=params.session_id,
                    status_code=400,
                    service=create_session.__name__,
                    message=f"Session {params.session_id} has been terminated.",
                    mode=app_module.MODE
                )
            else:
                log_and_raise(
                    session_id=params.session_id,
                    status_code=400,
                    service=create_session.__name__,
                    message=f"Session {params.session_id} is ongoing",
                    mode=app_module.MODE
                )

        session_exists = params.session_id in app_module.session_db

    # --- CREATE NEW SESSION ---
    if not session_exists:
        # Validate parameters
        if (len(params.first_speaker) == 0) or (len(params.second_speaker) == 0):
            log_and_raise(
                session_id=params.session_id,
                status_code=400,
                service=create_session.__name__,
                message=f"'first_speaker' and 'second_speaker' must be included within the GET request",
                mode=app_module.MODE
            )
        if (params.first_speaker_type == SpeakerType.MAS_RAG) and (
            params.first_speaker_knowledge_base_ensemble_or_model_name == PoliticalPositionEnsembleOrModelName.NO_USE_OF_KNOWLEDGE_BASE
        ):
            log_and_raise(
                session_id=params.session_id,
                status_code=400,
                service=create_session.__name__,
                message=f"As the 'first_speaker_type'={SpeakerType.MAS_RAG}, you must set the 'first_speaker_knowledge_base_ensemble_or_model_name' equal to one of the following: {[x.value for x in PoliticalPositionEnsembleOrModelName if x.value != '']}",
                mode=app_module.MODE
            )
        if (params.second_speaker_type == SpeakerType.MAS_RAG) and (
            params.second_speaker_knowledge_base_ensemble_or_model_name == PoliticalPositionEnsembleOrModelName.NO_USE_OF_KNOWLEDGE_BASE
        ):
            log_and_raise(
                session_id=params.session_id,
                status_code=400,
                service=create_session.__name__,
                message=f"As the 'second_speaker_type'={SpeakerType.MAS_RAG}, you must set the 'second_speaker_knowledge_base_ensemble_or_model_name' equal to one of the following: {[x.value for x in PoliticalPositionEnsembleOrModelName if x.value != '']}",
                mode=app_module.MODE
            )

        # Create an initial state
        state: ParentState = {
            "session_id": params.session_id,
            "debate_topic": params.debate_topic,
            "max_dialogue_turns": params.max_dialogue_turns,
            "max_sentences_per_turn": params.max_sentences_per_turn,

            "first_speaker": params.first_speaker,
            "first_speaker_type": params.first_speaker_type,
            "first_speaker_model_name": params.first_speaker_model_name,
            "first_speaker_model_temp": params.first_speaker_model_temp,
            "first_speaker_model_top_p": params.first_speaker_model_top_p,
            "first_speaker_model_seed": params.first_speaker_model_seed,
            "first_speaker_utterance": response.utterance.strip() if (params.first_speaker_type == SpeakerType.HUMAN) else "",
            "first_speaker_utterance_with_corresponding_types": [],
            "first_speaker_commitments": [],
            "first_speaker_original_claim": "",
            "first_speaker_political_position_range": params.first_speaker_political_political_position_range,
            "first_speaker_political_position_std": params.first_speaker_political_position_std,
            "first_speaker_political_position_prob_of_na": params.first_speaker_political_position_prob_of_na,
            "first_speaker_knowledge_base_ensemble_or_model_name": params.first_speaker_knowledge_base_ensemble_or_model_name,
            "first_speaker_number_of_vector_based_rag_examples": params.first_speaker_number_of_vector_based_rag_examples,
            "first_speaker_number_of_graph_rag_examples": params.first_speaker_number_of_graph_rag_examples,
            "first_speaker_graph_rag_examples": [],

            "second_speaker": params.second_speaker,
            "second_speaker_type": params.second_speaker_type,
            "second_speaker_model_name": params.second_speaker_model_name,
            "second_speaker_model_temp": params.second_speaker_model_temp,
            "second_speaker_model_top_p": params.second_speaker_model_top_p,
            "second_speaker_model_seed": params.second_speaker_model_seed,
            "second_speaker_utterance": "",
            "second_speaker_utterance_with_corresponding_types": [],
            "second_speaker_commitments": [],
            "second_speaker_original_claim": "",
            "second_speaker_political_position_range": params.second_speaker_political_position_range,
            "second_speaker_political_position_std": params.second_speaker_political_position_std,
            "second_speaker_political_position_prob_of_na": params.second_speaker_political_position_prob_of_na,
            "second_speaker_knowledge_base_ensemble_or_model_name": params.second_speaker_knowledge_base_ensemble_or_model_name,
            "second_speaker_number_of_vector_based_rag_examples": params.second_speaker_number_of_vector_based_rag_examples,
            "second_speaker_number_of_graph_rag_examples": params.second_speaker_number_of_graph_rag_examples,
            "second_speaker_graph_rag_examples": [],

            "current_speaker": SpeakerOrder.FIRST_SPEAKER,
            "next_speaker": SpeakerOrder.SECOND_SPEAKER,

            "dialogue_history": [],
            "winner": "",
            "loser": "",
            "reason_for_dialogue_termination": {},

            "human_model_name": params.human_model_name,
            "human_model_temp": params.human_model_temp,
            "human_model_top_p": params.human_model_top_p,
            "human_model_seed": params.human_model_seed,

            "utterance_classification_approach": params.utterance_classification_approach,
            "utterance_classification_number_of_classifications": params.utterance_classification_number_of_classifications,

            "mode" : app_module.MODE
        }

        # --- MODE-AWARE SESSION STORAGE ---
        if app_module.MODE == "production":
            # Production: Store in PostgreSQL
            app_module.cur.execute(
                create_session_data_sql_entry(),
                (params.session_id, SessionStatus.STARTED.value)
            )
            update_query, update_state = save_state_to_db(session_state=state, session_state_status=SaveState.RUN_STATE)
            app_module.cur.execute(
                update_query,
                (params.session_id, update_state)
            )
            app_module.db_connection.commit()
        else:
            # Development: Store in memory
            app_module.session_db[params.session_id] = {
                "status": SessionStatus.STARTED,
                "state": state
            }

        # Create a thread in LangGraph for the session
        config = {"configurable": {"thread_id": params.session_id}, "recursion_limit": 1000}

        # Run the graph
        log(
            session_id=params.session_id,
            level=LogLevels.INFO,
            service=create_session.__name__,
            message="DIALOGUE START: Parent graph invoked, waiting for Langgraph return state...",
            mode=app_module.MODE
        )

        graph = get_compiled_graph(compiled_graph=app_module.compiled_graph)
        state = await graph.ainvoke(state, config=config)
        log(
            session_id=params.session_id,
            level=LogLevels.INFO,
            service=create_session.__name__,
            message="Parent graph returned.",
            mode=app_module.MODE
        )

        # Update runtime state
        if app_module.MODE == "production":
            update_query, update_state = save_state_to_db(session_state=state, session_state_status=SaveState.RUN_STATE)
            app_module.cur.execute(
                update_query,
                (params.session_id, update_state)
            )
            app_module.db_connection.commit()

            # Check to see if terminated
            app_module.cur.execute(
                "SELECT * FROM terminated_states WHERE id = %s",
                (params.session_id,)
            )
            db_response = app_module.cur.fetchone()
            # Save updated terminated state
            if db_response:
                update_query, update_state = save_state_to_db(session_state=state, session_state_status=SaveState.TERMINATED_STATES)
                app_module.cur.execute(
                    update_query,
                    (params.session_id, update_state)
                )
                app_module.cur.execute(
                        update_session_data_sql_entry(),
                        (params.session_id, SessionStatus.TERMINATED.value)
                    )
                app_module.db_connection.commit()
                return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.TERMINATED)
        else:
            if params.session_id in app_module.session_db.get("terminated_states", {}):
                    app_module.session_db[params.session_id] = {
                        "status": SessionStatus.TERMINATED,
                        "state": state
                    }
                    return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.TERMINATED)
            else:
                app_module.session_db[params.session_id] = {
                    "status": SessionStatus.RUNNING,
                    "state": state
                }

        # Check if dialogue finished
        if (len(state["winner"]) > 0) and (len(state["loser"]) > 0) and (len(state["reason_for_dialogue_termination"]) > 0):
            # Dialogue finished
            if app_module.MODE == "production":
                app_module.cur.execute(
                    update_session_data_sql_entry(),
                    (params.session_id, SessionStatus.FINISHED.value)
                )
                update_query, update_state = save_state_to_db(session_state=state, session_state_status=SaveState.FINAL_STATE)
                app_module.cur.execute(
                    update_query,
                    (params.session_id, update_state)
                )
                app_module.db_connection.commit()

                # Check to see if terminated
                app_module.cur.execute(
                    "SELECT * FROM terminated_states WHERE id = %s",
                    (params.session_id,)
                )
                db_response = app_module.cur.fetchone()
                # Save updated terminated state
                if db_response:
                    update_query, update_state = save_state_to_db(session_state=state, session_state_status=SaveState.TERMINATED_STATES)
                    app_module.cur.execute(
                        update_query,
                        (params.session_id, update_state)
                    )
                    app_module.cur.execute(
                        update_session_data_sql_entry(),
                        (params.session_id, SessionStatus.TERMINATED.value)
                    )
                    app_module.db_connection.commit()
                    return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.TERMINATED)
            else:
                if params.session_id in app_module.session_db.get("terminated_states", {}):
                    app_module.session_db[params.session_id] = {
                        "status": SessionStatus.TERMINATED,
                        "state": state
                    }
                    return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.TERMINATED)
                else:
                    app_module.session_db[params.session_id] = {
                        "status": SessionStatus.FINISHED,
                        "state": state
                    }

            log(
                session_id=params.session_id,
                level=LogLevels.INFO,
                service=create_session.__name__,
                message=f"WINNER: {state['winner']}; LOSER: {state['loser']}, REASON: {state['reason_for_dialogue_termination']}",
                mode=app_module.MODE
            )
            log(
                session_id=params.session_id,
                level=LogLevels.INFO,
                service=create_session.__name__,
                message=f"DIALOGUE END: Session finished.",
                mode=app_module.MODE
            )
            

            return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.FINISHED)

        if (params.first_speaker_type == SpeakerType.HUMAN) or (params.second_speaker_type == SpeakerType.HUMAN):
            # Waiting for human response
            if app_module.MODE == "production":
                app_module.cur.execute(
                    update_session_data_sql_entry(),
                    (params.session_id, SessionStatus.RUNNING.value)
                )
                app_module.db_connection.commit()

                # Check to see if terminated
                app_module.cur.execute(
                    "SELECT * FROM terminated_states WHERE id = %s",
                    (params.session_id,)
                )
                db_response = app_module.cur.fetchone()
                # Save updated terminated state
                if db_response:
                    update_query, update_state = save_state_to_db(session_state=state, session_state_status=SaveState.TERMINATED_STATES)
                    app_module.cur.execute(
                        update_query,
                        (params.session_id, update_state)
                    )
                    app_module.cur.execute(
                        update_session_data_sql_entry(),
                        (params.session_id, SessionStatus.TERMINATED.value)
                    )
                    app_module.db_connection.commit()
                    return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.TERMINATED)
            else:
                if params.session_id in app_module.session_db.get("terminated_states", {}):
                    app_module.session_db[params.session_id] = {
                        "status": SessionStatus.TERMINATED,
                        "state": state
                    }
                    return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.TERMINATED)
            
                else:
                    app_module.session_db[params.session_id] = {
                        "status": SessionStatus.RUNNING,
                        "state": state
                    }

            log(
                session_id=params.session_id,
                level=LogLevels.INFO,
                service=create_session.__name__,
                message=f"Session {params.session_id} waiting on response from {state['second_speaker']} ({state['next_speaker'].value}).",
                mode=app_module.MODE
            )

            return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.RUNNING)


@router.get("/sessions/{session_id}/update", dependencies=[Depends(verify_api_key)])
async def get_session_update(
    session_id: str,
    response: ClientResponse = Depends(response)
) -> PersuasioResponse:
    """
    Update an existing debate session by supplying a new human response.

    This endpoint:
    1. Validates that the session exists and is in a RUNNING state.
    2. Passes the human response into LangGraph using `Command(resume=...)`.
    3. Saves the updated state to memory or PostgreSQL depending on mode.
    4. Returns the session as:
         - RUNNING (awaiting further human input),
         - FINISHED (dialogue completed), or
         - TERMINATED (runtime termination detected).

    Parameters
    ----------
    session_id : str
        Unique identifier of the session to update.
    response : ClientResponse
        Human utterance and metadata.

    Returns
    -------
    PersuasioResponse
        The updated dialogue state after the LangGraph invocation.

    Raises
    ------
    HTTPException
        If the session does not exist, has finished, or was terminated.
    """

    # --- MODE-AWARE SESSION CHECK ---
    if app_module.MODE == "production":
        log(
            session_id=session_id,
            level=LogLevels.INFO,
            service=get_session_update.__name__,
            message=f"Received human response from {response.utterance_from}.",
            mode=app_module.MODE
        )
        # Production: Check PostgreSQL
        app_module.cur.execute(
            "SELECT * FROM session_data WHERE id = %s",
            (session_id,)
        )
        db_response = app_module.cur.fetchone()

        if db_response is None:
            log_and_raise(
                session_id=session_id,
                status_code=404,
                service=get_session_update.__name__,
                message="This session has not been created yet.",
                mode=app_module.MODE
            )

        session_id_from_db = db_response[0]
        status = db_response[1]

        if session_id_from_db != session_id:
            log_and_raise(
                session_id=session_id,
                status_code=404,
                service=get_session_update.__name__,
                message=f"The session_id in PostgreSQL db ('{session_id_from_db}') and the session_id supplied to Persuasio ('{session_id}') were not the same.",
                mode=app_module.MODE
            )

        if status == SessionStatus.FINISHED.value:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=get_session_update.__name__,
                message=f"Session {session_id} is finished.",
                mode=app_module.MODE
            )

        if status == SessionStatus.TERMINATED.value:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=get_session_update.__name__,
                message=f"Session {session_id} has been terminated.",
                mode=app_module.MODE
            )

    else:
        # Set up logging for development mode
        logger_config = logging_config(session_id)
        logging.config.dictConfig(logger_config)
        logger = logging.getLogger(session_id)
        logger.info(f"Received human response from {response.utterance_from}.")
        log(
            session_id=session_id,
            level=LogLevels.INFO,
            service=get_session_update.__name__,
            message=f"Received human response from {response.utterance_from}.",
            mode=app_module.MODE
        )

        # Development: Check in-memory
        if session_id not in app_module.session_db:
            log_and_raise(
                session_id=session_id,
                status_code=404,
                service=get_session_update.__name__,
                message="This session has not been created yet.",
                mode=app_module.MODE
            )

        if app_module.session_db[session_id]["status"] == SessionStatus.FINISHED:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=get_session_update.__name__,
                message=f"Session {session_id} is finished.",
                mode=app_module.MODE
            )

        if app_module.session_db[session_id]["status"] == SessionStatus.TERMINATED:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=get_session_update.__name__,
                message=f"Session {session_id} has been terminated.",
                mode=app_module.MODE
            )

    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 1000}

    # Continue dialogue with a new response
    log(
        session_id=session_id,
        level=LogLevels.INFO,
        service=get_session_update.__name__,
        message=f"Parent graph invoked.",
        mode=app_module.MODE
    )
    graph = get_compiled_graph(compiled_graph=app_module.compiled_graph)
    state = await graph.ainvoke(
        Command(resume={f"response": response.model_dump()}),
        config=config
    )
    log(
        session_id=session_id,
        level=LogLevels.INFO,
        service=get_session_update.__name__,
        message=f"Parent graph returned.",
        mode=app_module.MODE
    )

    # Update runtime state
    if app_module.MODE == "production":
        update_query, update_state = save_state_to_db(session_state=state, session_state_status=SaveState.RUN_STATE)
        app_module.cur.execute(
            update_query,
            (session_id, update_state)
        )
        app_module.db_connection.commit()
    else:
        app_module.session_db[session_id]["state"] = state

    if "__interrupt__" in state:
        # If __interrupt__ in state, then the mode requires a response from user
        if app_module.MODE == "production":
            app_module.cur.execute(
                update_session_data_sql_entry(),
                (session_id, SessionStatus.RUNNING.value)
            )
            app_module.db_connection.commit()
            
            # Check to see if terminated
            app_module.cur.execute(
                "SELECT * FROM terminated_states WHERE id = %s",
                (session_id,)
            )
            db_response = app_module.cur.fetchone()
            # Save updated terminated state
            if db_response:
                update_query, update_state = save_state_to_db(session_state=state, session_state_status=SaveState.TERMINATED_STATES)
                app_module.cur.execute(
                    update_query,
                    (session_id, update_state)
                )
                app_module.cur.execute(
                        update_session_data_sql_entry(),
                        (session_id, SessionStatus.TERMINATED.value)
                    )
                app_module.db_connection.commit()
                return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.TERMINATED)
        else:
            if session_id in app_module.session_db.get("terminated_states", {}):
                app_module.session_db[session_id] = {
                    "status": SessionStatus.TERMINATED,
                    "state": state
                }
                return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.TERMINATED)
        
            else:
                app_module.session_db[session_id] = {
                    "status": SessionStatus.RUNNING,
                    "state": state
                }

        return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.RUNNING)

    else:
        # Dialogue has finished
        if app_module.MODE == "production":
            app_module.cur.execute(
                update_session_data_sql_entry(),
                (session_id, SessionStatus.FINISHED.value)
            )
            update_query, update_state = save_state_to_db(session_state=state, session_state_status=SaveState.FINAL_STATE)
            app_module.cur.execute(
                update_query,
                (session_id, update_state)
            )
            app_module.db_connection.commit()

            # Check to see if terminated
            app_module.cur.execute(
                "SELECT * FROM terminated_states WHERE id = %s",
                (session_id,)
            )
            db_response = app_module.cur.fetchone()
            # Save updated terminated state
            if db_response:
                update_query, update_state = save_state_to_db(session_state=state, session_state_status=SaveState.TERMINATED_STATES)
                app_module.cur.execute(
                    update_query,
                    (session_id, update_state)
                )
                app_module.cur.execute(
                        update_session_data_sql_entry(),
                        (session_id, SessionStatus.TERMINATED.value)
                    )
                app_module.db_connection.commit()
                return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.TERMINATED)
        else:
            if session_id in app_module.session_db.get("terminated_states", {}):
                app_module.session_db[session_id] = {
                    "status": SessionStatus.TERMINATED,
                    "state": state
                }
                return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.TERMINATED)
        
            else:
                app_module.session_db[session_id] = {
                    "status": SessionStatus.FINISHED,
                    "state": state
                }

        log(
            session_id=session_id,
            level=LogLevels.INFO,
            service=get_session_update.__name__,
            message=f"WINNER: {state['winner']}; LOSER: {state['loser']}, REASON: {state['reason_for_dialogue_termination']}",
            mode=app_module.MODE
        )
        log(
            session_id=session_id,
            level=LogLevels.INFO,
            service=get_session_update.__name__,
            message=f"DIALOGUE END: Session finished.",
            mode=app_module.MODE
        )

        return parse_state_to_persuasio_basemodel_object(session_state=state, session_status=SessionStatus.FINISHED)


@router.get("/sessions/finished", dependencies=[Depends(verify_api_key)])
async def get_completed_sessions():
    """
    Return all finished sessions
    """
    finished = get_finished()

    return {
        "sessions" : finished
    }


@router.get("/sessions/all_ended", dependencies=[Depends(verify_api_key)])
async def return_all_ended_sessions() -> Dict[str, List[str]]:
    """
    Return all terminated and finished sessions
    """
    ended = get_ended()

    return {
        "sessions" : ended
    }


@router.get("/sessions/ongoing", dependencies=[Depends(verify_api_key)])
async def get_ongoing_sessions() -> Dict[str, List[str]]:
    """
    Return all running sessions.
    """
    ongoing = get_ongoing()

    return {
        "sessions" : ongoing
    }


@router.get("/sessions/{session_id}/view_state", dependencies=[Depends(verify_api_key)])
async def view_state(session_id: str):
    """
    Retrieve the full session state
    """
    return get_state(session_id, view_state.__name__)


@router.get("/sessions/{session_id}/log", dependencies=[Depends(verify_api_key)])
async def get_session_log(session_id: str, limit: Optional[int] = 100):
    """
    Retrieve session logs from the file system (dev) or PostgreSQL db (production)

    Args:
        session_id: ID of the session
        limit: Maximum number of log entries to return (most recent). Default 100. Set to None or 0 for all logs.
    """
    logs = get_logs(session_id, get_session_log.__name__)

    # Apply limit to get last N entries
    if limit and limit > 0:
        logs = logs[-limit:]

    # Reverse to show newest first
    logs = list(reversed(logs))

    return {
        "log": logs
    }


@router.post("/sessions/terminate/{session_id}", dependencies=[Depends(verify_api_key)])
async def terminate_session(session_id : str):
    """
    Forcefully terminate a session and archive its last known state.

    In production:
        - Marks the session as TERMINATED in `session_data`.
        - Saves the final runtime state into `terminated_states`.

    In development:
        - Updates the in-memory session status to TERMINATED.
        - Copies the last state into the `terminated_states` dictionary.

    Parameters
    ----------
    session_id : str
        ID of the session to terminate.

    Raises
    ------
    HTTPException
        If the session does not exist and cannot be terminated.
    """
    if app_module.MODE == "production":
        # Production
                
        # Get the current state from runtime_states so we can save it to terminated_states table
        app_module.cur.execute(
            "SELECT * FROM runtime_states WHERE id = %s",
            (session_id,)
            )
        db_response = app_module.cur.fetchone()

        if not db_response:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=terminate_session.__name__,
                message=f"Session '{session_id}' has not been created yet.",
                mode=app_module.MODE
            )

        last_state = db_response[1]

        # Add terminated session_id and state terminated_sessions PostgreSQL table
        update_query, update_state = save_state_to_db(session_state=last_state, session_state_status=SaveState.TERMINATED_STATES)
        app_module.cur.execute(
            update_query,
                (session_id, update_state)
            )
        
        app_module.db_connection.commit()

    else:
        # Set the session_id to terminated
        app_module.session_db[session_id]["status"] = SessionStatus.TERMINATED

        # Find the last state
        last_state = app_module.session_db[session_id]["state"] 

        # Create a new key-value pair to store terminated dialogues
        if "terminated_states" not in app_module.session_db:
            app_module.session_db["terminated_states"] = {
                session_id : last_state
                }
        else:
            app_module.session_db["terminated_states"][session_id] = last_state



@router.get("/sessions/list-terminated", dependencies=[Depends(verify_api_key)])
async def return_terminated_sessions() -> Dict[str, List[str]]:
    """
    Return all terminated sessions
    """
    terminated = get_terminated()

    return {
        "sessions" : terminated
    }

@router.get("/sessions/view-terminated-state/{session_id}", dependencies=[Depends(verify_api_key)])
async def return_terminated_sessions(session_id : str) :
    """
    Retrieve the full terminated state of a session.

    In production, this pulls from the `terminated_states` PostgreSQL table.
    In development, it reads from the in-memory `terminated_states` mapping.

    Parameters
    ----------
    session_id : str
        The ID of the terminated session to retrieve.

    Returns
    -------
    PersuasioResponse
       A formatted representation of the terminated session state.

    Raises
    ------
    HTTPException
        If the session is not terminated or does not exist.
    """
    if app_module.MODE == "production":
        # Production: get session data from PostgreSQL       
        app_module.cur.execute(
                "SELECT * FROM terminated_states WHERE id = %s",
                (session_id,)
                )
        db_response = app_module.cur.fetchone()

        if db_response == None:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=terminate_session.__name__,
                message=f"Session '{session_id}' hasn't been terminated yet.",
                mode=app_module.MODE
            )

        terminated_session_state = db_response[1]


    else:
        # Development: Get from in-memory dict
        if "terminated_states" not in app_module.session_db:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=terminate_session.__name__,
                message=f"No sessions have been terminated yet.",
                mode=app_module.MODE
            )
        elif session_id not in app_module.session_db:
            log_and_raise(
                session_id=session_id,
                status_code=400,
                service=terminate_session.__name__,
                message=f"Session '{session_id}' has not been created.",
                mode=app_module.MODE
            )
        else:
            terminated_session_state = app_module.session_db.get("terminated_states").get(session_id, {})

            if len(terminated_session_state) == 0:
                log_and_raise(
                    session_id=session_id,
                    status_code=400,
                    service=terminate_session.__name__,
                    message=f"Session {session_id} was terminated but session db does not have access to the state.",
                    mode=app_module.MODE
                )

    return parse_state_to_persuasio_basemodel_object(session_state=terminated_session_state, session_status=SessionStatus.TERMINATED)
