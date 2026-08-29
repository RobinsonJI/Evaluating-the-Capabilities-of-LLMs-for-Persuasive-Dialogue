from langgraph.types import interrupt

from persuasio.states.state import HumanState
from persuasio.datatypes.enums import LogLevels
from persuasio.utils.logs import log_function, log

@log_function
def interrupt_and_resume(state: HumanState) -> HumanState:

    log(
        session_id=state["session_id"],
        level=LogLevels.INFO,
        service=interrupt_and_resume.__name__,
        message=f"LangGraph Interruption; waiting on response from '{state["speaker"]}' ({state["current_speaker"].value}); DIALOGUE TURN NUM = {len(state['dialogue_history'])}.",
        mode=state["mode"]
    )
    response = interrupt(state)
    log(
        session_id=state["session_id"],
        level=LogLevels.INFO,
        service=interrupt_and_resume.__name__,
        message=f"Response received from '{state["speaker"]}' ({state["current_speaker"].value}); DIALOGUE TURN NUM = {len(state['dialogue_history'])}.",
        mode=state["mode"]
    )

    utterance = response["response"]["utterance"].strip()
    utterance_from = response["response"]["utterance_from"]
    next_speaker = response["response"]["next_speaker"]

    return {
        f"utterance" : utterance,
        "current_speaker" : utterance_from,
        "next_speaker" : next_speaker
    }