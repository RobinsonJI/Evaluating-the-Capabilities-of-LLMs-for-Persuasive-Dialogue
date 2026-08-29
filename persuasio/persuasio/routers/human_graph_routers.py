from typing import Literal

from persuasio.states.state import HumanState
from persuasio.utils.logs import log_function

@log_function
def check_if_user_starts_dialogue(state: HumanState) -> Literal["UtteranceClassificationAgentBeforeInterrupt", "UtteranceClassificationAgentAfterInterrupt"]:
    """
    Checks to see if there has been any dialogue turns. If no dialogue turns, then we know that human is starting the dialogue.
    """
    if len(state["dialogue_history"]) == 0:
        return "UtteranceClassificationAgentAfterInterrupt"
    else:
        return "UtteranceClassificationAgentBeforeInterrupt"