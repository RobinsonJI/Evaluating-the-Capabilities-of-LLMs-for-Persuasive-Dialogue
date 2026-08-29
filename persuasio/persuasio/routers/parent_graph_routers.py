from typing import Literal

from persuasio.states.state import ParentState

from persuasio.utils.logs import log_function

@log_function
def check_for_end_of_dialogue_after_first_speaker(state: ParentState) -> Literal["SecondSpeaker", "EndOfDialogueOutputs"]:
    if (len(state["winner"]) > 0) and (len(state["loser"]) > 0) and (len(state["reason_for_dialogue_termination"]) > 0):
        return "EndOfDialogueOutputs"
    else:
        return "SecondSpeaker"

@log_function
def check_for_end_of_dialogue_after_second_speaker(state: ParentState) -> Literal["FirstSpeaker", "EndOfDialogueOutputs"]:
    if (len(state["winner"]) > 0) and (len(state["loser"]) > 0) and (len(state["reason_for_dialogue_termination"]) > 0):
        return "EndOfDialogueOutputs"
    else:
        return "FirstSpeaker"