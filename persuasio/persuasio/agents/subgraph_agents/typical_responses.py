from typing import TypeVar
from persuasio.states.state import HumanState, GenerationAgentsState
from persuasio.utils.logs import log_function

T = TypeVar("T", HumanState, GenerationAgentsState)

@log_function
def replies_for_last_utterance(state: T) -> T:
    """
    Identifies the typical responses for the sentences output by the human user/LLM
    """

    typical_utterances_to_reply_with = []
    for index, sentence in enumerate(state[f"opponents_utterance_with_corresponding_types"]):
        responses = {}
        if sentence[0] == "___Claim___":
            responses["___Why___"] = " ".join(sentence[1:])
            responses["___Question___"] = " ".join(sentence[1:])
            responses["___NotClaim___"] = " ".join(sentence[1:])
            responses["___Concede___"] = " ".join(sentence[1:])
            

        elif sentence[0] == "___Why___":
            responses["___Since___"] = " ".join(sentence[1:])
            responses["___Claim___"] = " ".join(sentence[1:])
            responses["___Retract___"] = " ".join(sentence[1:])

        elif sentence[0] == "___Since___":
            responses["___Why___"] = " ".join(sentence[1:])
            responses["___Question___"] = " ".join(sentence[1:])
            responses["___Concede___"] = " ".join(sentence[1:])

        elif sentence[0] == "___Question___":
            responses["___Claim___"] = " ".join(sentence[1:])
            responses["___NotClaim___"] = " ".join(sentence[1:])
            responses["___Retract___"] = " ".join(sentence[1:])
                
        # If the responses dictionary is not empty, append the typical responses
        if bool(responses):
            typical_utterances_to_reply_with.append((responses, len(state["dialogue_history"]), index))
  
    return {
        "typical_replies_for_last_speakers_response" : typical_utterances_to_reply_with,
    }