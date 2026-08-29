from typing import TypeVar

from persuasio.states.state import GenerationAgentsState, BaseModelState

T = TypeVar("T", GenerationAgentsState, BaseModelState)

def conversation_history(state : T) -> str:

    # Getting the conversation history
    combined = []
    for index, dialogue_data in enumerate(state.get("dialogue_history", [])):
        _role = dialogue_data.speaker
        sentences = dialogue_data.sentences_no_utterance_types
        if state["speaker"] == _role:
            role = "You"
        else:
            role = "Human"
        combined.append(f"Dialogue turn {index+1}, {role}: '{sentences}'\n")
    
    conversation_string = "\n".join(combined)

    string = f"""# Conversation History
    
You must bare the following conversation in mind when generating your response. This is the conservation up to this point in time:

{conversation_string}

"""
    return string


def dialogue_claims_and_commitments(state : T) -> str:
    """
    Creates a string containing the dialogue history. All generation agents use this string so created a function to create it so that it's more maintainable. 
    """

    commitments = "\n".join([f"'{commitment}'" for commitment in state.get("commitments", [])])
    opponents_commitments = "\n".join([f"'{commitment}'" for commitment in state.get("opponents_commitments", [])])

    string = f"""# User and System's Initial Claims and Commitments

Your initial claim:
{state.get("original_claim", "")}

Your prior commitments:
{commitments}

The human user's initial claim:
{state.get("opponents_original_claim", "")}

The human user's prior commitments:
{opponents_commitments}

"""
    return string