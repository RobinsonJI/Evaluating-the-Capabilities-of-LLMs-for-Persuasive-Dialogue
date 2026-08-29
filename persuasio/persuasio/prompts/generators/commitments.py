from typing import List, Dict

from persuasio.prompts.system.commitments import (
    utterance_similar_to_commitments_sys_msg,
    utterance_concede_to_opponents_commitments_sys_msg, 
    utterance_concede_to_opponents_orig_claim_sys_msg,
    utterance_retract_own_initial_claim_sys_msg,
    utterance_retracts_current_speakers_commitments_sys_msg,
    which_commitments_were_retracted_sys_msg)


def is_utterance_similar_to_any_commitments(utterance : str, commitments: List[str]) -> List[Dict[str, str]]:

    commitments = "\n".join([f"'{commitment}'" for commitment in commitments])

    human_msg = {
        "role" : "user",
        "content" : f"""Here is the user's list of commitments:

{commitments}

Here is the last utterance:

'{utterance}'

Is the utterance already in or semantically similar to the user's set of commitments?
"""
    }

    prompt = [utterance_similar_to_commitments_sys_msg, human_msg]

    return prompt

def utterance_concedes_to_opponents_commitments(utterance : str, commitments : List[str]) -> List[Dict[str, str]]:

    commitments = "\n".join([f"'{commitment}'" for commitment in commitments])

    human_msg = {
        "role" : "user",
        "content" : f"""Here is the user's list of commitments:

{commitments}

Here is the last utterance:

'{utterance}'

Does the utterance concede to a claim within the user's set of commitments (i.e. admit that one of the user's claims is the case)?
"""
    }

    prompt = [utterance_concede_to_opponents_commitments_sys_msg, human_msg]

    return prompt


def sentence_concede_to_initial_claim(utterance : str, opponents_initial_claim: str) -> List[Dict[str, str]]:

    human_msg = {
        "role" : "user",
        "content" : f"""Here is the user's initial claim.

'{opponents_initial_claim}'

Here is your last utterance:

'{utterance}'

Does your last utterance concede to the user's initial claim?
"""
    }

    prompt = [utterance_concede_to_opponents_orig_claim_sys_msg, human_msg]

    return prompt

def does_utterance_retract_initial_claim(utterance : str, current_speakers_initial_claim : str) -> List[Dict[str, str]]:

    human_msg = {
        "role" : "user",
        "content" : f"""Here is the user's initial claim:

'{current_speakers_initial_claim}'

Here is the user's last utterance:

'{utterance}'

Does the user's last utterance retract their initial claim?
"""
    }

    prompt = [utterance_retract_own_initial_claim_sys_msg, human_msg]

    return prompt

def does_utterance_retract_any_commitments(utterance : str, commitments : List[str]) -> List[Dict[str, str]]:

    commitments = "\n".join([f"'{commitment}'" for commitment in commitments])

    human_msg = {
        "role" : "user",
        "content" : f"""Here is the user's list of commitments:

{commitments}

Here is the user's last utterance:

'{utterance}'

Does the user's last utterance retract any of their commitments?
"""
    }

    prompt = [utterance_retracts_current_speakers_commitments_sys_msg, human_msg]

    return prompt

def find_retracted_commitments(utterance : str, commitments : List[str]) -> List[Dict[str, str]]:

    commitments = "\n".join([f"'{commitment}'" for commitment in commitments])

    human_msg = {
        "role" : "user",
        "content" : f"""Here is the user's list of commitments:

{commitments}

Here is the user's last utterance:

'{utterance}'

Use the user's last utterance to append the retracted sentences from the commitments to the list of retracted sentences.
"""
    }

    prompt = [which_commitments_were_retracted_sys_msg, human_msg]

    return prompt
