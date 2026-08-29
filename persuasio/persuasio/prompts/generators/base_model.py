from typing import List, Dict

from persuasio.states.state import BaseModelState
from persuasio.prompts.system.base_model import create_base_model_system_prompt
from persuasio.utils.logs import log_function

@log_function
def create_base_model_prompt(state : BaseModelState) -> List[Dict[str, str]]:
    """
    Construct a prompt for the base model consisting of a system message 
    and a user message. 

    This function creates the dialogue context by combining:
      1. A system prompt that defines the model's role and behavior.
      2. A user message containing the opponent's last utterance, 
         along with instructions for generating a persuasive response.

    Parameters
    ----------
    state : BaseModelState
        The current state containing metadata and dialogue context, 
        such as the opponent's last utterance and political stance.

    Returns
    -------
    List[Dict[str, str]]
        A list of messages forming the prompt, structured as:
        [
            {"role": "system", "content": <system prompt>},
            {"role": "user", "content": <formatted opponent utterance and instructions>}
        ]
    """

    # Generate the system message defining model behavior and constraints
    sys_msg = create_base_model_system_prompt(state)

    # Construct the human/user message with the opponent's last utterance
    human_msg = {
        "role" : "user",
        "content" : f"""Make a response:

Human user's last utterance:
'{state.get("opponents_utterance", "")}'

Follow the instructions in the system prompt to make a response, in accordance with your political position, to persuade your opponent to your position.
"""
    }

    # Combine system and user messages into a single prompt
    prompt = [sys_msg, human_msg]

    return prompt