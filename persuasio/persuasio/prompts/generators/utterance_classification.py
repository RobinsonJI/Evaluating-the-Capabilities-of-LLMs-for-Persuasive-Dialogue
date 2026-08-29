from typing import List, Dict

from persuasio.datatypes.enums import ClassifyingUtteranceOf
from persuasio.prompts.system.utterance_classification import utt_class_sys_msg

def create_utterance_classification_prompt(current_sentence, state, which_utterances : ClassifyingUtteranceOf) -> List[Dict[str, str]]:

    prior_utterances = ""
    if which_utterances == ClassifyingUtteranceOf.HUMAN_RESPONSE:
        if len(state["dialogue_history"]) > 1:
            prior_utterances += "User: "
            prior_utterances += " ".join(["<" + x[0].strip("_").lower() + "> '" + x[1] + "'<" + x[0].strip("_").lower() + "/>" for x in state["dialogue_history"][-2].sentences_with_utterance_types])
            prior_utterances += "'\n"
        if len(state["dialogue_history"]) > 0:
            prior_utterances += "Model: "
            prior_utterances += " ".join(["<" + x[0].strip("_").lower() + "> '" + x[1] + "'<" + x[0].strip("_").lower() + "/>" for x in state["dialogue_history"][-1].sentences_with_utterance_types])

    elif which_utterances == ClassifyingUtteranceOf.LAST_SPEAKER:
        if len(state["dialogue_history"]) > 2:
            prior_utterances += "User: "
            prior_utterances += " ".join(["<" + x[0].strip("_").lower() + "> '" + x[1] + "'<" + x[0].strip("_").lower() + "/>" for x in state["dialogue_history"][-3].sentences_with_utterance_types])
            prior_utterances += "'\n"
        if len(state["dialogue_history"]) > 1:
            prior_utterances += "Model: "
            prior_utterances += " ".join(["<" + x[0].strip("_").lower() + "> '" + x[1] + "'<" + x[0].strip("_").lower() + "/>" for x in state["dialogue_history"][-2].sentences_with_utterance_types])

    human_msg = {
        "role": "user",
        "content": f"""Classify the following sentence according to the defined utterance types:
        
        Previous dialogue turns:
        {prior_utterances}
        
        Current sentence:
        User: {current_sentence}"""
    }

    prompt = [utt_class_sys_msg, human_msg]

    return prompt