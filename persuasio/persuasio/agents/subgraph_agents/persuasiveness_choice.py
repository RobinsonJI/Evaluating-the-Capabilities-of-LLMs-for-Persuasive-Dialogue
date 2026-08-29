from persuasio.states.state import GenerationAgentsState
from persuasio.datatypes.pydantic_basemodels import PersuasivenessChoice, PersuasivenessChoicesList
from persuasio.datatypes.enums import LogLevels
from persuasio.prompts.generators.dialogue_moves import MASPromptGenerator
from persuasio.models.models import GenerateLLMResponses
from persuasio.utils.logs import log_function, log

@log_function
def llm_completion_choice(state: GenerationAgentsState) -> GenerationAgentsState:
    """
    Selects the most persuasive sentence(s) from a set of LLM-generated candidates,
    ensuring that chosen sentences are not semantically repetitive across multiple prompts.

    This function works as part of a persuasiveness choice agent:
    1. It generates prompts for the LLM to evaluate candidate sentences.
    2. It appends previously chosen sentences to new prompts so the model avoids repetition.
    3. It processes the LLM's choice, matching it against generated candidates.
    4. It returns the chosen sentences with their corresponding utterance types.

    Args:
        state (LLMResponseState): 
            A state dictionary containing the conversation context and any prior LLM outputs.
            The `PromptGenerator` uses this to produce prompts for evaluating persuasiveness.

    Returns:
        LLMResponseState:
            A dictionary containing:
            - "last_llm_utterance_with_corresponding_types": list of lists where:
                - index 0 = utterance type (e.g., "___Claim___", "___Why___")
                - index 1 = the chosen sentence string
    """    
    # Step 1: Generate a list of prompts for the persuasiveness choice agent.
    # Each prompt contains:
    #   prompt[0], prompt[1] -> system & user messages for the LLM ***These are the strings that are given to the model***
    #   prompt[2] -> list of generations as plain strings
    #   prompt[3] -> list of generations with utterance types
    
    prompts = MASPromptGenerator(state=state).persuasiveness_choice()

    # This conditional executes if the LLM has generated an initial claim and the human user has not made any utterances yet.
    if len(state["dialogue_history"]) == 0:
        choice = None
        try:
            result = GenerateLLMResponses(model_choice=state["speaker_model_name"],
                                            prompt = prompts[0][:2],
                                            temperature=state["model_temp"],
                                            top_p= state["model_top_p"],
                                            seed= state["model_seed"],
                                            datatype_schema=PersuasivenessChoice).return_completion()
            choice = result["Choice"]
            log(
                session_id=state["session_id"],
                level=LogLevels.INFO,
                service=llm_completion_choice.__name__,
                message=f"Completion returned and validated within 'PersuasivenessChoiceAgent'; MODEL = '{state['speaker_model_name'].value}'; DIALOGUE TURN NUM = {len(state['dialogue_history'])}.",
                mode=state["mode"]
            )
        except ValueError as e:
            log(
                session_id=state["session_id"],
                level=LogLevels.ERROR,
                service=llm_completion_choice.__name__,
                message=f"Completion validation error within 'PersuasivenessChoiceAgent'; \n REASON:\n\n {e}",
                mode=state["mode"],
                context={"prompt" : prompts[0][:2], "state" : state, "exception" : e}
            )
            

        return {
            "utterance_with_corresponding_types" : [("___Claim___", choice)]
        }
        
    # Tracks sentences already chosen by the LLM to avoid repetition    
    persuasive_sentence_choice = []

    # Tracks chosen sentences alongside their utterance types
    persuasive_sentence_choice_with_utterance_types = []

    # Step 2: Iterate through each prompt to make a persuasiveness choice
    for prompt in prompts:
        # We need to add previously chosen sentences to the prompt so that the model knows not to choose something similar if 
        # it has been uttered in a previous sentence. So we will add the previous choices contained in 'persuasive_sentence_choice' 
        # for each new prompt in the persuasiveness choice agent.
        adapted_prompt = prompt[:2]         # Prepare the initial part of the prompt (system + user message)

        generations_as_string = prompt[2]
        generations_with_utterance_types = prompt[3]
        
        # If there are previously chosen sentences, append them to the prompt
        # so the LLM avoids selecting something too similar
        if len(persuasive_sentence_choice) > 0:
            adapted_prompt[0]["content"] += "\n".join(persuasive_sentence_choice)

        result = None
        try:
            # Step 3: Query the LLM to choose the most persuasive sentence
            result = GenerateLLMResponses(model_choice=state["speaker_model_name"],
                                            prompt = adapted_prompt,
                                            temperature=state["model_temp"],
                                            top_p= state["model_top_p"],
                                            seed= state["model_seed"],
                                            datatype_schema=PersuasivenessChoicesList).return_completion()
            log(
                session_id=state["session_id"],
                level=LogLevels.INFO,
                service=llm_completion_choice.__name__,
                message=f"Completion returned and validated within 'PersuasivenessChoiceAgent'; MODEL = '{state['speaker_model_name'].value}'; DIALOGUE TURN NUM = {len(state['dialogue_history'])}.",
                mode=state["mode"]
            )
        except ValueError as e:
            log(
                session_id=state["session_id"],
                level=LogLevels.ERROR,
                service=llm_completion_choice.__name__,
                message=f"Completion validation error within 'PersuasivenessChoiceAgent'; \n REASON:\n\n {e}",
                mode=state["mode"],
                context={"prompt" : adapted_prompt, "state" : state, "exception" : e}
            )
            
        
        # Valid utterance type labels
        utterance_types = ["___Claim___", "___Why___", "___Since___", "___Question___", "___Concede___", "___Retract___"]

        # Step 4: Match the LLM's choice to the correct generation
        for choice in result["Choices"]:
            try:
                if any(item in choice for item in utterance_types):
                    # The choice contains its utterance type in the string
                    for generation in generations_as_string:
                        if choice == generation:
                            persuasive_sentence_choice.append(choice)
                            persuasive_sentence_choice_with_utterance_types.append(
                                tuple(choice.split(" ", 1)) # Split into [type, sentence]
                            )

                elif any(sublist[1] == choice for sublist in generations_with_utterance_types):
                    # The agent's choice matches only the sentence (utterance type provided separately)
                    for generation in generations_with_utterance_types:
                        if generation[1] == choice:
                            persuasive_sentence_choice.append(" ".join(generation))
                            persuasive_sentence_choice_with_utterance_types.append(tuple(generation))
            except ValueError as e:
                log(
                    session_id=state["session_id"],
                    level=LogLevels.ERROR,
                    service=llm_completion_choice.__name__,
                    message=f"Completion did not match prior utterances; MODEL = '{state['speaker_model_name'].value}'",
                    mode=state["mode"],
                    context={"completion" : choice, "prior_utterances" : [f'{x}' for x in generations_as_string], "state" : state, "exception" : e}
                )

    return {
        "utterance_with_corresponding_types" : persuasive_sentence_choice_with_utterance_types
    }