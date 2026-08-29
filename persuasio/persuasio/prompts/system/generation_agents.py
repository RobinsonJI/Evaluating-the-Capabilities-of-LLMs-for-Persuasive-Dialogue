from typing import Dict

from persuasio.states.state import GenerationAgentsState
from persuasio.prompts.system.dialogue_data import dialogue_claims_and_commitments, conversation_history
from persuasio.prompts.system.political_alignment import model_political_position
from persuasio.prompts.system.generic_rules import rules_for_all_models


def create_claim_negation_system_prompt(state : GenerationAgentsState) -> Dict[str, str]:

    political_position_string = model_political_position(
        debate_topic=state["debate_topic"],
        political_position_range=state["political_position_range"],
        political_position_std=state["political_position_std"]
    )
    dialogue_data = dialogue_claims_and_commitments(state=state)
    history = conversation_history(state=state)

    sys_msg = {"role" : "system",
            "content" : f"""# Task
            
Your task is to generate an alternative perspective to the user's sentence. The alternative perspective must be a single sentence that contradicts or opposes the user's sentence. 
The sentences should have a compelling tone, clarity, emotional impact, and convincing word choice.

{political_position_string}

{state.get("vector_based_persona_string", "")}

{dialogue_data}

{history}

# Rules

- Each response must be exactly one sentence.
- The response MUST contradict / conflict with what the user has said.
{rules_for_all_models}


"""
            }
    
    return sys_msg

def create_why_claim_system_prompt(state : GenerationAgentsState) -> Dict[str, str]:
    political_position_string = model_political_position(
        debate_topic=state["debate_topic"],
        political_position_range=state["political_position_range"],
        political_position_std=state["political_position_std"]
    )
    dialogue_data = dialogue_claims_and_commitments(state=state)
    history = conversation_history(state=state)

    sys_msg = {"role" : "system",
                "content" : f"""# Task
                
Your task is to generate 3 questions that challenge the human user's sentence. 
Your response MUST start with the word WHY. The question you generate should ask the user to provide the reasons why they believe the sentence they said. 
Responses should be ONE sentence in length and no longer. 

{political_position_string}

{state.get("vector_based_persona_string", "")}

{dialogue_data}

{history}

# Rules

- Each response must be exactly one sentence.
- The response MUST contain the word 'why'.
{rules_for_all_models}


"""
            }
    return sys_msg

def create_question_claim_system_prompt(state : GenerationAgentsState) -> Dict[str, str]:
    political_position_string = model_political_position(
        debate_topic=state["debate_topic"],
        political_position_range=state["political_position_range"],
        political_position_std=state["political_position_std"]
    )
    dialogue_data = dialogue_claims_and_commitments(state=state)
    history = conversation_history(state=state)
    
    sys_msg = {"role" : "system",
                "content" : f"""# Task
                
Your task is to generate 3 journalistic questions that challenge the human user's sentence.
Your response MUST start with either 'Who', 'What', 'When', 'Where', 'Why', or 'How'. 
The question you generate should ask the user to provide the reasons why they believe the sentence they said. 
Responses should be ONE sentence in length and no longer. 

{political_position_string}

{state.get("vector_based_persona_string", "")}

{dialogue_data}

{history}

# Rules

- Each response must be exactly one sentence.
- The response MUST start with either 'Who', 'What', 'When', 'Where', 'Why', or 'How'.
{rules_for_all_models}


"""   
        }
    
    return sys_msg

def create_concede_claim_system_prompt(state : GenerationAgentsState) -> Dict[str, str]:
    political_position_string = model_political_position(
        debate_topic=state["debate_topic"],
        political_position_range=state["political_position_range"],
        political_position_std=state["political_position_std"]
    )
    dialogue_data = dialogue_claims_and_commitments(state=state)
    history = conversation_history(state=state)
    
    sys_msg = {"role" : "system",
                "content" : f"""# Task 
                
Your task is to generate 3 sentences that explain why you accept (or concede to) the human user's sentence point of view. 
You should include the phrase 'I accept' or 'I admit' in your answer. Your sentence should be as concise as possible such that 
your sentence explicitly references the thing you are conceding to and does not include any superfluous preamble.

{political_position_string}

{state.get("vector_based_persona_string", "")}

{dialogue_data}

{history}

# Rules

- Each response must be exactly one sentence.
- The response MUST concede or agree with what the human user has previously said.
{rules_for_all_models}


"""   
        }    

    return sys_msg

def create_since_claim_system_prompt(state : GenerationAgentsState) -> Dict[str, str]:
    political_position_string = model_political_position(
        debate_topic=state["debate_topic"],
        political_position_range=state["political_position_range"],
        political_position_std=state["political_position_std"]
    )
    dialogue_data = dialogue_claims_and_commitments(state=state)
    history = conversation_history(state=state)
    
    sys_msg = {"role" : "system",
                "content" : f"""# Task
                
Your task is to generate 3 sentences that answer the human user's question. You must ensure that your answer agrees with 
list of prior commitments in the conversation and your initial claim, whilst also ensuring you explain why you said your last utterance.

{political_position_string}

{state.get("vector_based_persona_string", "")}

{dialogue_data}

{history}

# Rules

- Each response must be exactly one sentence.
- The response MUST support the claim that you previously made.
{rules_for_all_models}


"""
        }
    
    return sys_msg

def create_claim_system_prompt(state : GenerationAgentsState) -> Dict[str, str]:
    political_position_string = model_political_position(
        debate_topic=state["debate_topic"],
        political_position_range=state["political_position_range"],
        political_position_std=state["political_position_std"]
    )
    dialogue_data = dialogue_claims_and_commitments(state=state)
    history = conversation_history(state=state)
    
    sys_msg = {"role" : "system",
                "content" : f"""# Task 

Your task is to generate 3 sentences that answers to a human user's question. You must ensure that your answer agrees with 
list of prior commitments in the conversation and your initial claim, whilst also ensuring you explain why you said your last utterance.

{political_position_string}

{state.get("vector_based_persona_string", "")}

{dialogue_data}

{history}

# Rules

- Each response must be exactly one sentence.
- The response MUST be a claim that answers the user's question.
{rules_for_all_models}


"""
        }
    
    return sys_msg

def create_retract_claim_system_prompt(state : GenerationAgentsState,
                                       model_last_dialogue_move : str) -> Dict[str, str]:
    political_position_string = model_political_position(
        debate_topic=state["debate_topic"],
        political_position_range=state["political_position_range"],
        political_position_std=state["political_position_std"]
    )
    history = conversation_history(state=state)
    dialogue_data = dialogue_claims_and_commitments(state=state)
    
    sys_msg = {"role" : "system",
            "content" : f"""# Task 
                            
Your task is to generate 3 sentences that make a retracting statement. A human user has either challenged 
a statement you made earlier or asked you a question. You MUST either: use the human user's statement to retract a claim you previously 
said; or communicate that you are not committed to the claim implied in the question. A retraction can only be a retraction when are committed to 
the retracted statement. If the sentence is not in your set of commitments, then you should state that you are not committed to the 
claim implied within the question.

{political_position_string}

{state.get("vector_based_persona_string", "")}

{dialogue_data}

{history}

The statement you made was: '{model_last_dialogue_move}'

You must now retract that statement.
    
# Rules

- Each response must be exactly one sentence.
- The response MUST retract something you previously said.
{rules_for_all_models}


"""
        }

    return sys_msg 