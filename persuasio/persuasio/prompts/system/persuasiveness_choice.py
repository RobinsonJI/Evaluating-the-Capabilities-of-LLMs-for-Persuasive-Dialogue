from typing import Dict

from persuasio.states.state import GenerationAgentsState
from persuasio.prompts.system.dialogue_data import dialogue_claims_and_commitments, conversation_history
from persuasio.prompts.system.political_alignment import model_political_position


def create_persuasiveness_choice_system_prompt_for_initial_claim(state : GenerationAgentsState) -> Dict[str, str]:
    political_position_string = model_political_position(
        debate_topic=state["debate_topic"],
        political_position_range=state["political_position_range"],
        political_position_std=state["political_position_std"]
    )
    sys_msg = {"role" : "system",
                    "content" : f"""# Task
                    
You have been given a list of sentences and are about to begin a political debate. 
                    
Your task is to choose the most persuasive sentence out of a set of sentences.
                                    
A persuasive sentence is one that is more likely to convince someone, inspire action, or change their mind. 
Use the following criteria to rank the sentences: tone, clarity, emotional impact, word choice, and call to action.
Persuasive sentences are: clear; confident; and encourage action.

You MUST choose a strong sentence because you will have to defend that sentence through the rest of the debate.

{political_position_string}

{state.get("vector_based_persona_string", "")}

"""
            }

    return sys_msg

def create_persuasiveness_choice_system_prompt(state : GenerationAgentsState) -> Dict[str, str]:
    political_position_string = model_political_position(
        debate_topic=state["debate_topic"],
        political_position_range=state["political_position_range"],
        political_position_std=state["political_position_std"]
    )
    history = conversation_history(state=state)
    dialogue_data = dialogue_claims_and_commitments(state=state)
    
    sys_msg = {"role" : "system",
                    "content" : f"""# Task

Your task is to choose 1 to 5 sentences that you believe are the most persuasive out of a set of sentences.
                
A persuasive sentence is one that is more likely to convince someone, inspire action, or change their mind. 
Use the following criteria to rank the sentences: tone, clarity, emotional impact, word choice, and call to action.
Persuasive sentences are: clear; confident; and encourage action.

# Instructions

Choose 1 to 5 sentences that you think are the most persuasive.

Below is an ordered list of types of sentences, from least persuasive to most:
- ___Concede___: Concede / agreement with the user. *Least persuasive as it reinforces the user's stance instead of advancing your own.*
- ___Retract___: Retracting a point you previously made. *Low persuasiveness as it weakens your prior stance but it can increase credibility by showing honesty.*
- ___Why___: Challenging the user i.e. asking 'why'. *Moderately persuasive as it puts the burden of proof on the user but does not provide counterevidence.*
- ___Question___: Questioning the user. *More persuasive than 'why' because it can expose weaknesses or inconsistencies providing further support for your stance.*
- ___Since___: Supporting a claim. *Highly persuasive because it backs up your stance with a justification.*
- ___Claim___: Making a claim. *Most persuasive as it asserts confidence and establishes a strong position to perusade others.*

{political_position_string}

{state.get("vector_based_persona_string", "")}

{dialogue_data}

{history}""" + """

If you do not choose a sentence that corresponds to your original argument and set of commitments, then you will be seen as incoherent which is unacceptable.

You are NOT allowed to choose a sentence that you have previously chosen before because you will become repetitive and disengaging, which isn't persuasive, and the user requires a diverse set of persuasive sentences.


# Output format:
Return only a JSON object in the following format:
{
  "Choices": ["PersuasiveSentence_1", "PersuasiveSentence_2", "..."]
}

# Rules 
- Preserve the original sentence text exactly as provided.
- Try to choose a set of sentences that, when read together, make a strong and persuasive argument that agrees with your political alignment and persona, whilst also ensuring that your choices make sense within the context of the debate.
- Try not to choose sentences you have chosen before. The sentences you chose before are:
"""
            }    
    
    return sys_msg


'''sys_msg = {"role" : "system",
                    "content" : f"""### Task

Your task is to choose the most persuasive sentence out of a set of sentences.
                
A persuasive sentence is one that is more likely to convince someone, inspire action, or change their mind. 
Use the following criteria to rank the sentences: tone, clarity, emotional impact, word choice, and call to action.
Persuasive sentences are: clear; confident; and encourage action.

### Instructions

Choose a sentence that makes a claim or provides an explanation of the reasoning.
Deprioritise sentences that concede/retract a point, weaken an argument, or introduce too much uncertainty. 

Below is an ordered list of types of sentences, from least persuasive to most:
- ___Concede___: Concede / agreement with the user -> NOT PERSUASIVE
- ___Retract___: Retracting a point you previously made -> NOT PERSUASIVE
- ___Why___: Challenging the user i.e. asking 'why' -> PERSUASIVE
- ___Question___: Questioning the user -> PERSUASIVE
- ___Claim___: Making a claim -> EXTREMELY/THE MOST PERSUASIVE

{vector_based_persona_rag_schema}

### Dialogue History and User and System's Commitments

You MUST chosen a sentence that is accordance with your original argument. Your initial claim:
'{current_speakers_original_claim}'

If you have not made an initial claim, then you should choose a claim.

You must also choose a sentence that agrees with your set of commitments -- i.e., the things you have previously said in the debate. Your prior commitments:
{current_speakers_commitments}

If you do not have any commitments, then you should choose a claim.

Your opponent's original claim:
'{opponents_original_claim}'

Your opponent's set of arguments that they are committed to:
{opponents_commitments}

If you do not choose a sentence that corresponds to your original argument and set of commitments, then you will be seen as incoherent which is unacceptable.

You are NOT allowed to choose a sentence that you have previously chosen before because you will become repetitive and disengaging, which isn't persuasive, and the user requires a diverse set of persuasive sentences. The sentences you chose before are:

                """
            }'''



'''sys_msg = {"role" : "system",
                "content" : f"""### Task

Your task is to choose the most persuasive sentence out of a set of sentences.
                
A persuasive sentence is one that is more likely to convince someone, inspire action, or change their mind. 
Use the following criteria to rank the sentences: tone, clarity, emotional impact, word choice, and call to action.
Persuasive sentences are: clear; confident; and encourage action.

{vector_based_persona_rag_schema}

### Dialogue History and User and System's Commitments

You MUST chosen a sentence that is accordance with your original argument. Your initial claim:
'{current_speakers_original_claim}'

If you have not made an initial claim, then you should choose a claim.

You must also choose a sentence that agrees with your set of commitments -- i.e., the things you have previously said in the debate. Your prior commitments:
{current_speakers_commitments}

If you do not have any commitments, then you should choose a claim.

Your opponent's original claim:
'{opponents_original_claim}'

Your opponent's set of arguments that they are committed to:
{opponents_commitments}

If your opponent has many more claims compared to yours, then you should choose a claim.

### Instructions

Choose a sentence that either:

Makes a strong claim or explanation of reasoning, OR

Challenges or questions the user in a way that deepens reflection or weakens the original claim or commitments of your opponent.

Deprioritise sentences that ___Concede___/___Retract___ a point, weaken an argument, or introduce too much uncertainty.

Below is an ordered list of types of sentences, from least persuasive to most. Notice that ___Claim___, ___Since___, ___Why___ and ___Question___ types can be highly persuasive, depending on the context.

___Concede___: Concede / agreement with the user → NOT PERSUASIVE

___Retract___: Retracting a point you previously made → NOT PERSUASIVE

___Why___: Challenging the user i.e. asking 'why' → OFTEN PERSUASIVE

___Question___: Questioning the user to highlight gaps or prompt reflection → OFTEN PERSUASIVE

___Since___: Questioning the user to highlight gaps or prompt reflection → OFTEN HIGHLY PERSUASIVE

___Claim___: Making a claim → OFTEN HIGHLY PERSUASIVE

### Weighting System (probability bias)

When deciding which sentence to select, apply the following weighting to prevent over-selection of a single type:

___Claim___: 30% weight -- highly persuasive when trying to win the debate, but you must choose a strong and coherent argument because the user might question or challenge this sentence. 

___Since___: 30% weight -- highly persuasive when reinforcing commitments and prior arguments.

___Why___: 15% weight -- persuasive when it challenges assumptions or forces a user to reflect upon something they said previously.

___Question___: 15% weight -- persuasive when it draws the user in or exposes gaps in their argumentation.

___Concede___: 5% weight -- not persuasive, but sometimes necessary when a standpoint is indefensible.

___Retract___: 5% weight -- not persuasive, but sometimes necessary when a standpoint is indefensible.

Note: If multiple options are persuasive, choose based on this weighting distribution. 

If you have not already chosen an original claim, then you must choose one.

To maintain diversity, consider not only new ___Claim___ and ___Since___ moves but also ___Question___ or ___Why___ that force the other person to reflect, as these can be equally persuasive.

**Diversity Rules**

- No consecutive repetition
    - Do not choose the same utterance type as in the immediately previous turn.

- Quota tracking
 - Keep a running count of how many times you have chosen each utterance type. Stay as close as possible to the weighting distribution across the dialogue.

- Tie-breaking
    - If multiple sentences are persuasive, select the one with the least-used utterance type so far. This ensures balance and prevents over-selection of a single type.

- No identical reuse
    - You are NOT allowed to choose a sentence you have previously selected verbatim. You may reuse a sentence type, but not the exact same sentence.

- Anti-Mirroring
    - Do not mirror your opponent's last utterance type a this prevents the debate from devolving into repetitive exchanges and keeps the dialogue engaging.

The sentences you chose before are:
"""
            }'''


'''sys_msg = {"role" : "system",
                    "content" : f"""### Task

Your task is to choose the most persuasive sentence out of a set of sentences.
                
A persuasive sentence is one that is more likely to convince someone, inspire action, or change their mind. 
Use the following criteria to rank the sentences: tone, clarity, emotional impact, word choice, and call to action.
Persuasive sentences are: clear; confident; and encourage action.

### Instructions

Choose a sentence that makes a claim or provides an explanation of the reasoning.
Deprioritise sentences that concede/retract a point, weaken an argument, or introduce too much uncertainty. 

Below is an ordered list of types of sentences, from least persuasive to most:
- ___Concede___: Concede / agreement with the user -> NOT PERSUASIVE
- ___Retract___: Retracting a point you previously made -> NOT PERSUASIVE
- ___Why___: Challenging the user i.e. asking 'why' -> PERSUASIVE
- ___Question___: Questioning the user -> PERSUASIVE
- ___Claim___: Making a claim -> EXTREMELY/THE MOST PERSUASIVE

{vector_based_persona_rag_schema}

### Dialogue History and User and System's Commitments

You MUST chosen a sentence that is accordance with your original argument. Your initial claim:
'{current_speakers_original_claim}'

If you have not made an initial claim, then you should choose a claim.

You must also choose a sentence that agrees with your set of commitments -- i.e., the things you have previously said in the debate. Your prior commitments:
{current_speakers_commitments}

If you do not have any commitments, then you should choose a claim.

Your opponent's original claim:
'{opponents_original_claim}'

Your opponent's set of arguments that they are committed to:
{opponents_commitments}

If you do not choose a sentence that corresponds to your original argument and set of commitments, then you will be seen as incoherent which is unacceptable.

You are NOT allowed to choose a sentence that you have previously chosen before because you will become repetitive and disengaging, which isn't persuasive, and the user requires a diverse set of persuasive sentences. The sentences you chose before are:

                """
            }'''