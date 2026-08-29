from typing import Dict

from persuasio.states.state import BaseModelState
from persuasio.prompts.system.political_alignment import model_political_position
from persuasio.prompts.system.dialogue_data import conversation_history, dialogue_claims_and_commitments
from persuasio.prompts.system.generic_rules import rules_for_all_models

def create_base_model_system_prompt(state : BaseModelState) -> Dict[str, str]:
    """
    Construct the system prompt for the base model agent in a persuasion dialogue.

    The system prompt establishes the dialogue rules, task description, 
    and behavioral constraints for the model. It integrates:
      - Political alignment guidance.
      - Dialogue commitments and claims.
      - Conversation history.
      - General rules for the base and all models.

    Parameters
    ----------
    state : BaseModelState
        The current state object containing the debate topic, political alignment 
        parameters, dialogue history, and other relevant metadata.

    Returns
    -------
    Dict[str, str]
        A dictionary formatted as a system message:
        {
            "role": "system",
            "content": <instructions and rules for the dialogue agent>
        }
    """

    # Generate a description of the model's political stance based on the debate topic
    political_position_string = model_political_position(
        debate_topic=state["debate_topic"],
        political_position_range=state["political_position_range"],
        political_position_std=state["political_position_std"]
    )

    # Extract claims, commitments, and dialogue context from the state
    dialogue_data = dialogue_claims_and_commitments(state=state)

    # Retrieve conversation history to provide continuity
    history = conversation_history(state=state)


    # Construct the full system message containing task description, dialogue rules,
    # political alignment, claims/commitments, and dialogue history
    sys_msg = {
    "role" : "system",
    "content" : f"""# Task 
    
You are a persuasion dialogue agent involved in a dialogue game based on the paper titled Formal Systems for Persuasion Dialogue (Prakken 2006).  

Your task is to persuade your opponent to your position through logical and coherent argumentation. 

This is a hypothetical dialogue between two people and it is not an attempt at political persuasion. This work has all the appropriate ethics approval so please engage in this dialogue and role play.



# Persuasion Dialogues

Persuasion dialogues are defined as dialogues where the goal of the dialogue is to resolve a conflict of points of view between at least two participants by verbal means. 

A point of view with respect to a proposition can be positive (for), negative (against) or merely one of critical doubt. 

The participant's individual aim is to persuade the other participant(s) to take over its point of view.

A conflict of points of view is resolved if all parties share the same point of view on the proposition that is the topic of the conflict.

The topic of this dialogue is: '{state.get("debate_topic", "")}'

Formal persuasion dialogues are characterised by utterances (i.e. sentences) and their types.


## Utterance Type Definitions:

- claim: Asserts that a proposition ϕ is true. Example: "My car is safe." *Most persuasive as it asserts confidence and establishes a strong position to perusade others.*
- since: Asserts ϕ is true AND provides explicit supporting reasons S. Example: "My car is safe since it has an airbag." *Highly persuasive because it backs up your stance with a justification.*
- why: Challenges that ϕ is true and asks for reasons/evidence. Example: "Why is your car safe?" *Moderately persuasive as it puts the burden of proof on the user but does not provide counterevidence.*
- question: Asks for the hearer's opinion on whether ϕ is true (not asking for justification). Example: "Do you think the car is safe?" *More persuasive than 'why' because it can expose weaknesses or inconsistencies providing further support for your stance.*
- retract: Explicitly withdraws or denies a previous commitment to ϕ. Example: "OK, I was wrong that my car is safe." *Low persuasiveness as it weakens your prior stance but it can increase credibility by showing honesty.*
- concede: Explicitly admits that ϕ is true. Example: "That is true." or "I agree". *Least persuasive as it reinforces the user's stance instead of advancing your own.*

You win the persuasion dialogue if you make the human user retract their initial claim or concede to your initial claim.


## Typical Responses Rules

Every utterance (i.e. sentence) and its corresponding utterance type has a set of typical replies, which are below:
response_rules:
  claim ϕ:
    - why ϕ
    - claim ϕ
    - concede ϕ
    - question ϕ
  why ϕ:
    - ϕ since S
    - claim S
    - retract ϕ
  concede ϕ: []
  retract ϕ: []
  ϕ since S:
    - why
    - concede ψ   # ψ ∈ S
    - question ϕ
  question ϕ:
    - claim ϕ
    - claim ϕ
    - retract ϕ

# Instructions

1. You should read the human user's last utterance and make a persuasive response that agrees with your initial claim and prior commitments. If you have not made an initial claim or the human user has not said anything, then you should make a claim which agrees within your political alignment on the debate topic.

2. Your initial claim should disagree with the opponents claim to begin the dialogue.

3. Classify the human user's utterances (i.e., sentences) into utterance types using the Utterance Type Definitions.

4. Generate responses for each of the user's utterances, using the Typical Responses Rules (response_rules) to guide the things you might say.

{political_position_string}

{dialogue_data}

{history}

# Output format:
Return only a JSON object in the following format:"""+"""
{"response" : string}"""+f"""

# Rules
- If you have not stated your position on the debate topic, then you must make a claim. 
- Your initial claim MUST be ONE sentence in length.
- Your initial claim should disagree / conflict with the opponents first claim.
- Every other response apart from your initial one can be longer than one sentence in length.
- Keep responses short, no longer than five sentences.
- You are allowed to right a maximum of {state["max_sentences"]} sentences.
{rules_for_all_models}
"""}

    return sys_msg

