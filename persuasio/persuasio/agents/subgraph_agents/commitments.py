from typing import TypeVar
from persuasio.states.state import HumanState, GenerationAgentsState, BaseModelState
# from persuasio.tools.check_state import is_state
from persuasio.tools.check_state import is_instance_of_typed_dict

from persuasio.models.models import GenerateLLMResponses
from persuasio.prompts.generators.commitments import (
    is_utterance_similar_to_any_commitments, 
    utterance_concedes_to_opponents_commitments, 
    sentence_concede_to_initial_claim,
    does_utterance_retract_initial_claim,
    does_utterance_retract_any_commitments,
    find_retracted_commitments)
from persuasio.datatypes.pydantic_basemodels import DataForOneDialogueTurn, IsSimilar, Concedes, Retracts, RetractedSentences
from persuasio.datatypes.enums import LogLevels
from persuasio.utils.logs import log_function, log

T = TypeVar("T", HumanState, GenerationAgentsState, BaseModelState)

@log_function
def commitment_update(state : T) -> T:
    """
    This function conducts a commitment update based upon the commitment rules on Page 7 of Formal Systems for Persuasion Dialogues (Prakken, 2006). The rules go as follows:
    
    - If s(m) = ___Claim___ \\psi, then human_commitments = {previous_commitments} \\union {\\psi};
    - If s(m) = ___Why___ \\psi, then the human_commitments remain unchanged;
    - If s(m) = ___Concede____ \\psi, then human_commitments = {previous_commitments} \\union {\\psi};
    - If s(m) = ___Retract___, human_commitments = {previous_commitments} \\setminus {\\psi};
    - If s(m) = \\psi ___Since___ S, then human_commitments = {previous_commitments (including \\psi)} \\union {S}.
    
    where s refers to the speaker, which in this case is the human, and m refers to the last dialogue move (or utterance) the human made.
    
    This function can also be employed to monitor whether the dialogue should terminate. This dialogue game will end if and only if:
    - The current speaker makes a ___Retract___ move, where they withdraw their original claim; or 
    - The current speaker makes a ___Concede___ move, where the current speaker accepts/admits that the opponent's original claim is the case. 
    
    For more information read Pg 6-7 of Formal Systems for Persuasion Dialogues by H. Prakken (2006).

    Parameters
    ----------
    state : T
        A dialogue state object (`HumanState`, `GenerationAgentsState`, or `BaseModelState`) 
        containing dialogue metadata, commitments, and configuration.
    
    Returns
    -------
    T
        An updated state-like dictionary with:
        - Updated commitments
        - Dialogue history
        - Dialogue termination status (`winner`, `loser`, `reason_for_dialogue_termination`)"""

    # ==================================================================== MODEL SELECTION =========================================================================

    # check whether state is HumanState
    if is_instance_of_typed_dict(state, HumanState):
        model = state["human_model_name"]
        temp = state["human_model_temp"]
        top_p = state["human_model_top_p"]
        seed = state["human_model_seed"]

    elif (is_instance_of_typed_dict(state, GenerationAgentsState)) or (is_instance_of_typed_dict(state, BaseModelState)):
        # State will be either of type GenerationAgentsState or BaseModelState
        model = state["speaker_model_name"]
        temp = state["model_temp"]
        top_p = state["model_top_p"]
        seed = state["model_seed"]

    # ==================================================================== SPEAKER SET UP =========================================================================

    current_speaker = state["speaker"]
    next_speaker = state["opponent_speaker_name"]

    current_speaker_original_claim = state[f"original_claim"]
    current_speaker_commitments = state[f"commitments"]

    next_speaker_original_claim = state[f"opponents_original_claim"]
    next_speaker_commitments = state[f"opponents_commitments"]

    # Initialise termination tracking
    winner = ''
    loser = ''
    reason_for_termination = {}

    # ==================================================================== PROCESS UTTERANCES =========================================================================

    for sentence in state.get(f"utterance_with_corresponding_types"):
        utterance_type, utterance_text = sentence[0], sentence[1]
        
        # ------------------------------------------------------------------------------- CLAIM MOVES -------------------------------------------------------------------------------
        if utterance_type == "___Claim___":
            # Set original claim if this is first claim made
            if len(current_speaker_original_claim) == 0:
                current_speaker_original_claim = utterance_text

            # Only add utterance to commitments if not semantically similar to existing commitments
            prompt = is_utterance_similar_to_any_commitments(utterance=utterance_text, commitments=current_speaker_commitments)

            try:
                result = GenerateLLMResponses(model_choice=model,
                                            prompt=prompt,
                                            temperature=temp,
                                            top_p=top_p,
                                            seed=seed,
                                            datatype_schema=IsSimilar).return_completion()
                if not result["similar"]:
                    current_speaker_commitments.append(utterance_text)

                log(
                    session_id=state["session_id"],
                    level=LogLevels.INFO,
                    service=commitment_update.__name__,
                    message=f"'commitment_update' completed '___Claim___' update for current speaker's commitments; MODEL = '{model.value}'.",
                    mode=state["mode"]
                )
            except ValueError as e:
                log(
                    session_id=state["session_id"],
                    level=LogLevels.ERROR,
                    service=commitment_update.__name__,
                    message=f"'commitment_update' failed at '___Claim___' update for current speaker's commitments, completion not validated; MODEL = '{model.value}'; \n REASON: \n\n {e}",
                    mode=state["mode"],
                    context={"prompt" : prompt, "state":state, "exception":e}
                )

        # ------------------------------------------------------------------------------- CONCEDE MOVES -------------------------------------------------------------------------------
        elif utterance_type == "___Concede___":
            # Add conceded proposition to commitments (if not already present)
            if utterance_text not in current_speaker_commitments:
                # Work out whether the utterance is already the current speaker's commitments
                prompt = is_utterance_similar_to_any_commitments(utterance=utterance_text, commitments=current_speaker_commitments)
                try:
                    similar_result = GenerateLLMResponses(model_choice=model,
                                                prompt=prompt,
                                                temperature=temp,
                                                top_p=top_p,
                                                seed=seed,
                                                datatype_schema=IsSimilar).return_completion()
                    log(
                        session_id=state["session_id"],
                        level=LogLevels.INFO,
                        service=commitment_update.__name__,
                        message=f"'commitment_update' completed '___Concede___' update when checking if concede utterance was similar to current speaker's commitments; MODEL = '{model.value}'.",
                        mode=state["mode"]
                    )
                except ValueError as e:
                    log(
                        session_id=state["session_id"],
                        level=LogLevels.ERROR,
                        service=commitment_update.__name__,
                        message=f"'commitment_update' failed at '___Concede___' update when checking if concede utterance was similar to current speaker's commitments; MODEL = '{model.value}'; \n REASON: \n\n {e}",
                        mode=state["mode"],
                        context={"prompt" : prompt, "state":state, "exception":e}
                    )
                
                # Check if the utterance concedes to the opponent's commitments
                prompt = utterance_concedes_to_opponents_commitments(utterance=utterance_text, commitments=next_speaker_commitments)

                try:
                    result = GenerateLLMResponses(model_choice=model,
                                                prompt=prompt,
                                                temperature=temp,
                                                top_p=top_p,
                                                seed=seed,
                                                datatype_schema=Concedes).return_completion()
                    
                    # Model identifies utterance as a concede move
                    if result["concede"] and (not similar_result["similar"]):
                        current_speaker_commitments.append(utterance_text)

                    log(
                        session_id=state["session_id"],
                        level=LogLevels.INFO,
                        service=commitment_update.__name__,
                        message=f"'commitment_update' completed '___Concede___' update when checking if current speaker conceded to opponent's intial claim; MODEL = '{model.value}'.",
                        mode=state["mode"]
                    )
                except ValueError as e:
                    log(
                        session_id=state["session_id"],
                        level=LogLevels.ERROR,
                        service=commitment_update.__name__,
                        message=f"'commitment_update' failed at '___Concede___' update when checking if current speaker conceded to opponent's intial claim, completion not validated; MODEL = '{model.value}'; \n REASON: \n\n {e}",
                        mode=state["mode"],
                        context={"prompt" : prompt, "state":state, "exception":e}
                    )

                #LLM to check whether utterance concedes to opponent's original claim. If it does, add utterance to speaker's commitments and dialogue will end.
                prompt = sentence_concede_to_initial_claim(utterance=utterance_text, opponents_initial_claim=next_speaker_original_claim)

                try:
                    result = GenerateLLMResponses(model_choice=model,
                                            prompt=prompt,
                                            temperature=temp,
                                            top_p=top_p,
                                            seed=seed,
                                            datatype_schema=Concedes).return_completion()
                    
                    if result["concede"]:
                        if (not similar_result["similar"]):
                            current_speaker_commitments.append(utterance_text)
                        
                        # Dialogue termination
                        winner = next_speaker
                        loser = current_speaker
                        reason_for_termination = {
                            "reason" : f"{loser} conceded to {winner}'s original claim, where {winner}'s original claim was: '{next_speaker_original_claim}'",
                            "utterance_type" : utterance_type,
                            "utterance" : utterance_text, 
                            }
                    
                    log(
                        session_id=state["session_id"],
                        level=LogLevels.INFO,
                        service=commitment_update.__name__,
                        message=f"'commitment_update' completed '___Concede___' update when checking whether to terminate dialogue; MODEL = '{model.value}'.",
                        mode=state["mode"]
                    )
                except ValueError as e:
                    log(
                        session_id=state["session_id"],
                        level=LogLevels.ERROR,
                        service=commitment_update.__name__,
                        message=f"'commitment_update' failed at '___Concede___' update when checking whether to terminate dialogue, completion not validated; MODEL = '{model.value}'; \n REASON: \n\n {e}",
                        mode=state["mode"],
                        context={"prompt" : prompt, "state":state, "exception":e}
                    )

        # ------------------------------------------------------------------------------- RETRACT MOVES -------------------------------------------------------------------------------
        elif utterance_type == "___Retract___":
            # check if utterance retracts a commitment
            prompt = does_utterance_retract_any_commitments(utterance=utterance_text, commitments=current_speaker_commitments)

            try:
                result = GenerateLLMResponses(model_choice=model,
                                            prompt=prompt,
                                            temperature=temp,
                                            top_p=top_p,
                                            seed=seed,
                                            datatype_schema=Retracts).return_completion()
                
                if result["retract"]:
                    # We now need to find which commitments the current speaker is retracting.
                    prompt = find_retracted_commitments(utterance=utterance_text, commitments=current_speaker_commitments)

                    result = GenerateLLMResponses(model_choice=model,
                                            prompt=prompt,
                                            temperature=temp,
                                            top_p=top_p,
                                            seed=seed,
                                            datatype_schema=RetractedSentences).return_completion()
                    
                    for retraction in result["retracted_sentences"]:
                        current_speaker_commitments.remove(retraction)

                log(
                    session_id=state["session_id"],
                    level=LogLevels.INFO,
                    service=commitment_update.__name__,
                    message=f"'commitment_update' completed '___Retract___' update for current speaker's commitments; MODEL = '{model.value}'.",
                    mode=state["mode"]
                )
            except ValueError as e:
                log(
                    session_id=state["session_id"],
                    level=LogLevels.ERROR,
                    service=commitment_update.__name__,
                    message=f"'commitment_update' failed at '___Retract___' update for current speaker's commitments, completion not validated; MODEL = '{model.value}'; \n REASON: \n\n {e}",
                    mode=state["mode"],
                    context={"prompt" : prompt, "state":state, "exception":e}
                )

            # Check if initial claim is retracted
            prompt = does_utterance_retract_initial_claim(utterance=utterance_text, current_speakers_initial_claim=current_speaker_original_claim)

            try:
                result = GenerateLLMResponses(model_choice=model,
                                            prompt=prompt,
                                            temperature=temp,
                                            top_p=top_p,
                                            seed=seed,
                                            datatype_schema=Retracts).return_completion()
                    
                if result["retract"]:
                    if utterance_text in current_speaker_commitments:
                        # Remove the commitments from the current speaker's set of commitments.
                        current_speaker_commitments.remove(utterance_text)
                    
                    # Dialogue termination 
                    winner = next_speaker
                    loser = current_speaker
                    reason_for_termination = {
                        "reason" : f"{loser} retracted their original claim, where original_{loser}_claim: '{current_speaker_original_claim}'",
                        "utterance_type" : utterance_type,
                        "utterance" : utterance_text, 
                        }
                    
                log(
                    session_id=state["session_id"],
                    level=LogLevels.INFO,
                    service=commitment_update.__name__,
                    message=f"'commitment_update' completed '___Retract___' update when checking whether to terminate dialogue; MODEL = '{model.value}'.",
                    mode=state["mode"]
                )
            except ValueError as e:
                log(
                    session_id=state["session_id"],
                    level=LogLevels.ERROR,
                    service=commitment_update.__name__,
                    message=f"'commitment_update' failed at '___Retract___' update when checking whether to terminate dialogue, completion not validated; MODEL = '{model.value}'; \n REASON: \n\n {e}",
                    mode=state["mode"],
                    context={"prompt" : prompt, "state":state, "exception":e}
                )

        # ------------------------------------------------------------------------------- SINCE MOVES -------------------------------------------------------------------------------
        elif utterance_type == "___Since___":
            # Add premises introduced by ___Since___ unless already similar to commitments
            if utterance_text not in current_speaker_commitments:
                prompt = is_utterance_similar_to_any_commitments(utterance=utterance_text, commitments=current_speaker_commitments)

                try:
                    result = GenerateLLMResponses(model_choice=model,
                                                prompt=prompt,
                                                temperature=temp,
                                                top_p=top_p,
                                                seed=seed,
                                                datatype_schema=IsSimilar).return_completion()
                    
                    if not result["similar"]:
                        current_speaker_commitments.append(utterance_text)
                    log(
                        session_id=state["session_id"],
                        level=LogLevels.INFO,
                        service=commitment_update.__name__,
                        message=f"'commitment_update' completed '___Since___' update; MODEL = '{model.value}'.",
                        mode=state["mode"]
                    )
                except ValueError as e:
                    log(
                        session_id=state["session_id"],
                        level=LogLevels.ERROR,
                        service=commitment_update.__name__,
                        message=f"'commitment_update' failed at '___Since___' update, completion not validated; MODEL = '{model.value}'; \n REASON: \n\n {e}",
                        mode=state["mode"],
                        context={"prompt" : prompt, "state":state, "exception":e}
                    )

    # =============================================================================== Update dialogue history =======================================================================
    dialogue_turn_data = DataForOneDialogueTurn(speaker=current_speaker, 
                                                sentences_with_utterance_types=state[f"utterance_with_corresponding_types"], 
                                                sentences_no_utterance_types=" ".join([sent for _, sent in state[f"utterance_with_corresponding_types"]]))
    
    dialogue_history = state["dialogue_history"]
    dialogue_history.append(dialogue_turn_data)

    # =============================================================================== Check max turns ==============================================================================
    if (len(dialogue_history) == state.get("max_dialogue_turns")) and (len(winner) == 0) and (len(loser) == 0) and (len(reason_for_termination) == 0):
        winner = "None"
        loser = "None"
        reason_for_termination = {
            "reason" : f"Maximum number of dialogue turns reached ({state['max_dialogue_turns']} dialogue turns).",
            "utterance_type" : None,
            "utterance" : None, 
            }  

    # =============================================================================== Return updated commitments ======================================================================
    return {
        f"original_claim" : current_speaker_original_claim,
        f"commitments" : current_speaker_commitments, 
        
        "dialogue_history" : dialogue_history,

        "winner" : winner,
        "loser" : loser,
        "reason_for_dialogue_termination" :reason_for_termination,
            } 