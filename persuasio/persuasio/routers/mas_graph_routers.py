from typing import Literal

from persuasio.states.state import GenerationAgentsState
from persuasio.utils.logs import log_function

@log_function
def check_whether_models_starts_the_dialogue(state: GenerationAgentsState) -> Literal["InitialClaimGenerationAgent", 
                                                                                      "UtteranceClassificationAgent"]:
                                                                                        
    
    if len(state["dialogue_history"]) == 0:
        return "InitialClaimGenerationAgent"
    else:
        return "UtteranceClassificationAgent"


@log_function
def llm_response_router(state: GenerationAgentsState) -> Literal["SinceClaimGenerationAgent",
                                                                "ClaimNegationGenerationAgent",
                                                                "ClaimGenerationAgent",
                                                                "WhyClaimGenerationAgent",
                                                                "QuestionClaimGenerationAgent",
                                                                "ConcedeClaimGenerationAgent",
                                                                "RetractClaimGenerationAgent"]:
                                
        which_funcs_to_call = []

        claim_responses = frozenset([
            "ClaimNegationGenerationAgent", 
            "WhyClaimGenerationAgent", 
            "ConcedeClaimGenerationAgent", 
            "QuestionClaimGenerationAgent"
            ])                                          # Claim    ->   claim negation, why claim, concede claim, question claim
        why_responses = frozenset([
            "SinceClaimGenerationAgent", 
            "ClaimGenerationAgent", 
            "RetractClaimGenerationAgent"
            ])                                          # Why      ->   claim since claim, claim, retract claim
        claim_since_claim_responses = frozenset([
            "WhyClaimGenerationAgent", 
            "ConcedeClaimGenerationAgent", 
            "QuestionClaimGenerationAgent"
            ])                                          # Since    ->   why claim, concede claim, question claim
        question_claim_responses = frozenset([
            "ClaimGenerationAgent", 
            "ClaimNegationGenerationAgent", 
            "RetractClaimGenerationAgent"
            ])                                          # Question ->   claim, claim negation, retract claim
        
        for sentence in state["opponents_utterance_with_corresponding_types"]:
            utterance_type, _ = sentence[0], sentence[1]
            if utterance_type == "___Claim___":
                which_funcs_to_call.extend(list(claim_responses))
            elif utterance_type == "___Why___":
                which_funcs_to_call.extend(list(why_responses))
            elif utterance_type == "___Since___":
                which_funcs_to_call.extend(list(claim_since_claim_responses))
            elif utterance_type == "___Question___":
                which_funcs_to_call.extend(list(question_claim_responses))

        return list(set(which_funcs_to_call))