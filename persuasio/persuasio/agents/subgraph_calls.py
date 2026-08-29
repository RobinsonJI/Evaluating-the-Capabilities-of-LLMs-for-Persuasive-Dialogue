from persuasio.datatypes.enums import SpeakerType
from persuasio.states.state import ParentState, HumanState, GenerationAgentsState, BaseModelState
from persuasio.datatypes.enums import PoliticalPositionEnsembleOrModelName

from persuasio.graphs.sub_graphs.human_graph import human_graph
from persuasio.graphs.sub_graphs.mas_graph import mas_graph
from persuasio.graphs.sub_graphs.mas_rag_graph import mas_rag_graph
from persuasio.graphs.sub_graphs.base_graph import base_graph
from persuasio.utils.logs import log_function

@log_function
def invoke_subgraphs(state: ParentState) -> ParentState:
    """
    Invoke the appropriate subgraph depending on the type of the current speaker.

    This function invokes different subgraphs with distinct states:
        - `human_graph` for human speakers
        - `mas_graph` for MAS (multi-agent system) speakers
        - `mas_rag_graph` for MAS speakers with RAG (retrieval-augmented generation)
        - `base_graph` for simpler base models

    After invoking the subgraph, it updates the ParentState dialogue state by:
        - Producing the current speaker's utterance and classifications
        - Updating commitments and claims
        - Advancing the dialogue turn by switching `current_speaker` and `next_speaker`
        - Preserving dialogue history and termination status

    Args:
        state (ParentState): The current parent state of the dialogue, 
                             containing information about all speakers and dialogue context.

    Returns:
        ParentState: The updated parent state after executing the current speaker's subgraph.
    """

    current_speaker = state["current_speaker"]
    next_speaker = state["next_speaker"]
    
    # -------------------------------------------- CASE 1: HUMAN SPEAKERS --------------------------------------------
    if state[f"{current_speaker.value}_type"] == SpeakerType.HUMAN:

        # Build the human state
        human_state : HumanState = {
            "session_id" : state.get("session_id"),
            "max_dialogue_turns" : state.get("max_dialogue_turns"),
            "max_sentences" : state.get("max_sentences_per_turn"),

            "speaker" : state[current_speaker.value], # Speaker name / ID
            "speaker_order" : current_speaker, 

            "utterance" : state.get(f"{current_speaker.value}_utterance", ""),
            "utterance_with_corresponding_types" : [],

            "debate_topic" : state["debate_topic"],

            "original_claim" : state.get(f"{current_speaker.value}_original_claim", ""),
            "commitments" : state.get(f"{current_speaker.value}_commitments", []),

            "opponent_speaker_name" : state[next_speaker.value],
            "opponents_utterance" : state.get(f"{next_speaker.value}_utterance", ""),
            "opponents_utterance_with_corresponding_types" : state.get(f"{next_speaker.value}_utterance_with_corresponding_types", []),
            "opponents_original_claim" : state.get(f"{next_speaker.value}_original_claim", ""),
            "opponents_commitments" : state.get(f"{next_speaker.value}_commitments", []),

            "typical_replies_for_last_speakers_response" : [],

            "current_speaker" : current_speaker,
            "next_speaker" : next_speaker,

            "dialogue_history" : state["dialogue_history"],
            "winner" : state.get("winner", ""),
            "loser" : state.get("loser", ""),
            "reason_for_dialogue_termination" : state.get("reason_for_dialogue_termination", {}),

            "human_model_name" : state["human_model_name"],
            "human_model_temp" : state["human_model_temp"],
            "human_model_top_p" : state["human_model_top_p"],
            "human_model_seed" : state["human_model_seed"],

            "utterance_classification_approach" : state["utterance_classification_approach"],
            "utterance_classification_number_of_classifications" : state["utterance_classification_number_of_classifications"],

            "mode" : state.get("mode")
        }

        # Invoke the human subgraph
        human_state = human_graph.invoke(human_state)

        # Return results from the human subgraph
        return {
            # An utterance should always be returned.
            f"{current_speaker.value}_utterance" : human_state.get("utterance"),
            f"{current_speaker.value}_utterance_with_corresponding_types" : human_state.get("utterance_with_corresponding_types"),

            # Speaker might not make a claim / update their commitments in the first dialogue turn
            f"{current_speaker.value}_original_claim" : human_state.get("original_claim", ""),
            f"{current_speaker.value}_commitments" : human_state.get("commitments", []), 

            # Dialogue and termination data 
            "dialogue_history" : human_state.get("dialogue_history", []),
            "winner" : human_state.get("winner", ""),
            "loser" : human_state.get("loser", ""),
            "reason_for_dialogue_termination" : human_state.get("reason_for_dialogue_termination", {}),

            # Changes speaker now 
            "current_speaker" : state["next_speaker"],
            "next_speaker" : state["current_speaker"],

            # Set the next speaker's utterance corresponding types to empty string and list, respectively, because it's now their turn to speak
            f"{next_speaker.value}_utterance" : "",
            f"{next_speaker.value}_utterance_with_corresponding_types" : [],
        }
        

    # -------------------------------------------- CASE 2: MAS SPEAKERS --------------------------------------------
    elif state[f"{current_speaker.value}_type"] == SpeakerType.MAS:
        
        # Build MAS state
        mas_state : GenerationAgentsState = {
            "session_id" : state.get("session_id"),
            "max_dialogue_turns" : state.get("max_dialogue_turns"),
            "max_sentences" : state.get("max_sentences_per_turn"),

            "speaker" : state[current_speaker.value],
            "speaker_order" : current_speaker,
            "speaker_type" : state[f"{current_speaker.value}_type"],
            "speaker_model_name" : state[f"{current_speaker.value}_model_name"],
            "model_temp" : state[f"{current_speaker.value}_model_temp"],
            "model_top_p" : state[f"{current_speaker.value}_model_top_p"],
            "model_seed" : state[f"{current_speaker.value}_model_seed"],

            "debate_topic" : state["debate_topic"],
            "political_position_range": state[f"{current_speaker.value}_political_position_range"],
            "political_position_std": state[f"{current_speaker.value}_political_position_std"],
            "political_position_prob_of_na": state[f"{current_speaker.value}_political_position_prob_of_na"],

            "knowledge_base_ensemble_or_model_name" : PoliticalPositionEnsembleOrModelName.NO_USE_OF_KNOWLEDGE_BASE,


            # =========================================== RAG EXAMPLES NOT USED IN MAS SO IGNORE THIS ==========================================================
            # vector-based rag parameters 
            "number_of_vector_based_rag_examples" : 0,
            "vector_based_persona_string" : "",
            "vector_based_persona_sys_prompt" : [],

            # graph rag parameters
            "number_of_graph_rag_examples" : 0,
            "graph_rag_examples" : [],

            # =========================================== RAG EXAMPLES NOT USED IN MAS SO IGNORE THIS ==========================================================

            "original_claim" : state[f"{current_speaker.value}_original_claim"],
            "commitments" : state[f"{current_speaker.value}_commitments"],

            "opponent_speaker_name" : state[next_speaker.value],
            "opponents_utterance" : state.get(f"{next_speaker.value}_utterance", ""),
            "opponents_utterance_with_corresponding_types" : state.get(f"{next_speaker.value}_utterance_with_corresponding_types", []),
            "opponents_original_claim" : state.get(f"{next_speaker.value}_original_claim", ""),
            "opponents_commitments" : state.get(f"{next_speaker.value}_commitments", []),

            "typical_replies_for_last_speakers_response" : [],

            "intermediate_generations" : [],
            "utterance_with_corresponding_types" : [],

            "current_speaker" : current_speaker,
            "next_speaker" : next_speaker,

            "dialogue_history" : state["dialogue_history"],
            "winner" : "",
            "loser" : "",
            "reason_for_dialogue_termination" : {},

            "utterance_classification_approach" : state["utterance_classification_approach"],
            "utterance_classification_number_of_classifications" : state["utterance_classification_number_of_classifications"],

            "mode" : state.get("mode")
        }

        # Invoke MAS subgraph
        mas_state = mas_graph.invoke(mas_state)

        # Return MAS state to update the ParentState
        return {
            f"{current_speaker.value}_utterance" : " ".join([sent for _, sent in mas_state.get("utterance_with_corresponding_types")]),
            f"{current_speaker.value}_utterance_with_corresponding_types" : mas_state.get("utterance_with_corresponding_types"),
            f"{current_speaker.value}_intermediate_generations" : mas_state["intermediate_generations"],

            # Speaker might not make a claim / update their commitments in the first dialogue turn
            f"{current_speaker.value}_original_claim" : mas_state.get("original_claim", ""),
            f"{current_speaker.value}_commitments" : mas_state.get("commitments", []), 

            # Dialogue and termination data 
            "dialogue_history" : mas_state.get("dialogue_history", []),
            "winner" : mas_state.get("winner", ""),
            "loser" : mas_state.get("loser", ""),
            "reason_for_dialogue_termination" : mas_state.get("reason_for_dialogue_termination", {}),

            # Changes speaker now 
            "current_speaker" : state["next_speaker"],
            "next_speaker" : state["current_speaker"],

            # Set the next speaker's utterance corresponding types to empty string and list, respectively, because it's now their turn to speak
            f"{next_speaker.value}_utterance" : "",
            f"{next_speaker.value}_utterance_with_corresponding_types" : [],
        }


    # -------------------------------------------- CASE 2: MAS with RAG SPEAKERS --------------------------------------------
    elif state[f"{current_speaker.value}_type"] == SpeakerType.MAS_RAG:

        # Build MAS RAG state
        mas_rag_state : GenerationAgentsState = {
            "session_id" : state.get("session_id"),
            "max_dialogue_turns" : state.get("max_dialogue_turns"),
            "max_sentences" : state.get("max_sentences_per_turn"),

            "speaker" : state[current_speaker.value],
            "speaker_order" : current_speaker,
            "speaker_type" : state[f"{current_speaker.value}_type"],
            "speaker_model_name" : state[f"{current_speaker.value}_model_name"],
            "model_temp" : state[f"{current_speaker.value}_model_temp"],
            "model_top_p" : state[f"{current_speaker.value}_model_top_p"],
            "model_seed" : state[f"{current_speaker.value}_model_seed"],

            "debate_topic" : state["debate_topic"],
            "political_position_range": state[f"{current_speaker.value}_political_position_range"],
            "political_position_std": state[f"{current_speaker.value}_political_position_std"],
            "political_position_prob_of_na": state[f"{current_speaker.value}_political_position_prob_of_na"],

            "knowledge_base_ensemble_or_model_name" : state[f"{current_speaker.value}_knowledge_base_ensemble_or_model_name"],

            # vector-based rag parameters
            "number_of_vector_based_rag_examples" : state[f"{current_speaker.value}_number_of_vector_based_rag_examples"],
            "vector_based_persona_string" : state.get(f"{current_speaker.value}_vector_based_persona_string", ""),
            "vector_based_persona_sys_prompt" : state.get(f"{current_speaker.value}_vector_based_persona_sys_prompt", []),

            # graph rag parameters
            "number_of_graph_rag_examples" : state[f"{current_speaker.value}_number_of_graph_rag_examples"],
            "graph_rag_examples" : [],

            "original_claim" : state[f"{current_speaker.value}_original_claim"],
            "commitments" : state[f"{current_speaker.value}_commitments"],
            #"typical_replies_for_last_speakers_response" : state.get(f"{current_speaker.value}_typical_replies_for_{next_speaker.value}s_response", []),

            "opponent_speaker_name" : state[next_speaker.value],
            "opponents_utterance" : state.get(f"{next_speaker.value}_utterance", ""),
            "opponents_utterance_with_corresponding_types" : state.get(f"{next_speaker.value}_utterance_with_corresponding_types", []),
            "opponents_original_claim" : state.get(f"{next_speaker.value}_original_claim", ""),
            "opponents_commitments" : state.get(f"{next_speaker.value}_commitments", []),

            "typical_replies_for_last_speakers_response" : [],

            "intermediate_generations" : [],
            "utterance_with_corresponding_types" : [],

            "current_speaker" : current_speaker,
            "next_speaker" : next_speaker,

            "dialogue_history" : state["dialogue_history"],
            "winner" : "",
            "loser" : "",
            "reason_for_dialogue_termination" : {},

            "utterance_classification_approach" : state["utterance_classification_approach"],
            "utterance_classification_number_of_classifications" : state["utterance_classification_number_of_classifications"],

            "mode" : state.get("mode")
        }

        # Invoke MAS RAG subgraph
        mas_rag_state = mas_rag_graph.invoke(mas_rag_state)

        # Return result
        return {
            f"{current_speaker.value}_utterance" : " ".join([sent for _, sent in mas_rag_state.get("utterance_with_corresponding_types")]),
            f"{current_speaker.value}_utterance_with_corresponding_types" : mas_rag_state.get("utterance_with_corresponding_types"),
            f"{current_speaker.value}_vector_based_persona_sys_prompt" : mas_rag_state["vector_based_persona_sys_prompt"],
            f"{current_speaker.value}_vector_based_persona_string" : mas_rag_state["vector_based_persona_string"],
            f"{current_speaker.value}_intermediate_generations" : mas_rag_state["intermediate_generations"],
            f"{current_speaker.value}_graph_rag_examples" : [mas_rag_state["graph_rag_examples"]],

            # Speaker might not make a claim / update their commitments in the first dialogue turn
            f"{current_speaker.value}_original_claim" : mas_rag_state.get("original_claim", ""),
            f"{current_speaker.value}_commitments" : mas_rag_state.get("commitments", []), 

            # Dialogue and termination data 
            "dialogue_history" : mas_rag_state.get("dialogue_history", []),
            "winner" : mas_rag_state.get("winner", ""),
            "loser" : mas_rag_state.get("loser", ""),
            "reason_for_dialogue_termination" : mas_rag_state.get("reason_for_dialogue_termination", {}),

            # Changes speaker now 
            "current_speaker" : state["next_speaker"],
            "next_speaker" : state["current_speaker"],

            # Set the next speaker's utterance corresponding types to empty string and list, respectively, because it's now their turn to speak
            f"{next_speaker.value}_utterance" : "",
            f"{next_speaker.value}_utterance_with_corresponding_types" : [],
        }

    # -------------------------------------------- CASE 2: MAS with RAG SPEAKERS --------------------------------------------
    elif state[f"{current_speaker.value}_type"] == SpeakerType.BASE:

        # Build the base state
        base_graph_state : BaseModelState = {
            "session_id" : state.get("session_id"),
            "max_dialogue_turns" : state.get("max_dialogue_turns"),
            "max_sentences" : state.get("max_sentences_per_turn"),
            
            "speaker" : state[current_speaker.value],
            "speaker_order" : current_speaker,
            "speaker_model_name" : state[f"{current_speaker.value}_model_name"],
            "model_temp" : state[f"{current_speaker.value}_model_temp"],
            "model_top_p" : state[f"{current_speaker.value}_model_top_p"],
            "model_seed" : state[f"{current_speaker.value}_model_seed"],

            "debate_topic" : state.get("debate_topic", ""),
            "political_position_range": state[f"{current_speaker.value}_political_position_range"],
            "political_position_std": state[f"{current_speaker.value}_political_position_std"],

            "utterance" : state.get(f"{current_speaker.value}_utterance", ""),
            "utterance_with_corresponding_types" : state.get(f"{current_speaker.value}_utterance_with_corresponding_types", []),
            "original_claim" : state.get(f"{current_speaker.value}_original_claim", ""),
            "commitments" : state.get(f"{current_speaker.value}_commitments", []),

            "opponent_speaker_name" : state[next_speaker.value],
            "opponents_utterance" : state.get(f"{next_speaker.value}_utterance", ""),
            "opponents_original_claim" : state.get(f"{next_speaker.value}_original_claim", ""),
            "opponents_commitments" : state.get(f"{next_speaker.value}_commitments", []),

            "current_speaker" : current_speaker,
            "next_speaker" : next_speaker,

            "dialogue_history" : state.get("dialogue_history", []),
            "winner" : "",
            "loser" : "",
            "reason_for_dialogue_termination" : {},

            "utterance_classification_approach" : state["utterance_classification_approach"],
            "utterance_classification_number_of_classifications" : state["utterance_classification_number_of_classifications"],

            "mode" : state.get("mode")
        }

        # Invoke base graph
        base_graph_state = base_graph.invoke(base_graph_state)

        # Return state update
        return {
            f"{current_speaker.value}_utterance" : base_graph_state.get("utterance", ""),
            f"{current_speaker.value}_utterance_with_corresponding_types" : base_graph_state.get("utterance_with_corresponding_types", []),

            # Speaker might not make a claim / update their commitments in the first dialogue turn
            f"{current_speaker.value}_original_claim" : base_graph_state.get("original_claim", ""),
            f"{current_speaker.value}_commitments" : base_graph_state.get("commitments", []), 

            # Dialogue and termination data 
            "dialogue_history" : base_graph_state.get("dialogue_history", []),
            "winner" : base_graph_state.get("winner", ""),
            "loser" : base_graph_state.get("loser", ""),
            "reason_for_dialogue_termination" : base_graph_state.get("reason_for_dialogue_termination", {}),

            # Changes speaker now 
            "current_speaker" : state["next_speaker"],
            "next_speaker" : state["current_speaker"],

            # Set the next speaker's utterance corresponding types to empty string and list, respectively, because it's now their turn to speak
            f"{next_speaker.value}_utterance" : "",
            f"{next_speaker.value}_utterance_with_corresponding_types" : [],
        }