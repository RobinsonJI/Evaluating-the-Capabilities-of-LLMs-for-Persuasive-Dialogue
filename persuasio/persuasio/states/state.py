from typing_extensions import TypedDict
from typing import List, Tuple, Dict, Annotated, Any, Union
import operator

# --- IMPORT ENUMS AND DATAMODELS ---
from persuasio.datatypes.enums import SpeakerType, ModelName, SpeakerOrder, UtteranceClassificationApproach, PoliticalPositionEnsembleOrModelName, Mode
from persuasio.datatypes.pydantic_basemodels import DataForOneDialogueTurn


class ParentState(TypedDict, total=False):
    
    session_id : str
    debate_topic : str
    max_dialogue_turns : int
    max_sentences_per_turn : int
    commitments_cosine_similarity_threshold : float


    first_speaker : str
    first_speaker_type : SpeakerType
    first_speaker_model_name : ModelName
    first_speaker_model_temp : float
    first_speaker_model_top_p : float
    first_speaker_model_seed : int
    first_speaker_utterance : str
    first_speaker_utterance_with_corresponding_types : List[Tuple[str, str]]
    first_speaker_commitments : list
    first_speaker_original_claim : str
    first_speaker_typical_replies_for_second_speakers_response : List[Tuple[Dict[str, str], int, int]]
    first_speaker_political_position_range:str
    first_speaker_political_position_std:int
    first_speaker_political_position_prob_of_na:float

    first_speaker_knowledge_base_ensemble_or_model_name : PoliticalPositionEnsembleOrModelName
    first_speaker_number_of_vector_based_rag_examples : int
    first_speaker_vector_based_persona_examples : str

    first_speaker_number_of_graph_rag_examples : int
    
    second_speaker : str
    second_speaker_type : SpeakerType
    second_speaker_model_name : ModelName
    second_speaker_model_temp : float
    second_speaker_model_top_p : float
    second_speaker_model_seed : int
    second_speaker_utterance : str
    second_speaker_utterance_with_corresponding_types : List[Tuple[str, str]]
    second_speaker_commitments : list
    second_speaker_original_claim : str 
    second_speaker_typical_replies_for_first_speakers_response : List[Tuple[Dict[str, str], int, int]]
    second_speaker_political_position_range:str
    second_speaker_political_position_std:int
    second_speaker_political_position_prob_of_na:float

    second_speaker_knowledge_base_ensemble_or_model_name : PoliticalPositionEnsembleOrModelName
    second_speaker_number_of_vector_based_rag_examples : int
    second_speaker_vector_based_persona_examples : str

    second_speaker_number_of_graph_rag_examples : int
    
    current_speaker : SpeakerOrder
    next_speaker : SpeakerOrder

    dialogue_history : List[DataForOneDialogueTurn]
    winner : str 
    loser : str 
    reason_for_dialogue_termination : Dict[str, str]

    human_model_name : ModelName
    human_model_temp : float
    human_model_top_p : float
    human_model_seed : int

    utterance_classification_approach : UtteranceClassificationApproach
    utterance_classification_number_of_classifications : int

    first_speaker_intermediate_generations : Annotated[List[Tuple[Dict[str, Any], int, int]], operator.add]
    second_speaker_intermediate_generations : Annotated[List[Tuple[Dict[str, Any], int, int]], operator.add]

    first_speaker_vector_based_persona_sys_prompt : List[Dict[str, str]]
    second_speaker_vector_based_persona_sys_prompt : List[Dict[str, str]]
    first_speaker_vector_based_persona_string : str
    second_speaker_vector_based_persona_string : str

    first_speaker_graph_rag_examples : Annotated[List[Dict[str,List[str]]] , operator.add]
    second_speaker_graph_rag_examples : Annotated[List[Dict[str,List[str]]] , operator.add]

    mode : Mode


class GenerationAgentsState(TypedDict):
    session_id : str
    max_dialogue_turns : int
    max_sentences : int

    speaker : str
    speaker_order : SpeakerOrder
    speaker_type : SpeakerType
    speaker_model_name : ModelName
    model_temp : float
    model_top_p : float
    model_seed : int

    debate_topic : str
    political_position_range:str
    political_position_std:int
    political_position_prob_of_na:float

    knowledge_base_ensemble_or_model_name : PoliticalPositionEnsembleOrModelName
    
    number_of_vector_based_rag_examples : int
    vector_based_persona_string : str
    vector_based_persona_sys_prompt : List[Dict[str, str]]

    number_of_graph_rag_examples : int
    graph_rag_examples : List[Dict[str,List[str]]]

    original_claim : str 
    commitments : List[str]
    typical_replies_for_last_speakers_response : List[Tuple[Dict[str, str], int, int]]

    opponent_speaker_name : str
    opponents_utterance : str
    opponents_utterance_with_corresponding_types : List[Tuple[str, str]]
    opponents_original_claim : str
    opponents_commitments : List[str]

    #intermediate_generations : Annotated[List[Tuple[Dict[str, Any], int, int]], operator.add]
    intermediate_generations : Annotated[List[List[Union[Dict[str, List[str]], int, int]]], operator.add]
    utterance_with_corresponding_types : List[Tuple[str, str]]

    current_speaker : SpeakerOrder
    next_speaker : SpeakerOrder

    dialogue_history : List[DataForOneDialogueTurn]
    winner : str 
    loser : str 
    reason_for_dialogue_termination : Dict[str, str]

    utterance_classification_approach : UtteranceClassificationApproach
    utterance_classification_number_of_classifications : int

    mode : Mode

class BaseModelState(TypedDict):
    session_id : str
    max_dialogue_turns : int
    max_sentences : int

    speaker : str
    speaker_order : SpeakerOrder
    speaker_model_name : ModelName
    model_temp : float
    model_top_p : float
    model_seed : int

    debate_topic : str
    political_position_range:str
    political_position_std:int

    utterance : str
    utterance_with_corresponding_types : List[Tuple[str, str]]
    original_claim : str 
    commitments : List[str]

    opponent_speaker_name : str
    opponents_utterance : str
    opponents_original_claim : str
    opponents_commitments : List[str]

    current_speaker : SpeakerOrder
    next_speaker : SpeakerOrder

    dialogue_history : List[DataForOneDialogueTurn]
    winner : str 
    loser : str 
    reason_for_dialogue_termination : Dict[str, str]

    utterance_classification_approach : UtteranceClassificationApproach
    utterance_classification_number_of_classifications : int

    mode : Mode

class HumanState(TypedDict):
    session_id : str
    max_dialogue_turns : int
    max_sentences : int

    speaker : str
    speaker_order : SpeakerOrder
    utterance : str
    utterance_with_corresponding_types : List[Tuple[str, str]]
    original_claim : str 
    commitments : List[str]

    debate_topic : str
   
    opponent_speaker_name : str
    opponents_utterance : str
    opponents_utterance_with_corresponding_types : List[Tuple[str, str]]
    opponents_original_claim : str
    opponents_commitments : List[str]

    typical_replies_for_last_speakers_response : List[Tuple[Dict[str, str], int, int]]

    current_speaker : SpeakerOrder
    next_speaker : SpeakerOrder

    dialogue_history : List[DataForOneDialogueTurn]
    winner : str 
    loser : str 
    reason_for_dialogue_termination : Dict[str, str]

    human_model_name : ModelName
    human_model_temp : float
    human_model_top_p : float
    human_model_seed : int

    utterance_classification_approach : UtteranceClassificationApproach
    utterance_classification_number_of_classifications : int

    mode : Mode