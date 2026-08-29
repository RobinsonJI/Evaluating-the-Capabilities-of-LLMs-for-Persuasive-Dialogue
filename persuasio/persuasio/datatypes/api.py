from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union

from persuasio.datatypes.enums import SpeakerOrder, SpeakerType, ModelName, SessionStatus, UtteranceClassificationApproach, PoliticalPositionEnsembleOrModelName
from persuasio.datatypes.pydantic_basemodels import DataForOneDialogueTurn

class ClientResponse(BaseModel):
    """
    Stores a response from both speakers in a single step of the dialogue .
    Only 'first_speaker_utterance' or 'second_speaker_utterance' will be populated with response data.
    """
    utterance : str = Field(default="")
    utterance_from : SpeakerOrder
    next_speaker : SpeakerOrder


class SessionParameters(BaseModel):
    """
    Holds the parameters for starting a new debate session.
    """
    session_id : str = Field(default="")
    debate_topic : str = Field(default="Debate the trade-offs of government intervention across a range of issues, including healthcare, immigration, and welfare.")
    max_dialogue_turns : int = Field(default=40)
    max_sentences_per_turn : int = Field(default=5)
    #commitments_cosine_similarity_threshold : float = Field(default=0.75)

    # first speaker
    first_speaker : str = Field(default="")
    first_speaker_type : SpeakerType
    first_speaker_model_name : ModelName
    first_speaker_model_temp : float = Field(default=0)
    first_speaker_model_top_p : float = Field(default=1)
    first_speaker_model_seed : int = Field(default=123)
    # ERROR here - double "political"
    first_speaker_political_political_position_range: str = Field(default="0:100")
    first_speaker_political_position_std: int = Field(default=10)
    first_speaker_political_position_prob_of_na: float = Field(default=0.25)
    first_speaker_knowledge_base_ensemble_or_model_name : PoliticalPositionEnsembleOrModelName = Field(default=PoliticalPositionEnsembleOrModelName.ENSEMBLE_3_LESS_POL_SCORES_THAN_NA)
    first_speaker_number_of_vector_based_rag_examples : int = Field(default=5)
    first_speaker_number_of_graph_rag_examples: int = Field(default=5)

    # Second speaker
    second_speaker : str = Field(default="")
    second_speaker_type : SpeakerType
    second_speaker_model_name : ModelName
    second_speaker_model_temp : float = Field(default=0)
    second_speaker_model_top_p : float = Field(default=1)
    second_speaker_model_seed : int = Field(default=123)
    second_speaker_political_position_range: str = Field(default="0:100")
    second_speaker_political_position_std: int = Field(default=10)
    second_speaker_political_position_prob_of_na: float = Field(default=0.25)
    second_speaker_knowledge_base_ensemble_or_model_name : PoliticalPositionEnsembleOrModelName = Field(default=PoliticalPositionEnsembleOrModelName.ENSEMBLE_3_LESS_POL_SCORES_THAN_NA)
    second_speaker_number_of_vector_based_rag_examples : int = Field(default=5)
    second_speaker_number_of_graph_rag_examples : int = Field(default=5)

    human_model_name : ModelName = Field(default=ModelName.NO_MODEL)
    human_model_temp : float = Field(default=0)
    human_model_top_p : float = Field(default=1)
    human_model_seed : int = Field(default=123)

    utterance_classification_approach : UtteranceClassificationApproach = Field(default=UtteranceClassificationApproach.SINGLE_CLASSIFICATION)
    utterance_classification_number_of_classifications : int = Field(default=1)


class PersuasioResponse(BaseModel):
    """
    The response object that will be sent back to the front end.
    """
    dialogue_history : List[DataForOneDialogueTurn] = Field(default_factory=list)

    first_speaker : str = Field(default="")
    first_speaker_utterance : str = Field(default="")
    first_speaker_commitments : List[str] = Field(default_factory=list)
    first_speaker_original_claim : str = Field(default="")
    first_speaker_typical_replies_for_second_speakers_response : List[Union[Dict[str, str], Any, Any]] = Field(default_factory=list)

    second_speaker : str = Field(default="")
    second_speaker_utterance : str = Field(default="")
    second_speaker_commitments : List[str] = Field(default_factory=list)
    second_speaker_original_claim : str = Field(default="")
    second_speaker_typical_replies_for_first_speakers_response : List[Union[Dict[str, str], Any, Any]] = Field(default_factory=list)

    winner : str = Field(default="")
    loser : str = Field(default="")
    reason_for_dialogue_termination : Dict[str, Any] = Field(default_factory=dict)

    session_status : SessionStatus