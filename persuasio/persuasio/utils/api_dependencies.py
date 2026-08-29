from fastapi import HTTPException

from persuasio.datatypes.api import SessionParameters, ClientResponse
from persuasio.datatypes.enums import SpeakerOrder, SpeakerType, ModelName, UtteranceClassificationApproach, PoliticalPositionEnsembleOrModelName



async def session_parameters(
        first_speaker: str,
        second_speaker: str,
        first_speaker_type : SpeakerType,
        second_speaker_type : SpeakerType,
        first_speaker_model_name : ModelName = ModelName.NO_MODEL,
        second_speaker_model_name : ModelName = ModelName.NO_MODEL,
        session_id: str = "",
        debate_topic: str = "Debate the trade-offs of government intervention across a range of issues, including healthcare, immigration, and welfare.",
        max_dialogue_turns : int = 40,
        max_sentences_per_turn : int = 5,        
        
        first_speaker_model_temp : float = 0,
        first_speaker_model_top_p : float =1,
        first_speaker_model_seed : int = 123,
        first_speaker_political_political_position_range: str = "0:100",
        first_speaker_political_position_std: int = 10,
        first_speaker_political_position_prob_of_na: float = 0.25,
        first_speaker_knowledge_base_ensemble_or_model_name : PoliticalPositionEnsembleOrModelName = PoliticalPositionEnsembleOrModelName.NO_USE_OF_KNOWLEDGE_BASE,
        first_speaker_number_of_vector_based_rag_examples : int = 5,
        first_speaker_number_of_graph_rag_examples: int = 5,

        
        second_speaker_political_position_range: str = "0:100",
        second_speaker_political_position_std: int = 10,
        second_speaker_political_position_prob_of_na: float = 0.25,
        second_speaker_model_temp : float = 0,
        second_speaker_model_top_p : float = 1,
        second_speaker_model_seed : int = 123,
        second_speaker_knowledge_base_ensemble_or_model_name : PoliticalPositionEnsembleOrModelName = PoliticalPositionEnsembleOrModelName.NO_USE_OF_KNOWLEDGE_BASE,
        second_speaker_number_of_vector_based_rag_examples : int =5,
        second_speaker_number_of_graph_rag_examples : int = 5,

        human_model_name : ModelName = ModelName.NO_MODEL,
        human_model_temp : float = 0,
        human_model_top_p : float = 1,
        human_model_seed : int = 123,

        utterance_classification_approach : UtteranceClassificationApproach = UtteranceClassificationApproach.SINGLE_CLASSIFICATION,
        utterance_classification_number_of_classifications : int = 1            
                            ):
    """
    Dependency that builds a SessionInfo object from query parameters.
    """
    if first_speaker_type != SpeakerType.HUMAN:
        if first_speaker_model_name == ModelName.NO_MODEL:
            raise HTTPException(400, detail=f"You must set the 'first_speaker_model_name' to one of the following: {[name.value for name in ModelName if name.value != '']}")
        first_speaker = first_speaker_model_name.value + "_" + first_speaker_type.value
    if second_speaker_type != SpeakerType.HUMAN:
        if second_speaker_model_name == ModelName.NO_MODEL:
            raise HTTPException(400, detail=f"You must set the 'second_speaker_model_name' to one of the following: {[name.value for name in ModelName if name.value != '']}")
        second_speaker = second_speaker_model_name.value + "_" + second_speaker_type.value
        
    if first_speaker_type == SpeakerType.HUMAN:
        if human_model_name == ModelName.NO_MODEL:
            raise HTTPException(400, detail=f"You need to specify a 'human_model_name' because 'first_speaker_type' is human.")
    if second_speaker_type == SpeakerType.HUMAN:
        if human_model_name == ModelName.NO_MODEL:
            raise HTTPException(400, detail=f"You need to specify a 'human_model_name' because 'second_speaker_type' is human.")
        
    if first_speaker_type != SpeakerType.MAS_RAG:
        first_speaker_knowledge_base_ensemble_or_model_name = PoliticalPositionEnsembleOrModelName.NO_USE_OF_KNOWLEDGE_BASE
    if second_speaker_type != SpeakerType.MAS_RAG:
        second_speaker_knowledge_base_ensemble_or_model_name = PoliticalPositionEnsembleOrModelName.NO_USE_OF_KNOWLEDGE_BASE
        

    return SessionParameters(
        session_id=session_id,
        debate_topic=debate_topic,
        max_dialogue_turns = max_dialogue_turns,
        max_sentences_per_turn = max_sentences_per_turn,
        #commitments_cosine_similarity_threshold = commitments_cosine_similarity_threshold,

        first_speaker=first_speaker,
        first_speaker_type=first_speaker_type,
        first_speaker_model_name = first_speaker_model_name,
        first_speaker_model_temp = first_speaker_model_temp,
        first_speaker_model_top_p = first_speaker_model_top_p,
        first_speaker_model_seed = first_speaker_model_seed,
        first_speaker_political_political_position_range = first_speaker_political_political_position_range,
        first_speaker_political_position_std = first_speaker_political_position_std,
        first_speaker_political_position_prob_of_na = first_speaker_political_position_prob_of_na,
        first_speaker_knowledge_base_ensemble_or_model_name = first_speaker_knowledge_base_ensemble_or_model_name,
        first_speaker_number_of_vector_based_rag_examples = first_speaker_number_of_vector_based_rag_examples,
        first_speaker_number_of_graph_rag_examples = first_speaker_number_of_graph_rag_examples,

        second_speaker=second_speaker,
        second_speaker_type=second_speaker_type,
        second_speaker_model_name = second_speaker_model_name,
        second_speaker_model_temp = second_speaker_model_temp,
        second_speaker_model_top_p = second_speaker_model_top_p,
        second_speaker_model_seed = second_speaker_model_seed,
        second_speaker_political_position_range = second_speaker_political_position_range,
        second_speaker_political_position_std = second_speaker_political_position_std,
        second_speaker_political_position_prob_of_na = second_speaker_political_position_prob_of_na,
        second_speaker_knowledge_base_ensemble_or_model_name = second_speaker_knowledge_base_ensemble_or_model_name,
        second_speaker_number_of_vector_based_rag_examples = second_speaker_number_of_vector_based_rag_examples,
        second_speaker_number_of_graph_rag_examples = second_speaker_number_of_graph_rag_examples,

        human_model_name = human_model_name,
        human_model_temp = human_model_temp,
        human_model_top_p = human_model_top_p,
        human_model_seed = human_model_seed,

        utterance_classification_approach = utterance_classification_approach,
        utterance_classification_number_of_classifications = utterance_classification_number_of_classifications
    )


async def response(utterance_from : SpeakerOrder,
                        utterance : str = "",
                        ):
    """
    Dependency that builds a Response object from query parameters.
    """
    if utterance_from == SpeakerOrder.FIRST_SPEAKER:
        next_speaker = SpeakerOrder.SECOND_SPEAKER
    elif utterance_from == SpeakerOrder.SECOND_SPEAKER:
        next_speaker = SpeakerOrder.FIRST_SPEAKER
    return ClientResponse(
        utterance=utterance,
        utterance_from=utterance_from,
        next_speaker=next_speaker
        )