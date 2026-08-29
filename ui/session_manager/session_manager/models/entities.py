from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

from .enums import SpeakerOrder, SessionStatus, SpeakerType, ModelName, UtteranceClassificationApproach, PoliticalPositionEnsembleOrModelName
from .persuasio import DataForOneDialogueTurn

class SessionParameters(BaseModel):
    """
    Holds the parameters for starting a new debate session.
    """
    session_id: str = Field(default="")
    debate_topic: str = Field(default="Debate the trade-offs of government intervention across a range of issues, including healthcare, immigration, and welfare.")
    max_dialogue_turns: int = Field(default=40)
    max_sentences_per_turn: int = Field(default=5)

    # first speaker
    first_speaker: str = Field(default="")
    first_speaker_type: SpeakerType
    first_speaker_model_name: ModelName
    first_speaker_model_temp: float = Field(default=0)
    first_speaker_model_top_p: float = Field(default=1)
    first_speaker_model_seed: int = Field(default=123)
    first_speaker_political_political_position_range: str = Field(default="0:100")
    first_speaker_political_position_std: int = Field(default=10)
    first_speaker_political_position_prob_of_na: float = Field(default=0.25)
    first_speaker_knowledge_base_ensemble_or_model_name: PoliticalPositionEnsembleOrModelName = Field(default=PoliticalPositionEnsembleOrModelName.ENSEMBLE_3_LESS_POL_SCORES_THAN_NA)
    first_speaker_number_of_vector_based_rag_examples: int = Field(default=5)
    first_speaker_number_of_graph_rag_examples: int = Field(default=5)

    # Second speaker
    second_speaker: str = Field(default="")
    second_speaker_type: SpeakerType
    second_speaker_model_name: ModelName
    second_speaker_model_temp: float = Field(default=0)
    second_speaker_model_top_p: float = Field(default=1)
    second_speaker_model_seed: int = Field(default=123)
    second_speaker_political_position_range: str = Field(default="0:100")
    second_speaker_political_position_std: int = Field(default=10)
    second_speaker_political_position_prob_of_na: float = Field(default=0.25)
    second_speaker_knowledge_base_ensemble_or_model_name: PoliticalPositionEnsembleOrModelName = Field(default=PoliticalPositionEnsembleOrModelName.ENSEMBLE_3_LESS_POL_SCORES_THAN_NA)
    second_speaker_number_of_vector_based_rag_examples: int = Field(default=5)
    second_speaker_number_of_graph_rag_examples: int = Field(default=5)

    human_model_name: ModelName = Field(default=ModelName.GPT_4O)
    human_model_temp: float = Field(default=0)
    human_model_top_p: float = Field(default=1)
    human_model_seed: int = Field(default=123)

    utterance_classification_approach: UtteranceClassificationApproach = Field(default=UtteranceClassificationApproach.SINGLE_CLASSIFICATION)
    utterance_classification_number_of_classifications: int = Field(default=1)

class Session(BaseModel):
    """
    Session model for SessionManager.
    Wraps SessionParameters and tracks runtime state.
    """
    session_id: str
    parameters: SessionParameters
    persuasio_exists: bool = False
    current_turn: SpeakerOrder = SpeakerOrder.FIRST_SPEAKER
    status: SessionStatus = SessionStatus.STARTED
    dialogue_history: List[DataForOneDialogueTurn] = Field(default_factory=list)
    participant1_joined: bool = False
    participant2_joined: bool = False

    # Commitment tracking (persisted)
    first_speaker_commitments: List[str] = Field(default_factory=list)
    second_speaker_commitments: List[str] = Field(default_factory=list)

    # Typical replies (persisted) - stored as dicts from persuasio
    first_speaker_typical_replies: List[Any] = Field(default_factory=list)
    second_speaker_typical_replies: List[Any] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    # Session termination
    end_reason: Optional[str] = None

    @property
    def participant1_id(self) -> str:
        """First speaker (FOR position)"""
        return self.parameters.first_speaker

    @property
    def participant2_id(self) -> str:
        """Second speaker (AGAINST position)"""
        return self.parameters.second_speaker

    @property
    def topic(self) -> str:
        """Debate topic"""
        return self.parameters.debate_topic

    def can_participant_speak(self, participant_id: str) -> bool:
        """Check if it's the participant's turn to speak"""
        if self.status != SessionStatus.RUNNING and self.status != SessionStatus.STARTED:
            return False

        if participant_id == self.participant1_id:
            return self.current_turn == SpeakerOrder.FIRST_SPEAKER
        elif participant_id == self.participant2_id:
            return self.current_turn == SpeakerOrder.SECOND_SPEAKER

        return False

    def add_message(self, participant_id: str, message: str) -> DataForOneDialogueTurn:
        """
        Add a message to the dialogue history and switch turns.
        Creates a simple turn without utterance classification.

        Args:
            participant_id: ID of the participant sending the message
            message: The message content

        Returns:
            The created dialogue turn
        """
        # Create a simple dialogue turn without classifications
        turn = DataForOneDialogueTurn(
            speaker=participant_id,
            sentences_with_utterance_types=[],  # No classification for human input
            sentences_no_utterance_types=message,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Use unified method to add turn and update state
        return self.add_dialogue_turn(turn)

    def add_dialogue_turn(self, turn: DataForOneDialogueTurn) -> DataForOneDialogueTurn:
        """
        Add a pre-constructed dialogue turn (e.g., from persuasio with classifications).

        Args:
            turn: The dialogue turn to add

        Returns:
            The added dialogue turn
        """
        # Add to history
        self.dialogue_history.append(turn)

        # Switch turn
        if self.current_turn == SpeakerOrder.FIRST_SPEAKER:
            self.current_turn = SpeakerOrder.SECOND_SPEAKER
        else:
            self.current_turn = SpeakerOrder.FIRST_SPEAKER

        # Update timestamp for change detection
        self.last_updated = datetime.now()

        return turn

    def update_commitments(self, first_speaker_commitments: List[str], second_speaker_commitments: List[str]):
        """
        Update participant commitments from persuasio response.

        Args:
            first_speaker_commitments: Updated list of first speaker's commitments
            second_speaker_commitments: Updated list of second speaker's commitments
        """
        self.first_speaker_commitments = first_speaker_commitments
        self.second_speaker_commitments = second_speaker_commitments

        # Update timestamp for change detection
        self.last_updated = datetime.now()


class Participant(BaseModel):
    """
    Participant model for SessionManager.
    Represents both human and AI participants.
    """
    participant_id: str
    participant_type: SpeakerType
    auth_code: Optional[str] = None
    is_authenticated: bool = False
    current_session: Optional[str] = None
    is_admin: bool = False
    last_activity: Optional[datetime] = None
    message_count: int = 0

    @property
    def is_ai(self) -> bool:
        """Check if participant is an AI model"""
        return self.participant_type != SpeakerType.HUMAN

    @property
    def is_ready(self) -> bool:
        """Check if participant is ready to participate (AI always ready, humans need auth)"""
        return self.is_ai or self.is_authenticated

class State(BaseModel):
    """Placeholder for persuasio state restoration"""
    session_id: str
    # TODO: Add fields as needed for restore_session functionality
