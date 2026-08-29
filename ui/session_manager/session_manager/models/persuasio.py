from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Tuple

from .enums import SpeakerOrder, SessionStatus

class ClientResponse(BaseModel):
    """
    Stores a response from both speakers in a single step of the dialogue.
    Only 'first_speaker_utterance' or 'second_speaker_utterance' will be populated with response data.
    """
    utterance: str = Field(default="")
    utterance_from: SpeakerOrder
    next_speaker: SpeakerOrder

class DataForOneDialogueTurn(BaseModel):
    """
    Stores the data for a single dialogue turn
    """
    speaker: str
    sentences_with_utterance_types: List[Tuple[str, str]]
    sentences_no_utterance_types: str
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

class PersuasioResponse(BaseModel):
    """
    The response object that will be sent back to the front end.
    """
    dialogue_history: List[DataForOneDialogueTurn] = Field(default_factory=list)

    first_speaker: str = Field(default="")
    first_speaker_utterance: str = Field(default="")
    first_speaker_commitments: List[str] = Field(default_factory=list)
    first_speaker_original_claim: str = Field(default="")
    first_speaker_typical_replies_for_second_speakers_response: List[Union[Dict[str, str], Any, Any]] = Field(default_factory=list)

    second_speaker: str = Field(default="")
    second_speaker_utterance: str = Field(default="")
    second_speaker_commitments: List[str] = Field(default_factory=list)
    second_speaker_original_claim: str = Field(default="")
    second_speaker_typical_replies_for_first_speakers_response: List[Union[Dict[str, str], Any, Any]] = Field(default_factory=list)

    winner: str = Field(default="")
    loser: str = Field(default="")
    reason_for_dialogue_termination: Dict[str, Any] = Field(default_factory=dict)

    session_status: SessionStatus

# Type alias for compatibility with data.py
DialogueTurn = DataForOneDialogueTurn
