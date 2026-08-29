import string
from typing import Set, List
import random

from session_manager.models.entities import Participant, Session
from session_manager.models.enums import ModelName, SpeakerType

def generate_session_id(existing_ids: Set[str]) -> str:
    """Generate a unique 6-digit uppercase alphanumeric session ID."""
    while True:
        session_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if session_id not in existing_ids:
            existing_ids.add(session_id)
            return session_id
        

def generate_auth_code(existing_codes: Set[str]) -> str:
    """Generate a unique 6-digit alphanumeric auth code."""
    while True:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if code not in existing_codes:
            existing_codes.add(code)
            return code


def get_model_name_from_participant(p: Participant) -> ModelName:
    """Extract ModelName from a model participant's ID.

    For model participants, the ID format is: "{model_value}_{variant_value}"
    e.g., "gpt-4o_base" -> ModelName.GPT_4O

    For human participants, returns ModelName.NO_MODEL.
    """
    if p.participant_type == SpeakerType.HUMAN:
        return ModelName.NO_MODEL

    # Extract model part by removing the variant suffix
    variant_suffix = f"_{p.participant_type.value}"

    if not p.participant_id.endswith(variant_suffix):
        raise ValueError(f"Participant ID does not end with expected variant: {p.participant_id}")

    model_part = p.participant_id[:-len(variant_suffix)]

    for model in ModelName:
        if model.value == model_part:
            return model

    raise ValueError(f"Could not extract model name from participant_id: {p.participant_id}")

def get_human_participants_from_session(session: Session, human_ids: Set[str]) -> List[str]:
    """Extract human participant IDs from a session."""
    humans = []
    if session.parameters.first_speaker in human_ids:
        humans.append(session.parameters.first_speaker)
    if session.parameters.second_speaker in human_ids:
        humans.append(session.parameters.second_speaker)
    return humans


def is_hh_session(session: Session) -> bool:
    """Check if session is human-human."""
    return (session.parameters.first_speaker_type == SpeakerType.HUMAN and
            session.parameters.second_speaker_type == SpeakerType.HUMAN)
    
def get_session_type(session: Session) -> str:
    """Get session type as a string: 'HH', 'HM', or 'MM'."""
    first_type = session.parameters.first_speaker_type
    second_type = session.parameters.second_speaker_type
    if first_type == SpeakerType.HUMAN and second_type == SpeakerType.HUMAN:
        return "HH"
    elif first_type != SpeakerType.HUMAN and second_type != SpeakerType.HUMAN:
        return "MM"
    else:
        return "HM"