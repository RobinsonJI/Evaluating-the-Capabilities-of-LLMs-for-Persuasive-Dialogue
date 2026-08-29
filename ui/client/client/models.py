"""
Pydantic models for the Persuasui Client application.

These models define the structured data returned by the core logic layer
(BasePersuasuiClient) to be consumed by a UI adapter layer.
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
from enum import Enum

# --- Activity Logging Models ---

class ActivityEvent(BaseModel):
    """Activity log event for monitoring"""
    timestamp: datetime
    event_type: str  # 'login', 'session', 'message', 'system'
    severity: str    # 'debug', 'info', 'warning', 'error'
    participant_id: str
    message: str
    session_id: Optional[str] = None

# --- Data structures copied from session_manager for parsing ---

class DataForOneDialogueTurn(BaseModel):
    """
    Represents a single turn in the dialogue history as received from the server.
    """
    speaker: str
    sentences_with_utterance_types: List[Tuple[str, str]]
    sentences_no_utterance_types: str
    timestamp: str
    
# --- Utterance types/descriptions ---

class UtteranceType(Enum):
    CLAIM = "__Claim__"
    NOT_CLAIM = "__NotClaim__"
    WHY = "__Why__"
    SINCE = "__Since__"
    QUESTION = "__Question__"
    CONCEDE = "__Concede__"
    RETRACT = "__Retract__"

UTT_DESC = {
    UtteranceType.CLAIM: "Asserts that some proposition is true. Example: \"My car is safe.\"",
    UtteranceType.NOT_CLAIM: "Asserts that some proposition is not True. Example: \"My car is not safe.\"",
    UtteranceType.WHY: "Challenges that some proposition is true and asks for reasons/evidence. Example: \"Why is your car safe?\"",
    UtteranceType.SINCE: "Asserts some proposition is true AND provides explicit supporting reasons. Example: \"My car is safe since it has an airbag.\"",
    UtteranceType.QUESTION: "Asks for an opinion on whether some proposition is true. Example: \"Do you think the car is safe?\"",
    UtteranceType.CONCEDE: "Explicitly admits that proposition is true. Example: \"That is true.\" or \"I agree\"",
    UtteranceType.RETRACT: "Explicitly withdraws or denies a previous commitment to ϕ. Example: \"OK, I was wrong that my car is safe.\"",
}

UTT_TITLE = {
    UtteranceType.CLAIM: "✋ Claim",
    UtteranceType.NOT_CLAIM: "👎 Not Claim",
    UtteranceType.WHY: "❓ Why",
    UtteranceType.SINCE: "💡 Since",
    UtteranceType.QUESTION: "👀 Question",
    UtteranceType.CONCEDE: "🤝 Concede",
    UtteranceType.RETRACT: "↩️ Retract",
}

# --- Cleaned-up data structures for UI consumption ---

class ChatItem(BaseModel):
    """
    A cleaned-up, simple representation of a single chat message for the UI.
    """
    is_user: bool
    speaker_id: str
    content: str
    turn_number: int
    timestamp: Optional[str] = None

# --- Result models for BasePersuasuiClient handle_* methods ---

class LoginResult(BaseModel):
    """Return model for handle_login."""
    error: Optional[str] = None
    participant_id: Optional[str] = None
    is_admin: bool = False
    message: Optional[str] = None

class JoinSessionResult(BaseModel):
    """Return model for handle_join_session."""
    error: Optional[str] = None
    message: Optional[str] = None
    user_context: Optional[Dict[str, str]] = None

class SendMessageResult(BaseModel):
    """Return model for handle_send_message."""
    error: Optional[str] = None
    message: Optional[str] = None

class RefreshResult(BaseModel):
    """Return model for handle_refresh."""
    changed: bool
    error: Optional[str] = None
    status_text: Optional[str] = None
    topic_and_stance: Optional[str] = None
    session_info: Optional[Dict] = None  # Raw session dict for detailed display
    chat_items: Optional[List[ChatItem]] = None
    typical_replies: Optional[List[Any]] = None  # Typical replies for user's next turn
    first_speaker_commitments: Optional[List[str]] = None
    second_speaker_commitments: Optional[List[str]] = None

class AdminRefreshResult(BaseModel):
    """Return model for handle_admin_refresh."""
    error: Optional[str] = None
    dashboard_data: Optional[Dict] = None

class EndSessionResult(BaseModel):
    """Return model for handle_end_session."""
    error: Optional[str] = None
    message: Optional[str] = None


