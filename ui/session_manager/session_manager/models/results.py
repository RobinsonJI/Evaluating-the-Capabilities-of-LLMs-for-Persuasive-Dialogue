from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel

from .enums import AuthStatus, SendMessageStatus, EndSessionStatus, JoinSessionStatus
from .persuasio import DialogueTurn
from .entities import Participant, Session

class ActivityEvent(BaseModel):
    """Activity log event for monitoring"""
    timestamp: datetime
    event_type: str  # 'login', 'session', 'message', 'system'
    severity: str  # 'debug', 'info', 'warning', 'error'
    participant_id: str
    message: str
    session_id: Optional[str] = None

@dataclass
class AuthResult:
    """Result from authenticate_participant()"""
    status: AuthStatus
    participant: Optional[Participant] = None
    error_message: str = ""

@dataclass
class ValidateParticipantTurnResult:
    """Result from _validate_participant_turn()"""
    is_valid: bool
    participant: Optional[Participant] = None
    session: Optional[Session] = None
    error_status: Optional[SendMessageStatus] = None
    error_message: str = ""

@dataclass
class ValidateParticipantResult:
    """Result from atomic participant validators"""
    is_valid: bool
    participant: Optional[Participant] = None
    error_message: str = ""

@dataclass
class ValidateSessionResult:
    """Result from atomic session validators"""
    is_valid: bool
    session: Optional[Session] = None
    error_message: str = ""

@dataclass
class JoinSessionResult:
    """Result from join_session()"""
    status: JoinSessionStatus
    participant: Optional[Participant] = None
    session: Optional[Session] = None
    error_message: str = ""

@dataclass
class SendMessageResult:
    """Result from send_message()"""
    status: SendMessageStatus
    human_turn: Optional[DialogueTurn] = None
    typical_replies: List[Dict] = field(default_factory=list)
    commitments: Dict[str, List[str]] = field(default_factory=dict)
    error_message: str = ""
    # Termination info
    session_ended: bool = False
    termination_reason: Optional[str] = None

@dataclass
class EndSessionResult:
    """Result from user_end_session()"""
    status: EndSessionStatus
    session_id: str = ""
    error_message: str = ""
