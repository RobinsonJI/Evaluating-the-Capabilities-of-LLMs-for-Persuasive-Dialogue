from pydantic import BaseModel, Field
from typing import Dict, List, Any

class LoginRequest(BaseModel):
    """Request body for POST /auth/login"""
    code: str

class LoginResponse(BaseModel):
    """Response for POST /auth/login"""
    participant_id: str
    participant_type: str
    message: str

class JoinSessionRequest(BaseModel):
    """Request body for POST /sessions/{session_id}/join"""
    participant_id: str

class JoinSessionResponse(BaseModel):
    """Response for POST /sessions/{session_id}/join"""
    message: str
    session: dict

class SendMessageRequest(BaseModel):
    """Request body for POST /sessions/{session_id}/messages"""
    participant_id: str
    content: str

class SendMessageResponse(BaseModel):
    """Response for POST /sessions/{session_id}/messages"""
    message: str
    turn_number: int | None = None

class SessionStateResponse(BaseModel):
    """Response for GET /sessions/{session_id}/state"""
    changed: bool
    session: dict
    can_participant_speak: bool
    participant_role: str  # "first_speaker", "second_speaker", or "observer"
    typical_replies: List[Any] = Field(default_factory=list)  # List of dict objects from persuasio
    first_speaker_commitments: List[str] = Field(default_factory=list)
    second_speaker_commitments: List[str] = Field(default_factory=list)
