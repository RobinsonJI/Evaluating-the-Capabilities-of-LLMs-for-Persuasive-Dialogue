from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import APIKeyHeader

from session_manager.models import *
from session_manager.session_manager import SessionManager
import session_manager.app as app_module

### --- API KEY SECURITY ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key from the request header"""
    if api_key != app_module.SESSION_MANAGER_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# --- REUSABLE DEPENDENCIES ---

def get_session_manager() -> SessionManager:
    """Dependency that provides the SessionManager instance"""
    return app_module.get_session_manager()


async def verify_admin(
    participant_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
) -> Participant:
    """
    Dependency that verifies admin access.

    Args:
        participant_id: ID of participant to verify
        session_manager: SessionManager instance

    Returns:
        Participant object if admin and authenticated

    Raises:
        HTTPException 403 if not admin or not authenticated
    """
    participant = session_manager.participants.get(participant_id)
    if not participant or not participant.is_admin or not participant.is_authenticated:
        raise HTTPException(status_code=403, detail="Admin access required")
    return participant


async def get_validated_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
) -> Session:
    """
    Dependency that validates session exists.

    Args:
        session_id: Session identifier
        session_manager: SessionManager instance

    Returns:
        Session object if found

    Raises:
        HTTPException 404 if session not found
    """
    session = session_manager.sessions.get(session_id.upper())
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


async def get_validated_participant(
    participant_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
) -> Participant:
    """
    Dependency that validates participant exists.

    Args:
        participant_id: Participant identifier
        session_manager: SessionManager instance

    Returns:
        Participant object if found

    Raises:
        HTTPException 404 if participant not found
    """
    participant = session_manager.participants.get(participant_id)
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    return participant


# --- API ROUTER ---

router = APIRouter(dependencies=[Depends(verify_api_key)])

# --- API ENDPOINTS ---

# --- HEALTH CHECK ---

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "session-manager"}


# --- AUTHENTICATION ---

@router.post("/auth/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Authenticate participant with login code.

    Returns participant info if code is valid, raises 401 if invalid.
    """
    result = session_manager.authenticate_participant(request.code.strip())

    match result.status:
        case AuthStatus.SUCCESS:
            # Log activity
            session_manager._log_activity(
                event_type="login",
                participant_id=result.participant.participant_id,
                message=f"User logged in with code",
                session_id=None
            )

            return LoginResponse(
                participant_id=result.participant.participant_id,
                participant_type="human" if not result.participant.is_ai else "ai",
                message=f"Welcome, {result.participant.participant_id}!"
            )

        case AuthStatus.INVALID_CODE:
            raise HTTPException(status_code=401, detail=result.error_message)


# --- SESSION MANAGEMENT ---

@router.post("/sessions/{session_id}/join", response_model=JoinSessionResponse)
async def join_session(
    session_id: str,
    request: JoinSessionRequest,
    background_tasks: BackgroundTasks,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Join a session with authenticated participant.

    Args:
        session_id: Session code to join
        request: Contains participant_id

    Returns:
        Success message and session data

    Raises:
        401: Not authenticated
        403: Not assigned to this session
        404: Invalid session code
    """
    result = await session_manager.join_session(request.participant_id, session_id.upper(), background_tasks)

    match result.status:
        case JoinSessionStatus.JOINED:
            return JoinSessionResponse(
                message="Joined successfully!",
                session=result.session.model_dump(mode='json')
            )

        case JoinSessionStatus.NOT_AUTHENTICATED:
            raise HTTPException(status_code=401, detail=result.error_message)

        case JoinSessionStatus.NOT_ASSIGNED:
            raise HTTPException(status_code=403, detail=result.error_message)

        case JoinSessionStatus.INVALID_SESSION:
            raise HTTPException(status_code=404, detail=result.error_message)

        case JoinSessionStatus.INVALID_PARTICIPANT:
            raise HTTPException(status_code=404, detail=result.error_message)

        case JoinSessionStatus.ERROR:
            raise HTTPException(status_code=500, detail=result.error_message)


@router.get("/sessions/{session_id}/state", response_model=SessionStateResponse)
async def get_session_state(
    participant_id: str,
    session: Session = Depends(get_validated_session),
    participant: Participant = Depends(get_validated_participant)
):
    """
    Get current session state.

    Args:
        participant_id: Participant requesting state
        session: Validated session (from dependency)
        participant: Validated participant (from dependency)

    Returns:
        SessionStateResponse with:
        - changed: Always True (no change detection)
        - session: Full session data (dict)
        - can_participant_speak: Boolean
        - participant_role: "first_speaker", "second_speaker", or "observer"
        - typical_replies: Typical replies for participant's next turn
        - first_speaker_commitments: All commitments made by first speaker
        - second_speaker_commitments: All commitments made by second speaker
    """
    # Determine participant's role and get relevant typical replies
    if participant_id == session.participant1_id:
        role = "first_speaker"
        typical_replies = session.first_speaker_typical_replies
    elif participant_id == session.participant2_id:
        role = "second_speaker"
        typical_replies = session.second_speaker_typical_replies
    else:
        role = "observer"
        typical_replies = []

    # Always return full session data
    return SessionStateResponse(
        changed=True,
        session=session.model_dump(mode='json'),
        can_participant_speak=session.can_participant_speak(participant_id),
        participant_role=role,
        typical_replies=typical_replies,
        first_speaker_commitments=session.first_speaker_commitments,
        second_speaker_commitments=session.second_speaker_commitments
    )


@router.post("/sessions/{session_id}/messages", response_model=SendMessageResponse)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Send a message in the session.

    Args:
        session_id: Session identifier
        request: Contains participant_id and message content

    Returns:
        Success message and optional turn number

    Raises:
        400: Not in a session
        403: Not your turn to speak
        404: Session not found
        502: Backend processing error
    """
    result = await session_manager.send_message(
        request.participant_id,
        session_id.upper(),
        request.content.strip()
    )

    match result.status:
        case SendMessageStatus.SENT:
            # Log message activity
            participant = session_manager.participants.get(request.participant_id)
            if participant:
                participant.message_count += 1

            # Construct response message
            if result.session_ended:
                message = f"Message sent! Debate ended: {result.termination_reason}"
            else:
                message = "Message sent!"

            # Calculate turn number from dialogue history length
            session = session_manager.sessions.get(session_id.upper())
            turn_number = len(session.dialogue_history) if (result.human_turn and session) else None

            return SendMessageResponse(
                message=message,
                turn_number=turn_number
            )

        case SendMessageStatus.NOT_YOUR_TURN:
            raise HTTPException(status_code=403, detail=result.error_message)

        case SendMessageStatus.NOT_IN_SESSION:
            raise HTTPException(status_code=400, detail=result.error_message)

        case SendMessageStatus.INVALID_SESSION:
            raise HTTPException(status_code=404, detail=result.error_message)

        case SendMessageStatus.BACKEND_ERROR:
            raise HTTPException(status_code=502, detail=result.error_message)


@router.post("/sessions/{session_id}/end")
async def end_session(
    session_id: str,
    participant_id: str,
    message: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    End a session (user-initiated).

    Args:
        session_id: Session identifier
        participant_id: Participant ending the session
        message: Reason/message for ending

    Returns:
        Success confirmation

    Raises:
        404: Session not found
        500: Error ending session
    """
    result = await session_manager.user_end_session(
        session_id.upper(),
        participant_id,
        message
    )

    match result.status:
        case EndSessionStatus.SUCCESS:
            return {"message": "Session ended successfully", "session_id": result.session_id}

        case EndSessionStatus.INVALID_SESSION:
            raise HTTPException(status_code=404, detail=result.error_message)

        case EndSessionStatus.PERSUASIO_ERROR:
            raise HTTPException(status_code=502, detail=result.error_message)

        case EndSessionStatus.ERROR:
            raise HTTPException(status_code=500, detail=result.error_message)


# --- ADMIN ENDPOINTS ---

@router.get("/admin/dashboard")
async def admin_dashboard(
    admin: Participant = Depends(verify_admin),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get consolidated admin dashboard data.

    Requires admin participant authentication.

    Returns:
        - System health indicators
        - Session counts (active, waiting, completed)
        - Participant data (humans and AI models)
        - All sessions with details
        - Recent activity timeline
    """
    # Session categorization
    sessions_by_status = {
        "running": [],
        "started": [],
        "finished": []
    }

    for session in session_manager.sessions.values():
        status_key = session.status.value.lower()
        if status_key in sessions_by_status:
            sessions_by_status[status_key].append(session)

    # Build session data
    sessions_data = []
    for session in session_manager.sessions.values():
        session_dict = {
            "session_id": session.session_id,
            "status": session.status.value,
            "topic": session.topic,
            "participants": {
                "participant1": {
                    "id": session.participant1_id,
                    "joined": session.participant1_joined,
                    "is_ai": session_manager.participants.get(session.participant1_id).is_ai if session_manager.participants.get(session.participant1_id) else False
                },
                "participant2": {
                    "id": session.participant2_id,
                    "joined": session.participant2_joined,
                    "is_ai": session_manager.participants.get(session.participant2_id).is_ai if session_manager.participants.get(session.participant2_id) else False
                }
            },
            "message_count": len(session.dialogue_history),
            "current_turn": session.current_turn.value,
            "created_at": session.created_at.isoformat()
        }

        # Fetch Persuasio logs if session exists (limited to last 100 entries)
        if session.persuasio_exists:
            try:
                log_data = await session_manager.persuasio_client.get_session_log(session.session_id, limit=100)
                session_dict["logs"] = log_data.get("log", [])
            except Exception as e:
                session_dict["logs"] = [f"Error fetching logs: {str(e)}"]
        else:
            session_dict["logs"] = ["No Persuasio session created yet"]

        sessions_data.append(session_dict)

    # Build participant data
    participants_data = []
    for p in session_manager.participants.values():
        if p.is_admin:
            continue  # Skip admin participants

        participants_data.append({
            "id": p.participant_id,
            "type": "ai" if p.is_ai else "human",
            "is_authenticated": p.is_authenticated,
            "current_session": p.current_session,
            "is_ai": p.is_ai,
            "auth_code": p.auth_code if hasattr(p, 'auth_code') else None,
            "message_count": p.message_count,
        })

    # Recent activities (last 20)
    recent_activities = []
    for event in list(session_manager.activity_log)[-20:]:
        recent_activities.append({
            "timestamp": event.timestamp.strftime("%H:%M:%S"),
            "event_type": event.event_type,
            "severity": event.severity,
            "participant_id": event.participant_id,
            "message": event.message,
            "session_id": event.session_id
        })

    return {
        "session_counts": {
            "running": len(sessions_by_status["running"]),
            "started": len(sessions_by_status["started"]),
            "finished": len(sessions_by_status["finished"]),
            "total": len(session_manager.sessions)
        },
        "participant_counts": {
            "human": len([p for p in session_manager.participants.values() if not p.is_ai and not p.is_admin]),
            "ai": len([p for p in session_manager.participants.values() if p.is_ai]),
            "total_registered": len(session_manager.participants) - 1  # Exclude admin
        },
        "sessions": sessions_data,
        "participants": participants_data,
        "recent_activities": recent_activities,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/admin/health")
async def admin_health(
    admin: Participant = Depends(verify_admin),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Extended health check with system metrics for admin.

    Requires admin participant authentication.

    Returns:
        System stats including participant counts, session counts, memory usage
    """
    return {
        "status": "healthy",
        "service": "session-manager",
        "admin_access": True,
        "system_stats": {
            "total_participants": len(session_manager.participants),
            "total_sessions": len(session_manager.sessions),
            "activity_log_size": len(session_manager.activity_log)
        },
        "memory_usage": {
            "sessions_in_memory": len(session_manager.sessions),
            "participants_in_memory": len(session_manager.participants)
        }
    }


@router.get("/admin/persuasio-health")
async def admin_persuasio_health(
    admin: Participant = Depends(verify_admin),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Check Persuasio backend connection and health.

    Requires admin participant authentication.

    Returns:
        Persuasio connection status, URL, and session counts if healthy
    """
    try:
        # Try to fetch sessions from Persuasio
        sessions_data = await session_manager.persuasio_client.get_sessions()

        # Count sessions
        finished_count = len(sessions_data.get("finished", []))
        ongoing_count = len(sessions_data.get("ongoing", []))

        return {
            "persuasio_status": "healthy",
            "persuasio_url": session_manager.persuasio_base_url,
            "sessions": {
                "finished_count": finished_count,
                "ongoing_count": ongoing_count
            }
        }
    except Exception as e:
        # Return unhealthy status with error details
        return {
            "persuasio_status": "unhealthy",
            "persuasio_url": session_manager.persuasio_base_url,
            "error": str(e),
            "error_type": type(e).__name__
        }
