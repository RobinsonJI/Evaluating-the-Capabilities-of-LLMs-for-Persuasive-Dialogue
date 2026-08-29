from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Union, Dict
from collections import deque
import traceback
from fastapi import BackgroundTasks

from datetime import datetime as dt

import logging

# --- DATA MODELS ---
from session_manager.data import TemplateDB, TemplateLogs
from session_manager.models import (
    Session,
    Participant,
    DialogueTurn,
    ActivityEvent,
    SessionStatus,
    State,
    PersuasioResponse,
    SpeakerOrder,
    AuthStatus,
    AuthResult,
    ValidateParticipantTurnResult,
    ValidateParticipantResult,
    ValidateSessionResult,
    JoinSessionStatus,
    JoinSessionResult,
    SendMessageStatus,
    SendMessageResult,
    EndSessionStatus,
    EndSessionResult
)

# --- PERSUASIO CLIENT ---
from session_manager.persuasio_client import PersuasioClient


class SessionManager:
    def __init__(
        self,
        db: TemplateDB,
        logs: TemplateLogs,
        persuasio_base_url: str,
        persuasio_api_key: str):
        """
        Args:
            db (TemplateDB): Either a LocalDB or SQLDB instance.
            logs (TemplateLogs): Either a LocalLogs or SQLLogs instance.
            persuasio_base_url (str): Base URL for the Persuasio API.
            persuasio_api_key (str): API key for the Persuasio API.
        """
        self.db = db
        self.logs = logs
        
        # Init Persuasio client
        self.persuasio_base_url = persuasio_base_url
        self.persuasio_api_key = persuasio_api_key
        self._init_persuasio_client()
        
        # Load sessions and participants from DB
        self._setup_sessions()
        self._setup_participants()
        self._setup_auth()

        # Monitoring setup
        self.activity_log: deque = deque(maxlen=100)  # Keep last 100 activities
        self.system_start_time = dt.now()
        self.logger = logging.getLogger(__name__)
        self._log_activity("system", "SYSTEM", "Session manager initialised")
        
    # --- INIT METHODS ---
    def _init_persuasio_client(self):
        """Initializes PersuasioClient."""
        self.persuasio_client = PersuasioClient(
            base_url=self.persuasio_base_url,
            api_key=self.persuasio_api_key
        )
        
    def _setup_sessions(self):
        """Read sessions from DB and initialise objects."""
        sessions = self.db.load_sessions()
        self.sessions = {session.session_id: session for session in sessions}
        # Map for any ID translations if needed.
        # This is used by the session restore logic to map the original, public-facing
        # session_id to a new backend thread_id if the persuasio service restarts
        # and a session has to be restored.
        self.session_id_map = dict()  # Map for any ID translations if needed
    
    def _setup_participants(self):
        """Read participants from DB and initialise objects."""
        participants = self.db.load_participants()
        self.participants = {participant.participant_id: participant for participant in participants}
        self.human_participants = {pid: p for pid, p in self.participants.items() if not p.is_ai}
        self.ai_participants = {pid: p for pid, p in self.participants.items() if p.is_ai}

    def _setup_auth(self):
        """Create lookup for auth codes to participant IDs."""
        self.auth_lookup = {}
        for participant in self.participants.values():
            if participant.auth_code:  # Only add if auth_code exists (humans only)
                self.auth_lookup[participant.auth_code] = participant.participant_id

    # --- PARTICIPANT MANAGEMENT ---

    def authenticate_participant(self, code: str) -> AuthResult:
        """
        Authenticate using code.

        Args:
            code: authentication code

        Returns:
            AuthResult with status and participant if successful
        """
        participant_id = self.auth_lookup.get(code)
        if participant_id:
            participant = self.participants.get(participant_id)
            if participant:
                participant.is_authenticated = True
                return AuthResult(
                    status=AuthStatus.SUCCESS,
                    participant=participant
                )

        return AuthResult(
            status=AuthStatus.INVALID_CODE,
            error_message="Invalid authentication code"
        )
    
    # --- SESSION MANAGEMENT ---

    # --- ATOMIC VALIDATORS (Reusable building blocks) ---

    def _validate_participant_exists(self, participant_id: str) -> ValidateParticipantResult:
        """
        Validate that participant exists.

        Args:
            participant_id: ID of participant to check

        Returns:
            ValidateParticipantResult with participant if exists
        """
        participant = self.participants.get(participant_id)
        if not participant:
            return ValidateParticipantResult(
                is_valid=False,
                error_message="Participant not found"
            )
        return ValidateParticipantResult(
            is_valid=True,
            participant=participant
        )

    def _validate_session_exists(self, session_id: str) -> ValidateSessionResult:
        """
        Validate that session exists.

        Args:
            session_id: ID of session to check

        Returns:
            ValidateSessionResult with session if exists
        """
        session = self.sessions.get(session_id)
        if not session:
            return ValidateSessionResult(
                is_valid=False,
                error_message="Session not found"
            )
        return ValidateSessionResult(
            is_valid=True,
            session=session
        )

    def _validate_participant_authenticated(self, participant: Participant) -> ValidateParticipantResult:
        """
        Validate that participant is authenticated (ready).

        Args:
            participant: Participant to check

        Returns:
            ValidateParticipantResult indicating if authenticated
        """
        if not participant.is_ready:
            return ValidateParticipantResult(
                is_valid=False,
                error_message="Participant not authenticated"
            )
        return ValidateParticipantResult(
            is_valid=True,
            participant=participant
        )

    def _validate_participant_in_session(
        self,
        participant_id: str,
        session: Session
    ) -> ValidateParticipantResult:
        """
        Validate that participant is assigned to the session.

        Args:
            participant_id: ID of participant
            session: Session to check

        Returns:
            ValidateParticipantResult indicating if participant is assigned
        """
        if participant_id not in [session.participant1_id, session.participant2_id]:
            return ValidateParticipantResult(
                is_valid=False,
                error_message="Participant not assigned to this session"
            )
        return ValidateParticipantResult(is_valid=True)

    # --- COMPOSITE VALIDATORS (Method-specific) ---

    def _validate_participant_turn(self, participant_id: str, session_id: str) -> ValidateParticipantTurnResult:
        """
        Validate participant is in session and it's their turn.

        Args:
            participant_id: ID of participant to validate
            session_id: ID of session

        Returns:
            ValidateParticipantTurnResult with validation status
        """
        participant = self.participants.get(participant_id)
        if not participant or participant.current_session != session_id:
            return ValidateParticipantTurnResult(
                is_valid=False,
                error_status=SendMessageStatus.NOT_IN_SESSION,
                error_message="Participant not in session"
            )

        session = self.sessions.get(session_id)
        if not session:
            return ValidateParticipantTurnResult(
                is_valid=False,
                error_status=SendMessageStatus.INVALID_SESSION,
                error_message="Session not found"
            )

        if not session.can_participant_speak(participant_id):
            return ValidateParticipantTurnResult(
                is_valid=False,
                error_status=SendMessageStatus.NOT_YOUR_TURN,
                error_message="Not your turn to speak"
            )

        return ValidateParticipantTurnResult(
            is_valid=True,
            participant=participant,
            session=session
        )

    def _validate_for_join_session(
        self,
        participant_id: str,
        session_id: str
    ) -> JoinSessionResult:
        """
        Run all validations needed for join_session.

        Args:
            participant_id: ID of participant joining
            session_id: ID of session

        Returns:
            JoinSessionResult with JOINED status if valid, or error status if any validation fails.
            If successful, includes validated participant and session objects.
        """
        # Validate participant exists
        participant_check = self._validate_participant_exists(participant_id)
        if not participant_check.is_valid:
            return JoinSessionResult(
                status=JoinSessionStatus.INVALID_PARTICIPANT,
                error_message=participant_check.error_message
            )

        participant = participant_check.participant

        # Validate authenticated
        auth_check = self._validate_participant_authenticated(participant)
        if not auth_check.is_valid:
            return JoinSessionResult(
                status=JoinSessionStatus.NOT_AUTHENTICATED,
                error_message=auth_check.error_message
            )

        # Validate session exists
        session_check = self._validate_session_exists(session_id)
        if not session_check.is_valid:
            return JoinSessionResult(
                status=JoinSessionStatus.INVALID_SESSION,
                error_message=session_check.error_message
            )

        session = session_check.session

        # Validate participant assigned to session
        assignment_check = self._validate_participant_in_session(participant_id, session)
        if not assignment_check.is_valid:
            return JoinSessionResult(
                status=JoinSessionStatus.NOT_ASSIGNED,
                error_message=assignment_check.error_message
            )

        # All validations passed
        return JoinSessionResult(
            status=JoinSessionStatus.JOINED,
            participant=participant,
            session=session
        )

    def _validate_for_end_session(self, session_id: str) -> tuple[bool, Optional[Session], str]:
        """
        Validate session exists for ending.

        Args:
            session_id: ID of session to end

        Returns:
            Tuple of (is_valid, session, error_message)
        """
        session_check = self._validate_session_exists(session_id)
        if not session_check.is_valid:
            return False, None, session_check.error_message
        return True, session_check.session, ""


    async def join_session(
        self,
        participant_id: str,
        session_id: str,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> JoinSessionResult:
        """
        Join session with participant ID.

        Args:
            participant_id: ID of participant joining
            session_id: ID of session to join
            background_tasks: Optional FastAPI BackgroundTasks for async operations

        Returns:
            JoinSessionResult with status and optional participant/session data
        """
        # Validate all preconditions for joining
        validation = self._validate_for_join_session(participant_id, session_id)

        # Early return if validation failed
        if validation.status != JoinSessionStatus.JOINED:
            return validation

        # Extract validated objects
        participant = validation.participant
        session = validation.session

        #  Mark participant as joined
        participant.current_session = session_id

        # Update session to track who has joined
        if participant_id == session.participant1_id:
            session.participant1_joined = True
        elif participant_id == session.participant2_id:
            session.participant2_joined = True

        # Update timestamp for change detection
        session.last_updated = dt.now()

        # Log activity
        self._log_activity("session", participant_id, f"Joined session {session_id}", session_id)

        # Auto-join AI participants
        self._auto_join_ai_participants(session)

        # Create persuasio session when both participants have joined
        if session.participant1_joined and session.participant2_joined:
            # Only update status if not already finished
            if session.status != SessionStatus.FINISHED:
                session.status = SessionStatus.RUNNING
                # Update timestamp again for status change
                session.last_updated = dt.now()

            # Check if AI starts first (first_speaker is AI)
            first_participant = self.participants.get(session.participant1_id)
            if first_participant and first_participant.is_ai:
                # Only launch Persuasio if this is a fresh session
                if not session.persuasio_exists and not session.dialogue_history:
                    # Schedule AI-first session launch in background to prevent blocking
                    if background_tasks:
                        # Use background task - returns immediately
                        background_tasks.add_task(self._launch_ai_first_session, session_id)
                        self._log_activity(
                            "session",
                            "SYSTEM",
                            f"Scheduled AI-first session launch for {session_id}",
                            session_id
                        )
                    else:
                        # Fallback: run synchronously if background_tasks not available
                        await self._launch_ai_first_session(session_id)

        return JoinSessionResult(
            status=JoinSessionStatus.JOINED,
            participant=participant,
            session=session
        )

    async def _launch_ai_first_session(self, session_id: str):
        """
        Background task to launch Persuasio session when AI starts first.

        Decouples AI message generation from join to prevent timing differences
        that would reveal AI identity.

        Args:
            session_id: ID of session to launch
        """
        try:
            session = self.sessions.get(session_id)
            if not session:
                self._log_activity(
                    "error",
                    "SYSTEM",
                    f"Cannot launch AI-first session: session {session_id} not found",
                    session_id
                )
                return

            # AI starts - create session with empty utterance to trigger AI response
            response = await self._launch_persuasio_session(session_id, first_utterance="")

            # Update commitments from initial persuasio response
            session.update_commitments(
                response.first_speaker_commitments,
                response.second_speaker_commitments
            )

            # Store typical replies in session for persistence
            session.first_speaker_typical_replies = response.first_speaker_typical_replies_for_second_speakers_response
            session.second_speaker_typical_replies = response.second_speaker_typical_replies_for_first_speakers_response

            # Extract and add AI's first utterance (structured turn with classifications)
            ai_turn = self._extract_ai_turn(response, SpeakerOrder.FIRST_SPEAKER)
            if ai_turn:
                session.add_dialogue_turn(ai_turn)
                self._log_and_save_message(session.participant1_id, session_id)

        except Exception as e:
            self._log_activity(
                "error",
                "SYSTEM",
                f"Failed to create persuasio session for {session_id}: {str(e)}",
                session_id
            )

    def _auto_join_ai_participants(self, session: Session):
        """Automatically join AI participants to session"""
        for participant_id in [session.participant1_id, session.participant2_id]:
            participant = self.participants.get(participant_id)
            if participant and participant.is_ai:
                participant.current_session = session.session_id
                if participant_id == session.participant1_id:
                    session.participant1_joined = True
                elif participant_id == session.participant2_id:
                    session.participant2_joined = True
    
    async def send_message(
        self,
        participant_id: str,
        session_id: str,
        message: str) -> SendMessageResult:
        """
        Process a message from a participant.

        Args:
            participant_id: ID of participant sending message
            session_id: ID of session
            message: Message content

        Returns:
            SendMessageResult with status and optional data
        """
        # Validate participant and turn
        validation = self._validate_participant_turn(participant_id, session_id)
        if not validation.is_valid:
            return SendMessageResult(
                status=validation.error_status,
                error_message=validation.error_message
            )

        participant = validation.participant
        session = validation.session

        try:
            # TODO: Use a helper to get the correct persuasio ID, e.g.:
            # persuasio_id = self._get_persuasio_id(session.session_id)
            # The client method would then need to accept this ID.

            # Launch or update persuasio session
            if not session.persuasio_exists:
                response = await self._launch_persuasio_session(session_id, first_utterance=message)
            else:
                # TODO: Implement session restore logic (Scenario B)
                # Wrap this call in a try...except block to handle 404 Not Found errors from httpx.
                # In the except block:
                # 1. Call `restored_data = await self.persuasio_client.restore_session(session)`.
                # 2. Update the mapping: `self.session_id_map[session.session_id] = restored_data["new_session_id"]`.
                # 3. Retry this `update_session` call, ensuring you use the new ID.
                response = await self.persuasio_client.update_session(
                    session=session,
                    speaker_id=participant_id,
                    utterance=message
                )

            if not response:
                return SendMessageResult(
                    status=SendMessageStatus.BACKEND_ERROR,
                    error_message="No response from Persuasio"
                )

            # Determine current speaker order (before turn switches)
            current_speaker_order = session.current_turn

            # Extract and add human message with classifications from Persuasio's dialogue_history
            human_turn = self._extract_turn_from_dialogue_history(response, current_speaker_order)
            if human_turn:
                session.add_dialogue_turn(human_turn)  # This switches turn
                self._log_and_save_message(participant_id, session_id)
            else:
                # Fallback to manual creation if extraction fails
                self._log_activity(
                    "warning",
                    "SYSTEM",
                    f"Could not extract human turn from dialogue_history for session {session_id}, using fallback",
                    session_id
                )
                human_turn = session.add_message(participant_id, message)
                self._log_and_save_message(participant_id, session_id)

            # Update commitments from persuasio response
            session.update_commitments(
                response.first_speaker_commitments,
                response.second_speaker_commitments
            )

            # Store typical replies in session for persistence
            session.first_speaker_typical_replies = response.first_speaker_typical_replies_for_second_speakers_response
            session.second_speaker_typical_replies = response.second_speaker_typical_replies_for_first_speakers_response

            # Extract typical replies for the human participant
            # If human is first_speaker, they see typical replies for second_speaker's response
            # If human is second_speaker, they see typical replies for first_speaker's response
            typical_replies = []
            if participant_id == session.participant1_id:
                typical_replies = response.second_speaker_typical_replies_for_first_speakers_response
            elif participant_id == session.participant2_id:
                typical_replies = response.first_speaker_typical_replies_for_second_speakers_response

            # Extract AI response if next speaker is AI (BEFORE checking termination to capture final messages)
            next_speaker_order = session.current_turn  # Turn has been switched by add_dialogue_turn
            next_participant_id = (session.participant1_id
                                 if next_speaker_order == SpeakerOrder.FIRST_SPEAKER
                                 else session.participant2_id)
            current_turn_participant = self.participants.get(next_participant_id)

            if current_turn_participant and current_turn_participant.is_ai:
                ai_turn = self._extract_ai_turn(response, next_speaker_order)
                if ai_turn:
                    session.add_dialogue_turn(ai_turn)
                    self._log_and_save_message(next_participant_id, session_id)

            # Check if session ended (AFTER extracting final AI message)
            session_ended = self._check_and_handle_termination(session_id, response)

            # Extract termination reason if session ended
            termination_reason = None
            if session_ended and response.reason_for_dialogue_termination:
                termination_reason = response.reason_for_dialogue_termination.get("reason", "Unknown")

            # Return with typical replies, commitment data, and termination info
            return SendMessageResult(
                status=SendMessageStatus.SENT,
                human_turn=human_turn,
                typical_replies=typical_replies,
                commitments={
                    "first_speaker": session.first_speaker_commitments,
                    "second_speaker": session.second_speaker_commitments
                },
                session_ended=session_ended,
                termination_reason=termination_reason
            )

        except Exception as e:
            self._log_activity(
                "error",
                "SYSTEM",
                f"Error in send_message: [{type(e).__name__}] {repr(e)}\nTraceback:\n{traceback.format_exc()}",
                session_id
            )
            error_msg = str(e) if str(e) else "No error message"
            return SendMessageResult(
                status=SendMessageStatus.BACKEND_ERROR,
                error_message=f"[{type(e).__name__}] {error_msg}"
            )
        
    async def _launch_persuasio_session(
        self,
        session_id: str,
        first_utterance: Optional[str] = None
    ) -> PersuasioResponse:
        """
        Launch session in persuasio backend.

        Args:
            session_id: ID of session to launch
            first_utterance: First message if human starts, None if AI starts

        Returns:
            PersuasioResponse: Response from persuasio
        """
        session = self.sessions[session_id]

        # Pass session object directly to persuasio client
        # TODO: Implement session restore logic (Scenario A)
        # Wrap this call in a try...except block to handle the "session already exists" error from persuasio.
        # In the except block:
        # 1. Call `state_dict = await self.persuasio_client.get_session_state(session_id)`.
        # 2. Call `self._reconstruct_session_from_state(state_dict)` to update the in-memory session.
        # 3. Return the appropriate PersuasioResponse based on the reconstructed state.
        response = await self.persuasio_client.create_session(
            session=session,
            first_utterance=first_utterance
        )

        # Mark that persuasio session exists
        session.persuasio_exists = True

        # Log
        self._log_activity(
            event_type="session",
            participant_id="SYSTEM",
            message=f"Launched persuasio session",
            session_id=session_id,
            severity="info"
        )

        return response
    
    def _extract_turn_from_dialogue_history(
        self,
        persuasio_response: PersuasioResponse,
        speaker_order: SpeakerOrder
    ) -> Optional[DialogueTurn]:
        """
        Extract a dialogue turn from persuasio response for given speaker.
        Gets the full structured turn with utterance classifications.

        Args:
            persuasio_response: Response from persuasio
            speaker_order: Which speaker to extract (FIRST_SPEAKER or SECOND_SPEAKER)

        Returns:
            DialogueTurn: The speaker's dialogue turn with classifications, or None
        """
        # Use the speaker name from persuasio response (which may differ from session_manager IDs)
        if speaker_order == SpeakerOrder.FIRST_SPEAKER:
            persuasio_speaker_name = persuasio_response.first_speaker
        else:
            persuasio_speaker_name = persuasio_response.second_speaker

        # Get the latest turn from dialogue_history for this speaker
        # Dialogue history is ordered, so iterate backwards to find latest
        for turn in reversed(persuasio_response.dialogue_history):
            if turn.speaker == persuasio_speaker_name:
                return turn
        return None

    def _extract_ai_turn(
        self,
        persuasio_response: PersuasioResponse,
        speaker_order: SpeakerOrder
    ) -> Optional[DialogueTurn]:
        """
        Extract AI dialogue turn from persuasio response for given speaker.
        Gets the full structured turn with utterance classifications.

        Args:
            persuasio_response: Response from persuasio
            speaker_order: Which speaker to extract (FIRST_SPEAKER or SECOND_SPEAKER)

        Returns:
            DialogueTurn: The AI's dialogue turn with classifications, or None
        """
        return self._extract_turn_from_dialogue_history(persuasio_response, speaker_order)

    def _check_and_handle_termination(
        self,
        session_id: str,
        persuasio_response: PersuasioResponse) -> bool:
        """
        Check persuasio response for termination and handle if needed.

        Args:
            session_id: ID of the session
            persuasio_response: Response from persuasio API

        Returns:
            True if session was terminated, False otherwise
        """
        if persuasio_response.reason_for_dialogue_termination:
            reason = persuasio_response.reason_for_dialogue_termination.get("reason", "Unknown")
            self._end_session(session_id, reason, ended_by="SYSTEM")
            return True
        return False
        
    async def user_end_session(self, session_id: str, participant_id: str, message: str) -> EndSessionResult:
        """
        Process a user's request to end a session.

        Args:
            session_id: ID of session to end
            participant_id: ID of participant ending session
            message: Message/reason for ending session

        Returns:
            EndSessionResult with status
        """
        # Validate session exists
        is_valid, session, error_msg = self._validate_for_end_session(session_id)
        if not is_valid:
            return EndSessionResult(
                status=EndSessionStatus.INVALID_SESSION,
                error_message=error_msg
            )

        try:
            # Call persuasio_client.end_session
            await self.persuasio_client.end_session(session_id)
        except Exception as e:
            self._log_activity(
                "error",
                "SYSTEM",
                f"Failed to end session {session_id} in Persuasio: {str(e)}",
                session_id
            )
            return EndSessionResult(
                status=EndSessionStatus.PERSUASIO_ERROR,
                session_id=session_id,
                error_message=f"Failed to end session in Persuasio: {str(e)}"
            )

        # End session internally
        self._end_session(session_id, reason=message, ended_by=participant_id)

        return EndSessionResult(
            status=EndSessionStatus.SUCCESS,
            session_id=session_id
        )
        

    def _end_session(self, session_id: str, reason: str, ended_by: str = "SYSTEM"):
        """
        End a session internally.

        Args:
            session_id (str): ID of session to end.
            reason (str): Reason for ending the session.
            ended_by (str, optional): Participant ID who ended the session. Defaults to "SYSTEM".
        """
        session = self.sessions.get(session_id)
        if session:
            session.status = SessionStatus.FINISHED
            session.end_reason = reason
            # Update timestamp for change detection
            session.last_updated = dt.now()
            # Save dialogue history to DB
            self.db.save_dialogue(session_id, session.dialogue_history)
            # Update session in DB
            self.db.update_session(session)
        
        # Log activity
        self._log_activity(
            event_type="session_ended",
            participant_id=ended_by,
            message=f"Session {session_id} ended: {reason}",
            session_id=session_id,
            severity="info"
        )
    
    async def restore_session(self, session_id: str, state: Optional[State] = None) -> State:
        """
        Restore a session from an (optional) state.

        Args:
            session_id (str): ID of session to restore
            state (Optional[State], optional): State to restore. Defaults to None.

        Returns:
            State: Restored state
        """
        # TODO: restore_session
        # Method used when we try to update a session that doesn't exist on persuasio
        # If no state is provded, persuasio will load state from its own DB
        # Returns the restored state
        restored_state = await self.persuasio_client.restore_session(session_id, state)
        
        # TODO - revised ID mapping? 
        new_session_id = restored_state.session_id
        self.session_id_map[session_id] = new_session_id
        #self.sessions[new_session_id] = something?
        
        self._log_activity(
            event_type="session_restored",
            participant_id="SYSTEM",
            message=f"Session {session_id} restored to Persuasio as {new_session_id}",
            session_id=session_id,
            severity="info"
        )

        return restored_state
        
        
    # --- MONITORING ---
    
    def _log_and_save_message(
        self,
        participant_id: str,
        session_id: str):
        """Log a message and save the dialogue history to DB"""
        # Log message
        self._log_activity(
            event_type="message_sent",
            participant_id=participant_id,
            message=f"Participant {participant_id} sent message in session {session_id}",
            session_id=session_id,
            severity="info"
        )

        # Save dialogue history
        session = self.sessions.get(session_id)
        if session:
            self.db.save_dialogue(session_id, session.dialogue_history)
        
    
    def _log_activity(
        self,
        event_type: str,
        participant_id: str,
        message: str,
        session_id: Optional[str] = None,
        severity: str = "info"):
        """Log an activity event to both activity log and Python logger"""
        # Create activity event for dashboard
        event = ActivityEvent(
            timestamp=dt.now(),
            event_type=event_type,
            severity=severity,
            participant_id=participant_id,
            message=message,
            session_id=session_id
        )
        self.activity_log.append(event)

        # Update participant activity timestamp
        if participant_id != "SYSTEM" and participant_id in self.participants:
            self.participants[participant_id].last_activity = event.timestamp

        # Log to Python logger based on severity
        log_message = f"[{event_type}] {participant_id}: {message}"
        if session_id:
            log_message += f" (session: {session_id})"
            
        # Write to logs store
        self.logs.save_log(event)

        if severity == "debug":
            self.logger.debug(log_message)
        elif severity == "info":
            self.logger.info(log_message)
        elif severity == "warning":
            self.logger.warning(log_message)
        elif severity == "error":
            self.logger.error(log_message)
        else:
            self.logger.info(log_message)  # Default to info

    def check_database_health(self) -> dict:
        """
        Check health of database connections and log results.

        Returns:
            Dictionary with health status for both DB and Logs connections
        """
        from .data import SQLDB, SQLLogs

        results = {
            "timestamp": dt.now().isoformat(),
            "db_healthy": False,
            "logs_healthy": False,
            "db_error": None,
            "logs_error": None
        }

        # Check main database connection (if SQLDB)
        if isinstance(self.db, SQLDB):
            db_healthy, db_error = self.db.check_connection_health()
            results["db_healthy"] = db_healthy
            results["db_error"] = db_error

            # Log the result
            if db_healthy:
                self._log_activity(
                    event_type="system",
                    participant_id="SYSTEM",
                    message="Database connection healthy",
                    severity="info"
                )
            else:
                self._log_activity(
                    event_type="system",
                    participant_id="SYSTEM",
                    message=f"Database connection unhealthy: {db_error}",
                    severity="error"
                )
        else:
            results["db_healthy"] = True
            results["db_error"] = "Using LocalDB (no health check needed)"

        # Check logs database connection (if SQLLogs)
        if isinstance(self.logs, SQLLogs):
            logs_healthy, logs_error = self.logs.check_connection_health()
            results["logs_healthy"] = logs_healthy
            results["logs_error"] = logs_error

            # Log the result (careful: if logs DB is down, this will fail)
            if logs_healthy:
                self._log_activity(
                    event_type="system",
                    participant_id="SYSTEM",
                    message="Logs database connection healthy",
                    severity="info"
                )
            else:
                # If logs DB is down, log to Python logger only
                self.logger.error(f"Logs database connection unhealthy: {logs_error}")
        else:
            results["logs_healthy"] = True
            results["logs_error"] = "Using LocalLogs (no health check needed)"

        return results

    # --- RESTORE LOGIC HELPERS (TODO) ---

    def _get_persuasio_id(self, session_id: str) -> str:
        """
        Gets the current backend thread ID for a given session_id,
        consulting the session_id_map for restored sessions.
        """
        # TODO: Implement ID lookup
        # return self.session_id_map.get(session_id, session_id)
        return session_id

    def _reconstruct_session_from_state(self, state: dict):
        """
        Rebuilds an in-memory Session object from a raw persuasio state dictionary.
        """
        # TODO: Implement reconstruction logic
        # 1. Get/create Session object.
        # 2. Populate dialogue_history, commitments, status, etc., from the state dict.
        pass