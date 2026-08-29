import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from client.session_manager_client import SessionManagerClient
from client.models import LoginResult, JoinSessionResult, SendMessageResult, RefreshResult, AdminRefreshResult, ChatItem, DataForOneDialogueTurn, EndSessionResult, ActivityEvent
from client.logs import TemplateLogs

# Error code to friendly message mapping
ERROR_MESSAGES = {
    "NOT_YOUR_TURN": "Please wait for your opponent to respond",
    "INVALID_SESSION": "Session not found. Please check your session code",
    "BACKEND_ERROR": "Something went wrong. Please try again",
    "NOT_IN_SESSION": "You need to join a session first",
    "NOT_AUTHENTICATED": "Please log in first",
    "NOT_ASSIGNED": "You are not assigned to this session",
    "INVALID_PARTICIPANT": "Invalid participant"
}

class BasePersuasuiClient(ABC):
    
    def __init__(
        self,
        mode: str,
        refresh_interval: int,
        server_name: str,
        server_port: int,
        api_base_url: str,
        api_key: str,
        logs: TemplateLogs,
        ai_response_delay: int = 15):
        """
        Initialise The Abstract Base Class for a Persuasui Frontend Client.

        This class contains all framework-agnostic logic for the client application,
        including state management, API communication, and data processing. It is
        designed to be subclassed by a framework-specific client (e.g., for Gradio,
        Flet, etc.) that will handle the UI rendering and event loop.

        Args:
            mode (str): The execution mode, either 'dev' or 'prod'. Defaults to 'prod'.
            refresh_interval (int): Auto-refresh interval in seconds for polling session state.
            server_name (str): Hostname to run the web server on.
            server_port (int): Port to run the web server on.
            api_base_url (str): Base URL for the Session Manager API.
            api_key (str): API key for authentication with the Session Manager API.
            logs (TemplateLogs): Logging implementation (LocalLogs or SQLLogs).
            ai_response_delay (int): Simulated delay in seconds before showing AI responses.
                This helps blind users to whether their opponent is human or AI.
        """
        self.mode = mode.lower()
        self.refresh_interval = refresh_interval
        self.server_name = server_name
        self.server_port = server_port
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.logs = logs
        self.ai_response_delay = ai_response_delay

        # Setup session manager client
        self.sm_client = self._setup_session_manager_client()
        
    # --- SETUP METHODS ---
        
    def _setup_session_manager_client(self) -> SessionManagerClient:
        """Initialise the Session Manager HTTP Client."""
        return SessionManagerClient(
            base_url=self.api_base_url,
            api_key=self.api_key
        )

    def _log_event(
        self,
        event_type: str,
        participant_id: str,
        message: str,
        session_id: Optional[str] = None,
        severity: str = "info"
    ):
        """
        Log an activity event to persistent storage and Python logger.

        Args:
            event_type: Type of event ('login', 'session', 'message', 'system')
            participant_id: ID of the participant
            message: Log message
            session_id: Optional session ID
            severity: Log severity ('debug', 'info', 'warning', 'error')
        """
        event = ActivityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            participant_id=participant_id,
            message=message,
            session_id=session_id
        )

        # Write to persistent storage
        self.logs.save_log(event)

        # Also log to Python logger for console output
        logger = logging.getLogger(__name__)
        log_msg = f"[{event_type}] {participant_id}: {message}"
        if session_id:
            log_msg += f" (session: {session_id})"

        if severity == "debug":
            logger.debug(log_msg)
        elif severity == "info":
            logger.info(log_msg)
        elif severity == "warning":
            logger.warning(log_msg)
        elif severity == "error":
            logger.error(log_msg)
        else:
            logger.info(log_msg)

    # --- GENERIC DATA TRANSFORMATION HELPERS ---

    def _get_chat_data(self, dialogue_history: list, participant_id: str) -> List[ChatItem]:
        """
        Transforms the raw dialogue history from the API into a clean, structured
        list of ChatItem models for easier UI consumption.

        Args:
            dialogue_history (list): The raw list of turn dictionaries from the API.
            participant_id (str): The ID of the current user to determine message alignment.

        Returns:
            List[ChatItem]: A list of Pydantic ChatItem models.
        """
        chat_items = []
        for i, turn_dict in enumerate(dialogue_history):
            # Validate and parse the raw turn dictionary into our client-side model
            turn = DataForOneDialogueTurn(**turn_dict)
            chat_items.append(ChatItem(
                is_user=turn.speaker == participant_id,
                speaker_id=turn.speaker,
                content=turn.sentences_no_utterance_types,
                turn_number=i + 1,
                timestamp=turn.timestamp
            ))
        return chat_items

    def _determine_status_message(self, session: dict, can_speak: bool) -> str:
        """
        Determines the appropriate UI status message based on the current session state.
        Never reveals participant IDs.

        Args:
            session (dict): The raw session dictionary from the API.
            can_speak (bool): Whether the current user has the turn to speak.

        Returns:
            str: A user-facing status message.
        """
        status = session.get("status", "unknown")
        if status == "Started.":
            if not session.get("participant1_joined") or not session.get("participant2_joined"):
                return "Waiting for opponent to join..."
            else:
                return "Ready to begin."
        elif status == "Running.":
            if can_speak:
                return "Your turn"
            else:
                return "Waiting for opponent to respond..."
        elif status == "Finished.":
            return f"Debate finished: {session.get('end_reason', 'Unknown reason')}"
        else:
            return f"Status: {status}"

    # --- GENERIC EVENT HANDLERS (CORE LOGIC) ---
    
    async def handle_login(self, code: str) -> LoginResult:
        """
        Handles the user login flow.

        Args:
            code (str): The 6-digit login code entered by the user.

        Returns:
            LoginResult: A Pydantic model containing the outcome of the login attempt,
                         including participant info or an error message.
        """
        if not code or not code.strip():
            return LoginResult(error="Please enter a valid code.")

        # Log login attempt
        self._log_event(
            event_type="login",
            participant_id=code.strip(),
            message="Login attempt",
            severity="info"
        )

        data, error = await self.sm_client.login(code.strip())
        if error:
            # Log login failure
            self._log_event(
                event_type="login",
                participant_id=code.strip(),
                message=f"Login failed: {error}",
                severity="error"
            )
            return LoginResult(error=f"Login failed: {error}")

        participant_id = data.get("participant_id")

        # Log login success
        self._log_event(
            event_type="login",
            participant_id=participant_id,
            message="Login successful",
            severity="info"
        )

        return LoginResult(
            error=None,
            participant_id=participant_id,
            is_admin=participant_id == 'ADMIN',
            message=data.get("message")
        )
    
    async def handle_join_session(self, participant_id: str, session_code: str) -> JoinSessionResult:
        """
        Handles a participant joining a session.

        Args:
            participant_id (str): The ID of the participant who is joining.
            session_code (str): The code for the session to join.

        Returns:
            JoinSessionResult: A Pydantic model containing the outcome, including
                               the user context for state management or an error.
        """
        if not participant_id:
            return JoinSessionResult(error="Cannot join session, not logged in.")
        if not session_code or not session_code.strip():
            return JoinSessionResult(error="Please enter a valid session code.")

        upper_session_code = session_code.strip().upper()

        # Log join session attempt
        self._log_event(
            event_type="session",
            participant_id=participant_id,
            message=f"Attempting to join session {upper_session_code}",
            session_id=upper_session_code,
            severity="info"
        )

        data, error = await self.sm_client.join_session(upper_session_code, participant_id)

        if error:
            # Log join failure
            self._log_event(
                event_type="session",
                participant_id=participant_id,
                message=f"Failed to join session: {error}",
                session_id=upper_session_code,
                severity="error"
            )
            return JoinSessionResult(error=f"Failed to join session: {error}")

        # Log join success
        self._log_event(
            event_type="session",
            participant_id=participant_id,
            message=f"Successfully joined session {upper_session_code}",
            session_id=upper_session_code,
            severity="info"
        )

        return JoinSessionResult(
            error=None,
            message=data.get("message"),
            user_context={
                "participant_id": participant_id,
                "session_id": upper_session_code,
            }
        )

    async def handle_send_message(self, user_context: dict, message: str) -> SendMessageResult:
        """
        Handles sending a message from a participant to a session.

        Args:
            user_context (dict): A dictionary containing the current user's state
                                 (participant_id, session_id).
            message (str): The message content to send.

        Returns:
            SendMessageResult: A Pydantic model containing the outcome of the send attempt.
        """
        if not user_context or not user_context.get("session_id"):
            return SendMessageResult(error="Cannot send message, not in a session.")
        if not message or not message.strip():
            return SendMessageResult(error="Cannot send an empty message.")

        # Log message send attempt
        self._log_event(
            event_type="message",
            participant_id=user_context["participant_id"],
            message=f"Sending message (length: {len(message.strip())})",
            session_id=user_context["session_id"],
            severity="info"
        )

        data, error = await self.sm_client.send_message(
            session_id=user_context["session_id"],
            participant_id=user_context["participant_id"],
            content=message.strip()
        )

        if error:
            # Log message send failure
            self._log_event(
                event_type="message",
                participant_id=user_context["participant_id"],
                message=f"Failed to send message: {error}",
                session_id=user_context["session_id"],
                severity="error"
            )
            return SendMessageResult(error=f"Failed to send message: {error}")

        # Log message send success
        self._log_event(
            event_type="message",
            participant_id=user_context["participant_id"],
            message="Message sent successfully",
            session_id=user_context["session_id"],
            severity="info"
        )

        return SendMessageResult(error=None, message=data.get("message"))
    
    async def handle_refresh(self, user_context: dict) -> RefreshResult:
        """
        Handles a UI refresh request by polling the session state from the server.

        This is the main method for updating the UI. It always returns fresh data
        from the server and transforms the raw API data into clean, structured
        Pydantic models for the UI layer to consume.

        Args:
            user_context (dict): The current user's state (participant_id, session_id).

        Returns:
            RefreshResult: A Pydantic model representing the complete, updated state
                           required to render the UI, or an error.
        """
        if not user_context or not user_context.get("session_id"):
            return RefreshResult(error="No active session.", changed=True)

        data, error = await self.sm_client.get_session_state(
            session_id=user_context["session_id"],
            participant_id=user_context["participant_id"]
        )

        if error:
            return RefreshResult(error=f"Failed to refresh state: {error}", changed=True)

        session = data["session"]

        # Determine role display (don't show internal role names to user)
        participant_role = data.get("participant_role", "observer")
        role_display = "First Speaker" if participant_role == "first_speaker" else "Second Speaker" if participant_role == "second_speaker" else "Observer"

        return RefreshResult(
            changed=True,
            error=None,
            status_text=self._determine_status_message(session, data["can_participant_speak"]),
            topic_and_stance=f"### Topic: {session.get('parameters', {}).get('debate_topic', 'N/A')}\n**Your Role**: {role_display}",
            session_info=session,
            chat_items=self._get_chat_data(session.get("dialogue_history", []), user_context["participant_id"]),
            typical_replies=data.get("typical_replies", []),
            first_speaker_commitments=data.get("first_speaker_commitments", []),
            second_speaker_commitments=data.get("second_speaker_commitments", [])
        )
    
    async def handle_admin_refresh(self, participant_id: str) -> AdminRefreshResult:
        """
        Handles a refresh request for the admin dashboard.

        Args:
            participant_id (str): The ID of the participant requesting the refresh.
                                  Must be 'ADMIN'.

        Returns:
            AdminRefreshResult: A Pydantic model containing the dashboard data or an error.
        """
        if participant_id != 'ADMIN':
            return AdminRefreshResult(error="Not authorized.")

        data, error = await self.sm_client.admin_dashboard(participant_id)

        if error:
            return AdminRefreshResult(error=f"Failed to load admin dashboard: {error}")

        # Add client logs to dashboard data
        try:
            client_logs = self.logs.load_log()[-50:]  # Last 50 logs
            data['client_logs'] = [
                {
                    'timestamp': log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'severity': log.severity,
                    'event_type': log.event_type,
                    'participant_id': log.participant_id,
                    'message': log.message,
                    'session_id': log.session_id
                }
                for log in client_logs
            ]
        except Exception as e:
            data['client_logs'] = []
            self._log_event(
                event_type="system",
                participant_id="SYSTEM",
                message=f"Failed to load client logs: {str(e)}",
                severity="error"
            )

        return AdminRefreshResult(error=None, dashboard_data=data)

    async def handle_end_session(self, user_context: dict, message: str = "User ended session") -> EndSessionResult:
        """
        Handles ending a session.

        Args:
            user_context (dict): The current user's state (participant_id, session_id).
            message (str): Reason for ending the session. Defaults to "User ended session".

        Returns:
            EndSessionResult: A Pydantic model containing the outcome.
        """
        if not user_context or not user_context.get("session_id"):
            return EndSessionResult(error="Cannot end session, not in a session.")

        # Log session end attempt
        self._log_event(
            event_type="session",
            participant_id=user_context["participant_id"],
            message=f"Attempting to end session: {message}",
            session_id=user_context["session_id"],
            severity="info"
        )

        data, error = await self.sm_client.end_session(
            session_id=user_context["session_id"],
            participant_id=user_context["participant_id"],
            message=message
        )

        if error:
            # Log session end failure
            self._log_event(
                event_type="session",
                participant_id=user_context["participant_id"],
                message=f"Failed to end session: {error}",
                session_id=user_context["session_id"],
                severity="error"
            )
            # Convert technical errors to friendly messages
            friendly_error = ERROR_MESSAGES.get(error, error)
            return EndSessionResult(error=f"Failed to end session: {friendly_error}")

        # Log session end success
        self._log_event(
            event_type="session",
            participant_id=user_context["participant_id"],
            message="Session ended successfully",
            session_id=user_context["session_id"],
            severity="info"
        )

        return EndSessionResult(error=None, message=data.get("message", "Session ended successfully"))

    # --- ABSTRACT METHODS FOR UI IMPLEMENTATION ---
    
    @abstractmethod
    def create_interface(self):
        """
        Each subclass must implement this to build its specific UI layout
        (e.g., using Gradio, Flet, etc.).
        """
        pass
    
    @abstractmethod
    def launch(self, **kwargs):
        """
        Each subclass must implement this to run its specific application
        event loop and server.
        """
        pass