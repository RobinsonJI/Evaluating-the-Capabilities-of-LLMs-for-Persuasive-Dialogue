"""
SessionManagerClient - HTTP client for Session Manager API
"""

import httpx
from typing import Optional, Dict, Any, Tuple

class SessionManagerClient:
    """
    Async HTTP client for Session Manager API.

    All methods return (data, error) tuples:
    - Success: (response_dict, None)
    - Failure: (None, error_message)
    """

    def __init__(self, api_key: str, base_url: str):
        """
        Initialize the Session Manager client.

        Args:
            api_key: API key for X-API-Key header authentication
            base_url: Base URL of the Session Manager API 
        """
        self.api_key = api_key
        self.base_url = base_url
        # Set X-API-Key header for all requests
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-API-Key": self.api_key},
            timeout=30.0
        )

    async def close(self):
        """Close the HTTP client connection."""
        await self.client.aclose()

    # ============== CORE API CALLS ==============

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Make an HTTP request to the Session Manager API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., "/health")
            **kwargs: Additional arguments for httpx request (json, params, etc.)

        Returns:
            Tuple of (response_data, error_message):
            - response_data: Parsed JSON response if successful
            - error_message: Error description if failed
        """
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json(), None
        except httpx.HTTPStatusError as e:
            # HTTP error (4xx, 5xx)
            # Try to parse JSON error response from API
            try:
                error_data = e.response.json()
                error_msg = error_data.get("detail", str(error_data))
            except:
                # If not JSON, check if it's HTML (Azure error page)
                response_text = e.response.text
                if response_text.startswith("<!DOCTYPE") or response_text.startswith("<html"):
                    # It's HTML - provide user-friendly message based on status code
                    if e.response.status_code == 404:
                        error_msg = "Service unavailable (404). Please check if the session manager is running."
                    elif e.response.status_code == 401:
                        error_msg = "Invalid login code"
                    elif e.response.status_code == 403:
                        error_msg = "Access denied"
                    elif e.response.status_code == 500:
                        error_msg = "Server error. Please try again later."
                    else:
                        error_msg = f"HTTP {e.response.status_code} error"
                else:
                    # Plain text error
                    error_msg = f"HTTP {e.response.status_code}: {response_text[:200]}"  # Limit to 200 chars
            return None, error_msg
        except httpx.RequestError as e:
            # Connection error
            error_msg = f"Connection error: {str(e)}"
            return None, error_msg
        except Exception as e:
            # Unexpected error
            error_msg = f"Unexpected error: {str(e)}"
            return None, error_msg

    # ============== HEALTH CHECK ==============

    async def health_check(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Check if Session Manager API is healthy.

        Endpoint: GET /health

        Returns:
            Tuple of (data, error) where data is:
            {
                "status": "healthy",
                "service": "session-manager"
            }

            NOTE: This method returns a tuple instead of a Result object.
            Health checks are simple queries without status variants.
        """
        return await self._make_request("GET", "/health")

    # ============== AUTHENTICATION ==============

    async def login(self, code: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Authenticate participant with login code.

        Endpoint: POST /auth/login
        Request: LoginRequest {code: str}
        Response: LoginResponse {participant_id: str, participant_type: str, message: str}

        Args:
            code: 6-digit authentication code

        Returns:
            Tuple of (data, error):
            - data: LoginResponse dict if successful
            - error: Error message if failed

        Example:
            data, error = await client.login("123456")
            if data:
                participant_id = data["participant_id"]
                print(f"Logged in as {participant_id}")
            else:
                print(f"Login failed: {error}")
        """
        return await self._make_request(
            "POST",
            "/auth/login",
            json={"code": code}
        )

    # ============== SESSION MANAGEMENT ==============

    async def join_session(
        self,
        session_id: str,
        participant_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Join a session with authenticated participant.

        Endpoint: POST /sessions/{session_id}/join
        Request: JoinSessionRequest {participant_id: str}
        Response: JoinSessionResponse {message: str, session: dict}

        Args:
            session_id: Session code to join (e.g., "SESS1")
            participant_id: Authenticated participant ID

        Returns:
            Tuple of (data, error):
            - data: JoinSessionResponse dict with session data if successful
            - error: Error message if failed

        Example:
            data, error = await client.join_session("SESS1", "HUMAN1")
            if data:
                session = data["session"]
                print(f"Joined session: {session['topic']}")
            else:
                print(f"Join failed: {error}")
        """
        return await self._make_request(
            "POST",
            f"/sessions/{session_id}/join",
            json={"participant_id": participant_id}
        )

    async def get_session_state(
        self,
        session_id: str,
        participant_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Get current session state.

        Endpoint: GET /sessions/{session_id}/state
        Query Params: participant_id (required)
        Response: SessionStateResponse {changed: bool, session: dict, can_participant_speak: bool, participant_stance: str}

        Args:
            session_id: Session identifier
            participant_id: Participant requesting state

        Returns:
            Tuple of (data, error) where data is SessionStateResponse dict

            NOTE: This method returns a tuple instead of a Result object because
            SessionStateResponse is a simple query response without status variants.
            Consider creating a SessionStateResult type if needed.

        Status Codes:
            200: Success
            404: Session or participant not found
        """
        params = {"participant_id": participant_id}

        return await self._make_request(
            "GET",
            f"/sessions/{session_id}/state",
            params=params
        )

    async def send_message(
        self,
        session_id: str,
        participant_id: str,
        content: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Send a message in the session.

        Endpoint: POST /sessions/{session_id}/messages
        Request: SendMessageRequest {participant_id: str, content: str}
        Response: SendMessageResponse {message: str, turn_number: int | None}

        Args:
            session_id: Session identifier
            participant_id: Participant sending message
            content: Message text content

        Returns:
            Tuple of (data, error):
            - data: SendMessageResponse dict if successful
            - error: Error message if failed

        Note:
            This returns an acknowledgment. To get updated dialogue history,
            commitments, and typical replies, call get_session_state() after.

        Example:
            data, error = await client.send_message("SESS1", "HUMAN1", "I agree...")
            if data:
                print(data["message"])  # "Message sent!" or "Message sent! Debate ended: ..."
                # Then refresh UI by calling get_session_state()
            else:
                print(f"Error: {error}")
        """
        return await self._make_request(
            "POST",
            f"/sessions/{session_id}/messages",
            json={"participant_id": participant_id, "content": content}
        )

    async def end_session(
        self,
        session_id: str,
        participant_id: str,
        message: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        End a session (user-initiated).

        Endpoint: POST /sessions/{session_id}/end
        Query Params: participant_id, message
        Response: {message: str, session_id: str}

        Args:
            session_id: Session identifier
            participant_id: Participant ending the session
            message: Reason/message for ending

        Returns:
            Tuple of (data, error):
            - data: Response dict with confirmation if successful
            - error: Error message if failed

        Example:
            data, error = await client.end_session("SESS1", "HUMAN1", "User requested end")
            if data:
                print(f"Session {data['session_id']} ended: {data['message']}")
            else:
                print(f"Error: {error}")
        """
        return await self._make_request(
            "POST",
            f"/sessions/{session_id}/end",
            params={"participant_id": participant_id, "message": message}
        )

    # ============== ADMIN ENDPOINTS ==============

    async def admin_dashboard(
        self,
        participant_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Get consolidated admin dashboard data.

        Endpoint: GET /admin/dashboard
        Query Params: participant_id (must be admin)

        Args:
            participant_id: Admin participant ID

        Returns:
            Tuple of (data, error) where data contains:
            {
                "session_counts": {active: int, waiting: int, completed: int, total: int},
                "participant_counts": {human: int, ai: int, total_registered: int},
                "sessions": [session_data...],
                "participants": [participant_data...],
                "recent_activities": [activity_data...],
                "timestamp": str
            }

            NOTE: Admin endpoints return tuples instead of Result objects.
            These are query endpoints without complex status variants.

        Status Codes:
            200: Success
            403: Not admin
        """
        return await self._make_request(
            "GET",
            "/admin/dashboard",
            params={"participant_id": participant_id}
        )

    async def admin_health(
        self,
        participant_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Extended health check with system metrics.

        Endpoint: GET /admin/health
        Query Params: participant_id (must be admin)

        Args:
            participant_id: Admin participant ID

        Returns:
            Tuple of (data, error) where data contains:
            {
                "status": "healthy",
                "service": "session-manager",
                "admin_access": bool,
                "system_stats": {...},
                "memory_usage": {...}
            }

            NOTE: Admin endpoints return tuples instead of Result objects.

        Status Codes:
            200: Success
            403: Not admin
        """
        return await self._make_request(
            "GET",
            "/admin/health",
            params={"participant_id": participant_id}
        )

    async def admin_persuasio_health(
        self,
        participant_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Check Persuasio backend connection and health.

        Endpoint: GET /admin/persuasio-health
        Query Params: participant_id (must be admin)

        Args:
            participant_id: Admin participant ID

        Returns:
            Tuple of (data, error) where data contains:
            {
                "persuasio_status": "healthy" | "unhealthy",
                "persuasio_url": str,
                "sessions": {...} (if healthy),
                "error": str (if unhealthy),
                "error_type": str (if unhealthy)
            }

            NOTE: Admin endpoints return tuples instead of Result objects.

        Status Codes:
            200: Success (returns status in response body)
            403: Not admin
        """
        return await self._make_request(
            "GET",
            "/admin/persuasio-health",
            params={"participant_id": participant_id}
        )
