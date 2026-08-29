from pydantic import BaseModel
from typing import Optional
import httpx

from session_manager.models import (
    Session,
    ClientResponse,
    PersuasioResponse,
    SpeakerOrder,
    SpeakerType
)

class PersuasioClientError(Exception):
    """Custom exception for Persuasio API errors with detailed messages."""
    def __init__(self, status_code: int, detail: str, url: str):
        self.status_code = status_code
        self.detail = detail
        self.url = url
        super().__init__(f"{status_code} Error for {url}: {detail}")


class PersuasioClient:
    """
    Client for interacting with Persuasio API

    Attributes:
        base_url (str): Base URL for the Persuasio API
        api_key (str): API key for authentication
        client (httpx.AsyncClient): HTTP client for making requests
    Methods:
        create_session(response: ClientResponse, params: SessionParameters) -> PersuasioResponse:
            Create a new session using the provided response and parameters
        update_session(session_id: str, response: ClientResponse) -> PersuasioResponse:
            Update an existing session with a new response
        get_sessions() -> dict:
            Retrieve lists of finished and ongoing sessions
        get_session_state(session_id: str) -> dict:
            Retrieve the current state of a specific session
        get_session_log(session_id: str) -> dict:
            Retrieve the activity log of a specific session
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-API-Key": self.api_key},
            timeout=httpx.Timeout(300.0)  # 5 minutes for long-running LLM operations
        )

    def _raise_for_status(self, response: httpx.Response):
        """
        Check response status and raise PersuasioClientError with detailed message if error.

        Unlike httpx.raise_for_status(), this reads the response body to extract
        the FastAPI HTTPException detail field.
        """
        if response.status_code >= 400:
            # Try to extract detail from JSON response
            try:
                error_body = response.json()
                detail = error_body.get("detail", response.text)
            except Exception:
                detail = response.text or f"HTTP {response.status_code}"

            raise PersuasioClientError(
                status_code=response.status_code,
                detail=detail,
                url=str(response.url)
            )
        
    async def create_session(
        self,
        session: Session,
        first_utterance: Optional[str] = None
    ) -> PersuasioResponse:
        """
        Create a new session in persuasio backend.

        Args:
            session: Session object with parameters
            first_utterance: First message if human starts, None/empty if AI starts

        Returns:
            PersuasioResponse from persuasio API
        """
        # Determine utterance_from based on who starts
        # For now, always use FIRST_SPEAKER as utterance_from
        utterance_from = SpeakerOrder.FIRST_SPEAKER

        # Create ClientResponse
        client_response = ClientResponse(
            utterance=first_utterance or "",
            utterance_from=utterance_from,
            next_speaker=SpeakerOrder.SECOND_SPEAKER
        )

        # Combine session parameters and response for API call
        params = {
            **session.parameters.model_dump(mode='json'),
            **client_response.model_dump(mode='json')
        }

        response = await self.client.get("/sessions/create", params=params)
        self._raise_for_status(response)

        return PersuasioResponse(**response.json())
        
            
    async def update_session(
        self,
        session: Session,
        speaker_id: str,
        utterance: str
    ) -> PersuasioResponse:
        """
        Update an ongoing session with a new utterance.

        Args:
            session: Session object
            speaker_id: ID of the speaker sending the message
            utterance: The message content

        Returns:
            PersuasioResponse from persuasio API
        """
        # Determine which speaker this is
        utterance_from = (SpeakerOrder.FIRST_SPEAKER
                         if speaker_id == session.participant1_id
                         else SpeakerOrder.SECOND_SPEAKER)

        next_speaker = (SpeakerOrder.SECOND_SPEAKER
                       if utterance_from == SpeakerOrder.FIRST_SPEAKER
                       else SpeakerOrder.FIRST_SPEAKER)

        # Create ClientResponse
        client_response = ClientResponse(
            utterance=utterance,
            utterance_from=utterance_from,
            next_speaker=next_speaker
        )

        response = await self.client.get(
            f"/sessions/{session.session_id}/update",
            params=client_response.model_dump(mode='json')
        )
        self._raise_for_status(response)

        return PersuasioResponse(**response.json())
    
    async def end_session(self, session_id: str):
        """
        Terminate a session in the persuasio backend.

        Args:
            session_id: ID of the session to terminate.
        """
        
        end = await self.client.post(f"/sessions/terminate/{session_id}")
        
        self._raise_for_status(end)
        
        return 
            
    async def get_sessions(self) -> dict:
        """
        Retrieve and combine results of 
        GET sessions/finished 
        GET sessions/ongoing
        """   
        finished = await self.client.get("/sessions/all_ended")
        ongoing = await self.client.get("/sessions/ongoing")
        

        self._raise_for_status(finished)
        self._raise_for_status(ongoing)
        
        return {
            "finished" : finished.json(),
            "ongoing" : ongoing.json()
        }
    
    async def get_session_state(self, session_id: str) -> dict:
        """
        Call GET /sessions/{session_id}/view_state
        """
        response = await self.client.get(f"/sessions/{session_id}/view_state")
        self._raise_for_status(response)
        return response.json()
    
    async def get_session_log(self, session_id: str, limit: Optional[int] = 100) -> dict:
        """
        Call GET /sessions/{session_id}/log

        Args:
            session_id: ID of the session
            limit: Maximum number of log entries to return (most recent). Default 100.
        """
        params = {"limit": limit} if limit else {}
        response = await self.client.get(f"/sessions/{session_id}/log", params=params)
        self._raise_for_status(response)
        return response.json()
    
    async def restore_session(
        self,
        session: Session
    ):
        """ 
        Restore a session in the persuasio backend using the existing dialogue history.
        
        This should be called when the session_manager has a session but it's missing
        from the persuasio backend (e.g., after a persuasio restart).

        Args:
            session: The Session object from the session_manager, containing the
                     dialogue history and parameters needed for restoration.

        Returns:
            A dictionary containing the `new_session_id` for the restored thread
            on the persuasio backend.
        """
        # TODO: Implement this method.
        # 1. This method should make a POST request to a new `/sessions/restore` endpoint
        #    on the persuasio backend.
        # 2. The payload should contain the session.dialogue_history and session.parameters.
        # 3. The backend should use this data to rebuild its state and return the new thread ID.
        # 4. The client should return this new ID, e.g., `return {"new_session_id": "xyz"}`.
        raise NotImplementedError("The restore_session functionality is not implemented.")
    
    async def check_session(self, session_id: str) -> str:
        """ Check if a session exists in Persuasio """
        sessions = await self.get_sessions()
        if session_id in sessions['finished']:
            return "finished"
        elif session_id in sessions['ongoing']:
            return "ongoing"
        else:
            return False

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()