"""
Prolific API Client
A simple httpx-based client for interacting with the Prolific API.
"""

import os
from typing import Any, Dict, List, Optional, Union
import httpx


class ProlificAPIError(Exception):
    """Exception raised for Prolific API errors."""
    def __init__(self, status_code: int, message: str, response: Optional[Dict] = None):
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"Prolific API Error {status_code}: {message}")


class ProlificClient:
    """
    Client for interacting with the Prolific API.

    Usage:
        client = ProlificClient(api_token="your_token")
        studies = client.list_studies()
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        base_url: str = "https://api.prolific.com/api/v1"
    ):
        """
        Initialize the Prolific API client.

        Args:
            api_token: Your Prolific API token. If not provided, will try to read
                      from PROLIFIC_API_TOKEN environment variable.
            base_url: Base URL for the API (default: production endpoint)
        """
        self.api_token = api_token or os.environ.get("PROLIFIC_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "API token is required. Provide it directly or set PROLIFIC_API_TOKEN "
                "environment variable."
            )

        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            headers={
                "Authorization": f"Token {self.api_token}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[Dict, List]:
        """
        Make a request to the Prolific API.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint (will be appended to base_url)
            params: Query parameters
            json: JSON body
            **kwargs: Additional arguments to pass to httpx

        Returns:
            Response JSON data

        Raises:
            ProlificAPIError: If the API returns an error
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.client.request(
                method=method,
                url=url,
                params=params,
                json=json,
                **kwargs
            )
            response.raise_for_status()

            # Handle empty responses
            if response.status_code == 204 or not response.content:
                return {}

            return response.json()

        except httpx.HTTPStatusError as e:
            error_msg = str(e)
            error_data = None

            try:
                error_data = e.response.json()
                if isinstance(error_data, dict):
                    error_msg = error_data.get("error", error_data.get("detail", str(e)))
            except Exception:
                pass

            raise ProlificAPIError(
                status_code=e.response.status_code,
                message=error_msg,
                response=error_data
            ) from e

    # ==================== User Info ====================

    def get_user_info(self) -> Dict:
        """Get information about the authenticated user."""
        return self._request("GET", "/users/me/")

    def create_test_participant(self, email: str) -> Dict:
        """
        Create a test participant for testing studies.

        Args:
            email: Email address for the test participant

        Returns:
            Test participant details including participant ID
        """
        return self._request("POST", "/researchers/participants/", json={"email": email})

    # ==================== Studies ====================

    def list_studies(self, state: Optional[str] = None) -> Dict:
        """
        List all studies.

        Args:
            state: Filter by study state (ACTIVE, PAUSED, UNPUBLISHED, PUBLISHING,
                  COMPLETED, AWAITING REVIEW, UNKNOWN, SCHEDULED)

        Returns:
            Dictionary with 'results' containing list of studies
        """
        params = {}
        if state:
            params["state"] = state
        return self._request("GET", "/studies/", params=params)

    def get_study(self, study_id: str) -> Dict:
        """
        Get details of a specific study.

        Args:
            study_id: The study ID

        Returns:
            Study details
        """
        return self._request("GET", f"/studies/{study_id}/")

    def create_study(
        self,
        name: str,
        internal_name: str,
        description: str,
        external_study_url: str,
        total_available_places: int,
        estimated_completion_time: int,
        reward: int,
        prolific_id_option: str = "url_parameters",
        completion_code: Optional[str] = None,
        completion_codes: Optional[List[Dict]] = None,
        device_compatibility: Optional[List[str]] = None,
        peripheral_requirements: Optional[List[str]] = None,
        eligibility_requirements: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict:
        """
        Create a draft study.

        Args:
            name: Public name of the study
            internal_name: Internal name (only visible to you)
            description: Study description shown to participants
            external_study_url: URL where participants will be redirected
            total_available_places: Number of participants needed
            estimated_completion_time: Estimated time in minutes
            reward: Reward in pence/cents (e.g., 100 = £1.00 or $1.00)
            prolific_id_option: How to pass participant ID (default: "url_parameters")
            completion_code: Simple completion code (or use completion_codes for advanced)
            completion_codes: Advanced completion codes with actions
            device_compatibility: List of compatible devices (e.g., ["desktop", "mobile"])
            peripheral_requirements: List of required peripherals
            eligibility_requirements: List of eligibility requirement dicts
            **kwargs: Additional study parameters

        Returns:
            Created study details
        """
        data = {
            "name": name,
            "internal_name": internal_name,
            "description": description,
            "external_study_url": external_study_url,
            "total_available_places": total_available_places,
            "estimated_completion_time": estimated_completion_time,
            "reward": reward,
            "prolific_id_option": prolific_id_option,
        }

        if completion_code:
            data["completion_code"] = completion_code
        if completion_codes:
            data["completion_codes"] = completion_codes
        if device_compatibility:
            data["device_compatibility"] = device_compatibility
        if peripheral_requirements:
            data["peripheral_requirements"] = peripheral_requirements
        if eligibility_requirements:
            data["eligibility_requirements"] = eligibility_requirements

        # Add any additional parameters
        data.update(kwargs)

        return self._request("POST", "/studies/", json=data)

    def update_study(self, study_id: str, **kwargs) -> Dict:
        """
        Update a study.

        Args:
            study_id: The study ID
            **kwargs: Fields to update

        Returns:
            Updated study details
        """
        return self._request("PATCH", f"/studies/{study_id}/", json=kwargs)

    def publish_study(self, study_id: str) -> Dict:
        """
        Publish a draft study to make it live.

        Args:
            study_id: The study ID

        Returns:
            Updated study details
        """
        return self._request(
            "POST",
            f"/studies/{study_id}/transition/",
            json={"action": "PUBLISH"}
        )

    def pause_study(self, study_id: str) -> Dict:
        """
        Pause an active study.

        Args:
            study_id: The study ID

        Returns:
            Updated study details
        """
        return self._request(
            "POST",
            f"/studies/{study_id}/transition/",
            json={"action": "PAUSE"}
        )

    def start_study(self, study_id: str) -> Dict:
        """
        Start/resume a paused study.

        Args:
            study_id: The study ID

        Returns:
            Updated study details
        """
        return self._request(
            "POST",
            f"/studies/{study_id}/transition/",
            json={"action": "START"}
        )

    def stop_study(self, study_id: str) -> Dict:
        """
        Stop a study completely.

        Args:
            study_id: The study ID

        Returns:
            Updated study details
        """
        return self._request(
            "POST",
            f"/studies/{study_id}/transition/",
            json={"action": "STOP"}
        )

    def get_study_cost(self, study_id: str) -> Dict:
        """
        Get the cost breakdown for a study.

        Args:
            study_id: The study ID

        Returns:
            Cost information
        """
        return self._request("GET", f"/studies/{study_id}/cost/")

    def clone_study(self, study_id: str) -> Dict:
        """
        Clone an existing study.

        Args:
            study_id: The study ID to clone

        Returns:
            New cloned study details
        """
        return self._request("POST", f"/studies/{study_id}/clone/")

    def get_study_submissions(self, study_id: str) -> Dict:
        """
        List all submissions for a study.

        Args:
            study_id: The study ID

        Returns:
            Dictionary with 'results' containing list of submissions
        """
        return self._request("GET", f"/studies/{study_id}/submissions/")

    def get_study_submissions_count(self, study_id: str) -> Dict:
        """
        Get count of submissions by status for a study.

        Args:
            study_id: The study ID

        Returns:
            Dictionary with counts per submission status
        """
        return self._request("GET", f"/studies/{study_id}/submissions/counts/")

    def export_study_demographics(
        self,
        study_id: str,
        filters: Optional[List[Dict]] = None
    ) -> bytes:
        """
        Export demographic data for a study (new method).

        Args:
            study_id: The study ID
            filters: Optional list of demographic filters to apply

        Returns:
            CSV data as bytes
        """
        data = {}
        if filters:
            data["filters"] = filters

        response = self.client.post(
            f"{self.base_url}/studies/{study_id}/demographic-export/",
            json=data if data else None
        )
        response.raise_for_status()
        return response.content

    def get_demographic_export_history(self, study_id: str) -> Dict:
        """
        Get history of demographic exports for a study.

        Args:
            study_id: The study ID

        Returns:
            List of previous exports
        """
        return self._request("GET", f"/studies/{study_id}/demographic-export-history/")

    def create_test_study(self, study_id: str) -> Dict:
        """
        Create a test study from a draft study for test participants.

        Args:
            study_id: The draft study ID

        Returns:
            Test study details
        """
        return self._request("POST", f"/studies/{study_id}/test-study")

    def calculate_study_cost(
        self,
        total_available_places: int,
        reward: int,
        estimated_completion_time: int,
        eligibility_requirements: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict:
        """
        Calculate the cost of a study before creating it.

        Args:
            total_available_places: Number of participants
            reward: Reward in pence/cents
            estimated_completion_time: Time in minutes
            eligibility_requirements: Optional list of eligibility requirements
            **kwargs: Additional parameters

        Returns:
            Cost calculation details
        """
        data = {
            "total_available_places": total_available_places,
            "reward": reward,
            "estimated_completion_time": estimated_completion_time
        }
        if eligibility_requirements:
            data["eligibility_requirements"] = eligibility_requirements
        data.update(kwargs)

        return self._request("POST", "/study-cost-calculator/", json=data)

    def get_predicted_recruitment_time(
        self,
        study_id: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Get predicted recruitment time for a study.

        Args:
            study_id: Optional study ID (if None, uses kwargs to calculate)
            **kwargs: Study parameters if study_id not provided

        Returns:
            Predicted recruitment time
        """
        if study_id:
            return self._request("GET", f"/studies/{study_id}/predicted-recruitment-time/")
        else:
            return self._request("POST", "/studies/predicted-recruitment-time/", json=kwargs)

    # ==================== Submissions ====================

    def list_submissions(
        self,
        study: Optional[str] = None,
        status: Optional[str] = None,
        completed: Optional[bool] = None,
        returned: Optional[bool] = None,
        awaiting_review: Optional[bool] = None,
        approved: Optional[bool] = None,
        active: Optional[bool] = None,
        timeout: Optional[bool] = None,
        rejected: Optional[bool] = None,
        screened_out: Optional[bool] = None,
        completeoractive: Optional[bool] = None,
        submitted_from: Optional[str] = None,
        submitted_before: Optional[str] = None,
        ordering: Optional[str] = None,
        page_size: Optional[int] = None,
        page: Optional[int] = None
    ) -> Dict:
        """
        List all submissions across all studies with extensive filtering options.

        Args:
            study: Filter by study ID
            status: Filter by status (ACTIVE, AWAITING_REVIEW, APPROVED, REJECTED, etc.)
            completed: Filter by completion status (True/False)
            returned: Filter by returned status (True/False)
            awaiting_review: Filter by awaiting review status (True/False)
            approved: Filter by approved status (True/False)
            active: Filter by active status (True/False)
            timeout: Filter by timed out status (True/False)
            rejected: Filter by rejected status (True/False)
            screened_out: Filter by screened out status (True/False)
            completeoractive: Filter for completed or active submissions (True/False)
            submitted_from: Filter submissions after date (format: DD/MM/YYYY)
            submitted_before: Filter submissions before date (format: DD/MM/YYYY)
            ordering: Order results (started_at, -started_at, submission_reward,
                     -submission_reward, study_name, -study_name)
            page_size: Number of items per page (default: 20)
            page: Page number (default: 1)

        Returns:
            Dictionary with 'results' containing list of submissions
        """
        params = {}
        if study:
            params["study"] = study
        if status:
            params["status"] = status
        if completed is not None:
            params["completed"] = "1" if completed else "0"
        if returned is not None:
            params["returned"] = "1" if returned else "0"
        if awaiting_review is not None:
            params["awaiting_review"] = "1" if awaiting_review else "0"
        if approved is not None:
            params["approved"] = "1" if approved else "0"
        if active is not None:
            params["active"] = "1" if active else "0"
        if timeout is not None:
            params["timeout"] = "1" if timeout else "0"
        if rejected is not None:
            params["rejected"] = "1" if rejected else "0"
        if screened_out is not None:
            params["screened_out"] = "1" if screened_out else "0"
        if completeoractive is not None:
            params["completeoractive"] = "1" if completeoractive else "0"
        if submitted_from:
            params["submitted_from"] = submitted_from
        if submitted_before:
            params["submitted_before"] = submitted_before
        if ordering:
            params["ordering"] = ordering
        if page_size:
            params["page_size"] = page_size
        if page:
            params["page"] = page

        return self._request("GET", "/submissions/", params=params)

    def get_submission(self, submission_id: str) -> Dict:
        """
        Get details of a specific submission.

        Args:
            submission_id: The submission ID

        Returns:
            Submission details
        """
        return self._request("GET", f"/submissions/{submission_id}/")

    def approve_submission(self, submission_id: str) -> Dict:
        """
        Approve a submission.

        Args:
            submission_id: The submission ID

        Returns:
            Updated submission details
        """
        return self._request(
            "POST",
            f"/submissions/{submission_id}/transition/",
            json={"action": "APPROVE"}
        )

    def reject_submission(
        self,
        submission_id: str,
        message: Optional[str] = None,
        rejection_category: Optional[str] = None
    ) -> Dict:
        """
        Reject a submission.

        Args:
            submission_id: The submission ID
            message: Optional rejection message (must be >100 chars)
            rejection_category: Optional rejection category (e.g., FAILED_INSTRUCTIONS)

        Returns:
            Updated submission details
        """
        data = {"action": "REJECT"}
        if message:
            data["message"] = message
        if rejection_category:
            data["rejection_category"] = rejection_category

        return self._request(
            "POST",
            f"/submissions/{submission_id}/transition/",
            json=data
        )

    def bulk_approve_submissions(
        self,
        study_id: Optional[str] = None,
        participant_ids: Optional[List[str]] = None,
        submission_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Bulk approve submissions. Two options:
        1. Provide study_id + participant_ids
        2. Provide submission_ids (recommended - can be from different studies)

        Args:
            study_id: The study ID (required if using participant_ids)
            participant_ids: List of participant IDs (requires study_id)
            submission_ids: List of submission IDs (recommended method)

        Returns:
            Result of bulk approval

        Note:
            Either (study_id + participant_ids) OR submission_ids must be provided.
        """
        data = {}
        if submission_ids:
            data["submission_ids"] = submission_ids
        elif study_id and participant_ids:
            data["study_id"] = study_id
            data["participant_ids"] = participant_ids
        elif study_id:
            # Backwards compatibility: approve all for study
            data["study_id"] = study_id

        return self._request("POST", "/submissions/bulk-approve/", json=data)

    def create_bonus_payment(
        self,
        study_id: str,
        csv_bonuses: str
    ) -> Dict:
        """
        Create bonus payments for participants.

        Args:
            study_id: The study ID
            csv_bonuses: CSV string with format "participant_id,amount_in_cents"
                        Example: "5f1234567890abcdef123456,150"

        Returns:
            Bonus payment batch details
        """
        return self._request(
            "POST",
            "/submissions/bonus-payments/",
            json={
                "study_id": study_id,
                "csv_bonuses": csv_bonuses
            }
        )

    def pay_bonus_batch(self, batch_id: str) -> Dict:
        """
        Execute a bonus payment batch.

        Args:
            batch_id: The bonus batch ID

        Returns:
            Payment result
        """
        return self._request("POST", f"/bulk-bonus-payments/{batch_id}/pay/")

    def request_submission_return(
        self,
        submission_id: str,
        return_reasons: List[str]
    ) -> Dict:
        """
        Request a participant to return their submission.

        Args:
            submission_id: The submission ID
            return_reasons: List of reasons (e.g., ["Didn't finish the study",
                          "Encountered technical problems", "Withdrew consent"])

        Returns:
            Return request confirmation
        """
        return self._request(
            "POST",
            f"/submissions/{submission_id}/request-return/",
            json={"request_return_reasons": return_reasons}
        )

    # ==================== Workspaces ====================

    def list_workspaces(self) -> Dict:
        """
        List all workspaces the user has access to.

        Returns:
            Dictionary with 'results' containing list of workspaces
        """
        return self._request("GET", "/workspaces/")

    def get_workspace(self, workspace_id: str) -> Dict:
        """
        Get details of a specific workspace.

        Args:
            workspace_id: The workspace ID

        Returns:
            Workspace details
        """
        return self._request("GET", f"/workspaces/{workspace_id}/")

    def get_workspace_balance(self, workspace_id: str) -> Dict:
        """
        Get the balance of a workspace.

        Args:
            workspace_id: The workspace ID

        Returns:
            Balance information
        """
        return self._request("GET", f"/workspaces/{workspace_id}/balance/")

    def list_projects(self, workspace_id: str) -> Dict:
        """
        List all projects in a workspace.

        Args:
            workspace_id: The workspace ID

        Returns:
            Dictionary with 'results' containing list of projects
        """
        return self._request("GET", f"/workspaces/{workspace_id}/projects/")

    # ==================== Projects ====================

    def get_project(self, project_id: str) -> Dict:
        """
        Get details of a specific project.

        Args:
            project_id: The project ID

        Returns:
            Project details
        """
        return self._request("GET", f"/projects/{project_id}/")

    def create_project(
        self,
        workspace_id: str,
        title: str,
        description: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Create a new project in a workspace.

        Args:
            workspace_id: The workspace ID
            title: Project title
            description: Project description
            **kwargs: Additional project parameters

        Returns:
            Created project details
        """
        data = {"title": title}
        if description:
            data["description"] = description
        data.update(kwargs)

        return self._request(
            "POST",
            f"/workspaces/{workspace_id}/projects/",
            json=data
        )

    def update_project(self, project_id: str, **kwargs) -> Dict:
        """
        Update a project.

        Args:
            project_id: The project ID
            **kwargs: Fields to update

        Returns:
            Updated project details
        """
        return self._request("PATCH", f"/projects/{project_id}/", json=kwargs)

    def delete_project(self, project_id: str) -> Dict:
        """
        Delete a project.

        Args:
            project_id: The project ID

        Returns:
            Empty response on success
        """
        return self._request("DELETE", f"/projects/{project_id}/")

    def list_project_studies(self, project_id: str) -> Dict:
        """
        List all studies in a project.

        Args:
            project_id: The project ID

        Returns:
            Dictionary with 'results' containing list of studies
        """
        return self._request("GET", f"/projects/{project_id}/studies/")

    # ==================== Participant Groups ====================

    def list_participant_groups(
        self,
        workspace_id: Optional[str] = None,
        project_id: Optional[str] = None,
        active: Optional[bool] = None
    ) -> Dict:
        """
        List all participant groups within a project or workspace.

        Args:
            workspace_id: Filter by workspace ID (recommended)
            project_id: Filter by project ID (deprecated, use workspace_id)
            active: Filter by active (not deleted) status

        Returns:
            Dictionary with 'results' containing list of participant groups

        Note:
            Either workspace_id or project_id is required.
        """
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        elif project_id:
            params["project_id"] = project_id

        if active is not None:
            params["active"] = "true" if active else "false"

        return self._request("GET", "/participant-groups/", params=params)

    def get_participant_group(self, group_id: str) -> Dict:
        """
        Get details of a specific participant group.

        Args:
            group_id: The participant group ID

        Returns:
            Participant group details
        """
        return self._request("GET", f"/participant-groups/{group_id}/")

    def create_participant_group(
        self,
        name: str,
        workspace_id: Optional[str] = None,
        organisation_id: Optional[str] = None,
        description: Optional[str] = None,
        participant_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Create a new participant group within a workspace or organisation.

        Args:
            name: Group name
            workspace_id: The workspace ID (either workspace_id or organisation_id required)
            organisation_id: The organisation ID (either workspace_id or organisation_id required)
            description: Group description
            participant_ids: List of participant IDs to add initially
            **kwargs: Additional group parameters

        Returns:
            Created participant group details

        Note:
            Either workspace_id or organisation_id must be specified.
        """
        data = {"name": name}
        if workspace_id:
            data["workspace_id"] = workspace_id
        if organisation_id:
            data["organisation_id"] = organisation_id
        if description:
            data["description"] = description
        if participant_ids:
            data["participant_ids"] = participant_ids
        data.update(kwargs)

        return self._request("POST", "/participant-groups/", json=data)

    def update_participant_group(self, group_id: str, **kwargs) -> Dict:
        """
        Update a participant group.

        Args:
            group_id: The participant group ID
            **kwargs: Fields to update

        Returns:
            Updated participant group details
        """
        return self._request("PATCH", f"/participant-groups/{group_id}/", json=kwargs)

    def delete_participant_group(self, group_id: str) -> Dict:
        """
        Delete a participant group.

        Args:
            group_id: The participant group ID

        Returns:
            Empty response on success
        """
        return self._request("DELETE", f"/participant-groups/{group_id}/")

    def add_participants_to_group(
        self,
        group_id: str,
        participant_ids: List[str]
    ) -> Dict:
        """
        Add participants to a group.

        Args:
            group_id: The participant group ID
            participant_ids: List of participant IDs to add

        Returns:
            Updated participant group details
        """
        return self._request(
            "POST",
            f"/participant-groups/{group_id}/participants/",
            json={"participant_ids": participant_ids}
        )

    def remove_participants_from_group(
        self,
        group_id: str,
        participant_ids: List[str]
    ) -> Dict:
        """
        Remove participants from a group.

        Args:
            group_id: The participant group ID
            participant_ids: List of participant IDs to remove

        Returns:
            Updated participant group details
        """
        return self._request(
            "DELETE",
            f"/participant-groups/{group_id}/participants/",
            json={"participant_ids": participant_ids}
        )

    # ==================== Messages ====================

    def list_messages(
        self,
        user_id: Optional[str] = None,
        created_after: Optional[str] = None
    ) -> Dict:
        """
        Retrieve messages between you and another user, or all recent messages.

        Args:
            user_id: Another user ID to get messages with specific user
            created_after: Fetch messages after this timestamp (ISO8601 format).
                          Can fetch up to last 30 days. Either user_id or
                          created_after must be provided.

        Returns:
            Dictionary with 'results' containing list of messages

        Note:
            Either user_id or created_after must be provided.
        """
        params = {}
        if user_id:
            params["user_id"] = user_id
        if created_after:
            params["created_after"] = created_after

        return self._request("GET", "/messages/", params=params)

    def list_unread_messages(self) -> Dict:
        """
        List all unread messages.

        Returns:
            Dictionary with 'results' containing list of unread messages
        """
        return self._request("GET", "/messages/unread/")

    def send_message(
        self,
        recipient_id: str,
        body: str,
        study_id: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Send a message to a participant or another researcher.

        Args:
            recipient_id: The participant or researcher ID
            body: Message body
            study_id: Optional study ID (required when messaging participants)
            **kwargs: Additional message parameters

        Returns:
            Empty dict on success (204 response)
        """
        data = {
            "recipient_id": recipient_id,
            "body": body
        }
        if study_id:
            data["study_id"] = study_id
        data.update(kwargs)

        return self._request("POST", "/messages/", json=data)

    def send_bulk_message(
        self,
        study_id: str,
        body: str,
        participant_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Send a message to multiple participants.

        Args:
            study_id: The study ID
            body: Message body
            participant_ids: List of participant IDs (if None, sends to all)
            **kwargs: Additional message parameters

        Returns:
            Bulk message result
        """
        data = {
            "study_id": study_id,
            "body": body
        }
        if participant_ids:
            data["participant_ids"] = participant_ids
        data.update(kwargs)

        return self._request("POST", "/messages/bulk/", json=data)

    # ==================== Filters & Eligibility ====================

    def list_eligibility_requirements(self) -> Dict:
        """
        List all available eligibility requirements/filters.

        Returns:
            Dictionary with 'results' containing list of eligibility requirements
        """
        return self._request("GET", "/eligibility-requirements/")

    def list_filters(
        self,
        workspace_id: Optional[str] = None,
        detailed: Optional[bool] = None,
        filter_tag: Optional[str] = None
    ) -> Dict:
        """
        List all available filters that can be applied to studies or filter sets.

        Args:
            workspace_id: The workspace ID to get contextual data (e.g., participant groups)
            detailed: Include extra categorization info for Prolific app (not needed for API)
            filter_tag: Filter by tag (e.g., "custom-group" for Custom Group filters only)

        Returns:
            Dictionary with 'results' containing list of filters
        """
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if detailed is not None:
            params["detailed"] = "true" if detailed else "false"
        if filter_tag:
            params["filter_tag"] = filter_tag

        return self._request("GET", "/filters/", params=params)

    def get_filter(self, filter_id: str) -> Dict:
        """
        Get details of a specific filter.

        Args:
            filter_id: The filter ID

        Returns:
            Filter details
        """
        return self._request("GET", f"/filters/{filter_id}/")

    def get_eligibility_count(
        self,
        eligibility_requirements: List[Dict]
    ) -> Dict:
        """
        Get the number of participants matching eligibility requirements.

        Args:
            eligibility_requirements: List of eligibility requirement dicts

        Returns:
            Count information
        """
        return self._request(
            "POST",
            "/eligibility-count/",
            json={"eligibility_requirements": eligibility_requirements}
        )

    def get_filter_distribution(self, filter_id: str) -> Dict:
        """
        Get the distribution of responses for a filter.

        Args:
            filter_id: The filter ID

        Returns:
            Distribution data
        """
        return self._request("GET", f"/filters/{filter_id}/distribution/")

    # ==================== Filter Sets ====================

    def list_filter_sets(
        self,
        workspace_id: Optional[str] = None,
        organisation_id: Optional[str] = None
    ) -> Dict:
        """
        List all filter sets in a workspace or organisation.

        Args:
            workspace_id: Optional workspace ID
            organisation_id: Optional organisation ID

        Returns:
            Dictionary with 'results' containing list of filter sets
        """
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if organisation_id:
            params["organisation_id"] = organisation_id
        return self._request("GET", "/filter-sets/", params=params)

    def get_filter_set(self, filter_set_id: str) -> Dict:
        """
        Get details of a specific filter set.

        Args:
            filter_set_id: The filter set ID

        Returns:
            Filter set details
        """
        return self._request("GET", f"/filter-sets/{filter_set_id}/")

    def create_filter_set(
        self,
        workspace_id: str,
        name: str,
        filters: List[Dict],
        **kwargs
    ) -> Dict:
        """
        Create a new filter set.

        Args:
            workspace_id: The workspace ID
            name: Filter set name
            filters: List of filter dicts with filter_id and selected_values/selected_range
            **kwargs: Additional parameters

        Returns:
            Created filter set details
        """
        data = {
            "workspace_id": workspace_id,
            "name": name,
            "filters": filters
        }
        data.update(kwargs)
        return self._request("POST", "/filter-sets/", json=data)

    def update_filter_set(self, filter_set_id: str, **kwargs) -> Dict:
        """
        Update a filter set.

        Args:
            filter_set_id: The filter set ID
            **kwargs: Fields to update

        Returns:
            Updated filter set details
        """
        return self._request("PATCH", f"/filter-sets/{filter_set_id}/", json=kwargs)

    def delete_filter_set(self, filter_set_id: str) -> Dict:
        """
        Delete a filter set.

        Args:
            filter_set_id: The filter set ID

        Returns:
            Empty response on success
        """
        return self._request("DELETE", f"/filter-sets/{filter_set_id}/")

    def clone_filter_set(self, filter_set_id: str) -> Dict:
        """
        Clone an existing filter set.

        Args:
            filter_set_id: The filter set ID to clone

        Returns:
            New cloned filter set details
        """
        return self._request("POST", f"/filter-sets/{filter_set_id}/clone/")

    def lock_filter_set(self, filter_set_id: str) -> Dict:
        """
        Lock a filter set to prevent modifications.

        Args:
            filter_set_id: The filter set ID

        Returns:
            Updated filter set details
        """
        return self._request("POST", f"/filter-sets/{filter_set_id}/lock/")

    def unlock_filter_set(self, filter_set_id: str) -> Dict:
        """
        Unlock a filter set to allow modifications.

        Args:
            filter_set_id: The filter set ID

        Returns:
            Updated filter set details
        """
        return self._request("POST", f"/filter-sets/{filter_set_id}/unlock/")

    # ==================== Surveys ====================

    def list_surveys(
        self,
        researcher_id: str,
        offset: int = 0,
        limit: int = 100
    ) -> Dict:
        """
        List all surveys for a researcher.

        Args:
            researcher_id: The researcher ID
            offset: Pagination offset (default: 0)
            limit: Pagination limit (default: 100, max: 1000)

        Returns:
            Dictionary with survey list
        """
        return self._request(
            "GET",
            "/surveys/",
            params={
                "researcher_id": researcher_id,
                "offset": offset,
                "limit": limit
            }
        )

    def get_survey(self, survey_id: str) -> Dict:
        """
        Get details of a specific survey.

        Args:
            survey_id: The survey ID

        Returns:
            Survey details
        """
        return self._request("GET", f"/surveys/{survey_id}")

    def create_survey(
        self,
        researcher_id: str,
        title: str,
        sections: Optional[List[Dict]] = None,
        questions: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict:
        """
        Create a new survey.

        Args:
            researcher_id: The researcher ID
            title: Survey title
            sections: List of section dicts (for Prolific app rendering)
            questions: List of question dicts (alternative to sections)
            **kwargs: Additional survey parameters

        Returns:
            Created survey details
        """
        data = {
            "researcher_id": researcher_id,
            "title": title
        }
        if sections:
            data["sections"] = sections
        if questions:
            data["questions"] = questions
        data.update(kwargs)
        return self._request("POST", "/surveys/", json=data)

    def update_survey(self, survey_id: str, **kwargs) -> Dict:
        """
        Update a survey.

        Args:
            survey_id: The survey ID
            **kwargs: Fields to update

        Returns:
            Updated survey details
        """
        return self._request("PATCH", f"/surveys/{survey_id}", json=kwargs)

    def delete_survey(self, survey_id: str) -> Dict:
        """
        Delete a survey.

        Args:
            survey_id: The survey ID

        Returns:
            Empty response on success
        """
        return self._request("DELETE", f"/surveys/{survey_id}")

    def list_survey_responses(
        self,
        survey_id: str,
        offset: int = 0,
        limit: int = 100
    ) -> Dict:
        """
        List all responses for a survey.

        Args:
            survey_id: The survey ID
            offset: Pagination offset (default: 0)
            limit: Pagination limit (default: 100)

        Returns:
            Dictionary with response list
        """
        return self._request(
            "GET",
            f"/surveys/{survey_id}/responses/",
            params={"offset": offset, "limit": limit}
        )

    def get_survey_response(self, survey_id: str, response_id: str) -> Dict:
        """
        Get a specific survey response.

        Args:
            survey_id: The survey ID
            response_id: The response ID

        Returns:
            Response details
        """
        return self._request("GET", f"/surveys/{survey_id}/responses/{response_id}")

    def get_survey_response_summary(self, survey_id: str) -> Dict:
        """
        Get summary of all responses for a survey.

        Args:
            survey_id: The survey ID

        Returns:
            Response summary statistics
        """
        return self._request("GET", f"/surveys/{survey_id}/responses/summary/")

    # ==================== Webhooks ====================

    def list_webhook_event_types(self) -> Dict:
        """
        List all subscribable webhook event types.

        Returns:
            Dictionary with list of event types
        """
        return self._request("GET", "/hooks/event-types/")

    def list_webhook_secrets(self, workspace_id: str) -> Dict:
        """
        List all webhook secrets for a workspace.

        Args:
            workspace_id: The workspace ID

        Returns:
            Dictionary with list of secrets
        """
        return self._request("GET", "/hooks/secrets/", params={"workspace_id": workspace_id})

    def create_webhook_secret(self, workspace_id: str) -> Dict:
        """
        Create or replace a webhook secret for verifying payloads.

        Args:
            workspace_id: The workspace ID

        Returns:
            Secret details
        """
        return self._request("POST", "/hooks/secrets/", json={"workspace_id": workspace_id})

    def list_webhook_subscriptions(self) -> Dict:
        """
        List all webhook subscriptions.

        Returns:
            Dictionary with list of subscriptions
        """
        return self._request("GET", "/hooks/subscriptions/")

    def get_webhook_subscription(self, subscription_id: str) -> Dict:
        """
        Get details of a specific webhook subscription.

        Args:
            subscription_id: The subscription ID

        Returns:
            Subscription details
        """
        return self._request("GET", f"/hooks/subscriptions/{subscription_id}/")

    def create_webhook_subscription(
        self,
        event_type: str,
        url: str,
        **kwargs
    ) -> Dict:
        """
        Create a new webhook subscription.

        Args:
            event_type: The event type to subscribe to
            url: The webhook URL to receive events
            **kwargs: Additional subscription parameters

        Returns:
            Created subscription details
        """
        data = {
            "event_type": event_type,
            "url": url
        }
        data.update(kwargs)
        return self._request("POST", "/hooks/subscriptions/", json=data)

    def update_webhook_subscription(self, subscription_id: str, **kwargs) -> Dict:
        """
        Update a webhook subscription.

        Args:
            subscription_id: The subscription ID
            **kwargs: Fields to update

        Returns:
            Updated subscription details
        """
        return self._request("PATCH", f"/hooks/subscriptions/{subscription_id}/", json=kwargs)

    def delete_webhook_subscription(self, subscription_id: str) -> Dict:
        """
        Delete a webhook subscription.

        Args:
            subscription_id: The subscription ID

        Returns:
            Empty response on success
        """
        return self._request("DELETE", f"/hooks/subscriptions/{subscription_id}/")

    def list_webhook_subscription_events(
        self,
        subscription_id: str,
        offset: int = 0,
        limit: int = 100
    ) -> Dict:
        """
        List events for a webhook subscription.

        Args:
            subscription_id: The subscription ID
            offset: Pagination offset (default: 0)
            limit: Pagination limit (default: 100)

        Returns:
            Dictionary with event list
        """
        return self._request(
            "GET",
            f"/hooks/subscriptions/{subscription_id}/events/",
            params={"offset": offset, "limit": limit}
        )

    # ==================== Invitations ====================

    def create_invitation(
        self,
        workspace_id: str,
        email: str,
        role: str,
        **kwargs
    ) -> Dict:
        """
        Create an invitation to add a user to a workspace.

        Args:
            workspace_id: The workspace ID
            email: Email address to invite
            role: Role to assign (e.g., "admin", "member")
            **kwargs: Additional invitation parameters

        Returns:
            Invitation details
        """
        data = {
            "workspace_id": workspace_id,
            "email": email,
            "role": role
        }
        data.update(kwargs)
        return self._request("POST", "/invitations/", json=data)

    # ==================== Reward Recommendations ====================

    def get_reward_recommendations(
        self,
        workspace_id: str,
        currency: str,
        screener_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Calculate recommended reward rates for a study.

        Args:
            workspace_id: The workspace ID
            currency: Currency code (USD or GBP)
            screener_ids: Optional list of filter IDs to consider

        Returns:
            Reward recommendations
        """
        params = {
            "workspace_id": workspace_id,
            "currency": currency
        }
        if screener_ids:
            params["screener_ids"] = ",".join(screener_ids)
        return self._request("GET", "/reward-recommendations/", params=params)

    # ==================== Helper Methods ====================

    def create_and_publish_study(
        self,
        name: str,
        internal_name: str,
        description: str,
        external_study_url: str,
        total_available_places: int,
        estimated_completion_time: int,
        reward: int,
        **kwargs
    ) -> Dict:
        """
        Helper method to create and immediately publish a study.

        Args:
            Same as create_study()

        Returns:
            Published study details
        """
        study = self.create_study(
            name=name,
            internal_name=internal_name,
            description=description,
            external_study_url=external_study_url,
            total_available_places=total_available_places,
            estimated_completion_time=estimated_completion_time,
            reward=reward,
            **kwargs
        )

        study_id = study["id"]
        return self.publish_study(study_id)

    def get_all_active_studies(self) -> List[Dict]:
        """
        Helper method to get all active studies.

        Returns:
            List of active studies
        """
        response = self.list_studies(state="ACTIVE")
        return response.get("results", [])

    def approve_all_submissions(self, study_id: str) -> Dict:
        """
        Helper method to approve all pending submissions for a study.

        Args:
            study_id: The study ID

        Returns:
            Result of bulk approval
        """
        return self.bulk_approve_submissions(study_id)
