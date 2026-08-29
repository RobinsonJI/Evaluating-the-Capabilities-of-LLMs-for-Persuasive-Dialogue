import gradio as gr
import os
from typing import List, Dict
from datetime import datetime
import asyncio
import random

from client.client import BasePersuasuiClient
from client.models import ChatItem, UtteranceType, UTT_TITLE, UTT_DESC

class GradioPersuasuiClient(BasePersuasuiClient):
    """
    The Gradio-specific implementation of the Persuasui client.

    This class inherits the core application logic from BasePersuasuiClient
    and implements the methods required to build and run a Gradio user interface.
    """
    def __init__(self, **kwargs):
        """
        Initialises the Gradio client.

        It calls the base class constructor and then initializes Gradio-specific
        attributes and loads custom CSS.
        """
        super().__init__(**kwargs)
        self.interface = None
        self.custom_css = self._load_css()
        # JavaScript to force light mode and prevent theme switching
        self.force_light_mode_js = """
        // Set theme to light on load
        if (document.body.classList.contains('dark')) {
            document.body.classList.remove('dark');
        }
        // Override Gradio's theme toggle to always stay in light mode
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.attributeName === 'class' && document.body.classList.contains('dark')) {
                    document.body.classList.remove('dark');
                }
            });
        });
        observer.observe(document.body, { attributes: true });
        """

    def _load_css(self) -> str:
        """Loads custom CSS for the Gradio interface."""
        css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
        try:
            with open(css_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            self._log_event(
                event_type="system",
                participant_id="SYSTEM",
                message=f"CSS file not found at {css_path}",
                severity="warning"
            )
            return ""

    # --- OPPONENT TYPE DETECTION ---

    def _is_opponent_ai(self, session_info: dict, participant_id: str) -> bool:
        """
        Determines if the opponent in a session is an AI model.

        This is used to decide whether to apply a simulated delay before showing
        the opponent's response, helping blind users to whether their opponent
        is human or AI.

        Args:
            session_info: The session dictionary containing parameters
            participant_id: The current user's participant ID

        Returns:
            True if opponent is AI (BASE, MAS, or MAS_RAG), False if human
        """
        params = session_info.get("parameters", {})
        first_speaker = params.get("first_speaker")

        # Determine which speaker is the opponent based on current user's role
        if participant_id == first_speaker:
            opponent_type = params.get("second_speaker_type", "")
        else:
            opponent_type = params.get("first_speaker_type", "")

        # AI types: BASE, MAS, MAS_RAG (not HUMAN)
        return opponent_type.upper() in ["BASE", "MAS", "MAS_RAG"]

    # --- GRADIO-SPECIFIC RENDERING HELPERS ---

    def _render_chat_to_html(self, chat_items: list[ChatItem]) -> str:
        """Takes a list of ChatItem models and returns a Gradio-compatible HTML string.
        Never shows participant IDs."""
        if not chat_items:
            return "<div class='chat-empty-state'>No messages</div>"

        html = "<div class='chat-container'>"
        for item in chat_items:
            css_class = "chat-message-current-user" if item.is_user else "chat-message-other-user"
            try:
                time_str = datetime.fromisoformat(item.timestamp).strftime('%H:%M:%S')
            except (ValueError, TypeError):
                time_str = ""

            # Show "You" or "Opponent" instead of participant IDs
            speaker_label = "You" if item.is_user else "Opponent"

            html += f"""
            <div class='chat-message {css_class}'>
                <div class='chat-message-header'>
                    {speaker_label}
                    <span class='chat-message-turn-number'>Turn {item.turn_number}</span>
                </div>
                <div class='chat-message-content'>{item.content}</div>
                <div class='chat-message-timestamp'>{time_str}</div>
            </div>
            """
        html += "</div>"
        return html

    def _render_session_info_to_markdown(
        self,
        session_info: dict,
        first_speaker_commitments: list[str],
        second_speaker_commitments: list[str],
        participant_role: str
    ) -> str:
        """Minimal session info: code, status badge, and commitments for both speakers."""
        status = session_info.get("status", "unknown")

        # Status badge with color
        if status == "Running.":
            status_badge = "🟢 Ongoing"
        elif status == "Finished.":
            status_badge = "🔴 Finished"
        else:
            status_badge = "🟡 Starting"

        # Determine which commitments are "yours" vs "opponent's"
        if participant_role == "first_speaker":
            your_commitments = first_speaker_commitments
            opponent_commitments = second_speaker_commitments
        elif participant_role == "second_speaker":
            your_commitments = second_speaker_commitments
            opponent_commitments = first_speaker_commitments
        else:
            your_commitments = []
            opponent_commitments = []

        # Format commitments
        your_commits_md = "\n".join([f"  - {c}" for c in your_commitments]) if your_commitments else "  - None yet"
        opponent_commits_md = "\n".join([f"  - {c}" for c in opponent_commitments]) if opponent_commitments else "  - None yet"

        # Calculate turn count
        current_turns = len(session_info.get("dialogue_history", []))
        max_turns = session_info.get("parameters", {}).get("max_dialogue_turns", "?")

        return f"""### Session Info

**Session Code**: `{session_info.get('session_id')}`

**Status**: {status_badge}

**Turns**: {current_turns}/{max_turns}

---

**Your Commitments**:
{your_commits_md}

**Opponent's Commitments**:
{opponent_commits_md}
"""

    def _render_typical_replies_to_html(self, typical_replies: list, session_info: dict, participant_role: str) -> str:
        """Format typical replies as HTML for display above chat box.

        Expected structure: [[{utterance_type: sentence}, count, sentence_idx], ...]

        Args:
            typical_replies: List of typical reply items from the API
            session_info: Session dict containing dialogue_history
            participant_role: Current participant's role ("first_speaker" or "second_speaker")
        """
        if not typical_replies:
            return ""

        # Get opponent's last turn to extract utterance classifications
        dialogue_history = session_info.get("dialogue_history", [])
        opponent_last_turn = None

        # Find the most recent turn that's NOT from the current user
        participant_id = session_info.get("parameters", {}).get("first_speaker") if participant_role == "first_speaker" else session_info.get("parameters", {}).get("second_speaker")
        for turn in reversed(dialogue_history):
            if turn.get("speaker") != participant_id:
                opponent_last_turn = turn
                break

        html = "<div class='typical-replies-container' style='margin: 12px 0;'>"
        html += "<div class='typical-replies-header' style='font-weight: bold; margin-bottom: 8px; font-size: 1.1em;'>Suggested replies:</div>"

        # Parse the typical replies structure
        for reply_item in typical_replies:
            if not isinstance(reply_item, list) or len(reply_item) < 3:
                continue

            utterance_dict = reply_item[0]
            sentence_idx = reply_item[2]

            if not isinstance(utterance_dict, dict):
                continue

            # Get the sentence being responded to (first value in dict, they're all the same)
            sentence = next(iter(utterance_dict.values())) if utterance_dict else "..."

            # Get the utterance classification for this sentence from opponent's turn
            sentence_classification = ""
            if opponent_last_turn and opponent_last_turn.get("sentences_with_utterance_types"):
                classifications = opponent_last_turn.get("sentences_with_utterance_types", [])
                if sentence_idx < len(classifications):
                    utt_type_raw, _ = classifications[sentence_idx]
                    # Clean up the classification (remove underscores)
                    sentence_classification = f" [{utt_type_raw.strip('_')}]"

            # Create a box for this sentence's reply options
            html += f"""<div class='typical-reply-response-box'>
                <div class='typical-reply-header'>
                    In response to: "{sentence}"{sentence_classification}
                </div>
                <div class='typical-reply-options'>"""

            # Add each utterance type option
            for utt_key, _ in utterance_dict.items():
                # Map the key (e.g., "___Why___") to UtteranceType enum
                try:
                    # Remove underscores and convert to uppercase
                    utt_name = utt_key.strip('_').upper()
                    utt_type = UtteranceType[utt_name]
                    title = UTT_TITLE.get(utt_type, utt_name)
                    desc = UTT_DESC.get(utt_type, "")

                    # Create a styled button/badge for each reply type
                    html += f"""<div class='typical-reply-button' title='{desc}'>{title}</div>"""
                except (KeyError, AttributeError):
                    continue

            html += "</div></div>"  # Close flex container and box

        html += "</div>"  # Close main container
        return html

    def _render_header_to_markdown(self, topic: str, status_text: str, session_info: dict, participant_role: str, participant_id: str) -> str:
        """Render topic, turn indicator, and political perspective as header markdown."""
        header = f"## {topic}\n\n"

        # Add political perspective for human participants (below topic, above status)
        if participant_role in ["first_speaker", "second_speaker"]:
            parameters = session_info.get("parameters", {})

            # Get the political position range for this participant
            if participant_role == "first_speaker":
                political_range = parameters.get("first_speaker_political_political_position_range", "")
            else:
                political_range = parameters.get("second_speaker_political_position_range", "")

            # Parse and interpret the range
            if political_range and ":" in political_range:
                try:
                    range_parts = political_range.split(":")
                    min_val = int(range_parts[0])
                    max_val = int(range_parts[1])
                    midpoint = (min_val + max_val) / 2

                    # Determine political wing
                    wing = "left" if midpoint < 50 else "right"

                    header += f"You are arguing from a **{wing}-wing** perspective\n\n"
                except (ValueError, IndexError):
                    pass  # Skip if parsing fails

        # Add status/turn indicator
        header += f"{status_text}"

        return header

    def _render_admin_dashboard_to_html(self, dashboard_data: dict) -> str:
        """Takes admin dashboard data and generates the complete HTML for the monitoring panel."""
        # This is a direct port of the HTML generation from old_app.py
        monitoring_html = "<div class='admin-dashboard'>"
        
        # Recent Activity
        monitoring_html += "<div class='admin-card'><h3>Recent Activity</h3><div class='activity-timeline'>"
        recent_activities = dashboard_data.get('recent_activities', [])
        if not recent_activities:
            monitoring_html += "<p>No recent activity</p>"
        else:
            for activity in reversed(recent_activities[-15:]):
                session_id_info = f" (Session: {activity.get('session_id')})" if activity.get('session_id') else ''
                severity = activity.get('severity', 'info')
                monitoring_html += f"""
                    <div class='activity-item severity-{severity}'>
                        <div class='activity-timestamp'>{activity.get('timestamp', '')}</div>
                        <div class='activity-type {activity.get('event_type', '')}'>[{severity.upper()}] {activity.get('event_type', '').upper()}</div>
                        <div class='activity-message'>
                            <strong>{activity.get('participant_id', '')}:</strong> {activity.get('message', '')}{session_id_info}
                        </div>
                    </div>"""
        monitoring_html += "</div></div>"

        # Build participant-to-sessions mapping
        sessions = dashboard_data.get('sessions', [])
        participant_sessions = {}
        for session in sessions:
            session_id = session.get('session_id')
            status = session.get('status', 'unknown').lower().replace('.', '')
            topic = session.get('topic', 'N/A')
            participants = session.get('participants', {})

            # Add session to participant1's list
            p1 = participants.get('participant1', {})
            p1_id = p1.get('id')
            if p1_id:
                if p1_id not in participant_sessions:
                    participant_sessions[p1_id] = []
                participant_sessions[p1_id].append({
                    'session_id': session_id,
                    'status': status,
                    'topic': topic
                })

            # Add session to participant2's list
            p2 = participants.get('participant2', {})
            p2_id = p2.get('id')
            if p2_id:
                if p2_id not in participant_sessions:
                    participant_sessions[p2_id] = []
                participant_sessions[p2_id].append({
                    'session_id': session_id,
                    'status': status,
                    'topic': topic
                })

        # Participants
        monitoring_html += "<div class='admin-grid'>"
        all_participants = dashboard_data.get('participants', [])
        human_participants = [p for p in all_participants if not p.get('is_ai', False)]
        ai_participants = [p for p in all_participants if p.get('is_ai', False)]

        # Human Participants Card
        monitoring_html += "<div class='admin-card'><h3>Human Participants</h3><div class='all-sessions-scrollable'>"
        if not human_participants:
            monitoring_html += "<p>No human participants found</p>"
        else:
            for p in human_participants:
                status_label = "Offline"
                if p.get('is_authenticated'):
                    status_label = "Active" if p.get('current_session') else "Online"

                # Get all sessions for this participant
                participant_id = p.get('id')
                current_session = p.get('current_session')
                assigned_sessions = participant_sessions.get(participant_id, [])

                # Build sessions HTML
                if not assigned_sessions:
                    sessions_html = "<div class='participant-session-item'>None</div>"
                else:
                    sessions_html = ""
                    for sess in assigned_sessions:
                        # Only highlight as current if it matches AND is still active
                        is_current = (sess['session_id'] == current_session and
                                     sess['status'] in ['running', 'active', 'started', 'waiting'])
                        current_class = " participant-session-current" if is_current else ""
                        status_class = f" {sess['status']}"
                        sessions_html += f"""<div class='participant-session-item{current_class}{status_class}'>{sess['session_id']}</div>"""

                monitoring_html += f"""
                <div class='participant-item status-{status_label.lower()}'>
                    <div class='participant-header'><h4>{participant_id}</h4><span class='participant-status-badge {status_label.lower()}'>{status_label}</span></div>
                    <div class='participant-info'><div class='participant-type'><strong>Type:</strong> Human</div>
                        <div class='participant-details'>
                            <div class='detail-item'><span class='detail-label'>Auth Code:</span> <span class='detail-value'>{p.get('auth_code', 'N/A')}</span></div>
                            <div class='detail-item detail-item-sessions'>
                                <span class='detail-label'>Assigned Sessions:</span>
                                <div class='participant-sessions-list'>{sessions_html}</div>
                            </div>
                        </div>
                    </div>
                </div>"""
        monitoring_html += "</div></div>"

        # AI Models Card
        monitoring_html += "<div class='admin-card'><h3>AI Models</h3><div class='all-sessions-scrollable'>"
        if not ai_participants:
            monitoring_html += "<p>No AI models found</p>"
        else:
            for p in ai_participants:
                status_label = "Active" if p.get('current_session') else "Online"

                # Get all sessions for this participant
                participant_id = p.get('id')
                current_session = p.get('current_session')
                assigned_sessions = participant_sessions.get(participant_id, [])

                # Build sessions HTML
                if not assigned_sessions:
                    sessions_html = "<div class='participant-session-item'>None</div>"
                else:
                    sessions_html = ""
                    for sess in assigned_sessions:
                        # Only highlight as current if it matches AND is still active
                        is_current = (sess['session_id'] == current_session and
                                     sess['status'] in ['running', 'active', 'started', 'waiting'])
                        current_class = " participant-session-current" if is_current else ""
                        status_class = f" {sess['status']}"
                        sessions_html += f"""<div class='participant-session-item{current_class}{status_class}'>{sess['session_id']}</div>"""

                monitoring_html += f"""
                <div class='participant-item status-{status_label.lower()}'>
                    <div class='participant-header'><h4>{participant_id}</h4><span class='participant-status-badge {status_label.lower()}'>{status_label}</span></div>
                    <div class='participant-info'><div class='participant-type'><strong>Type:</strong> AI Model</div>
                        <div class='participant-details'>
                            <div class='detail-item'><span class='detail-label'>Messages:</span> <span class='detail-value'>{p.get('message_count', 0)}</span></div>
                            <div class='detail-item detail-item-sessions'>
                                <span class='detail-label'>Assigned Sessions:</span>
                                <div class='participant-sessions-list'>{sessions_html}</div>
                            </div>
                        </div>
                    </div>
                </div>"""
        monitoring_html += "</div></div>"
        monitoring_html += "</div>" # Close admin-grid

        # All Sessions
        monitoring_html += "<div class='all-sessions-container'><h3>All Sessions</h3><div class='all-sessions-scrollable'>"
        sessions = dashboard_data.get('sessions', [])
        if not sessions:
            monitoring_html += "<p>No sessions found</p>"
        else:
            for s in sessions:
                p1 = s.get('participants', {}).get('participant1', {})
                p2 = s.get('participants', {}).get('participant2', {})
                status = s.get('status', 'unknown').lower().replace('.', '')
                logs = s.get('logs', [])

                monitoring_html += f"""
                <div class='session-detail-item session-detail-item-scrollable status-{status}'>
                    <div class='session-detail-grid'>
                        <div class='session-detail-main'>
                            <h4>Session {s.get('session_id')}</h4>
                            <p class='session-topic'>Topic: {s.get('topic')}</p>
                            <p>Status: <span class='session-status-label {status}'>{status.upper()}</span></p>
                        </div>
                        <div class='session-participants'>
                            <p><strong>Participants:</strong></p>
                            <p>{p1.get('id')} ({'AI' if p1.get('is_ai') else 'Human'}) - {'Joined' if p1.get('joined') else 'Not joined'}</p>
                            <p>{p2.get('id')} ({'AI' if p2.get('is_ai') else 'Human'}) - {'Joined' if p2.get('joined') else 'Not joined'}</p>
                        </div>
                    </div>"""

                # Add logs section
                if logs:
                    monitoring_html += """
                    <div class='session-logs session-logs-container'>
                        <p><strong>Persuasio Logs:</strong></p>
                        <pre class='session-logs-pre'>"""
                    for log_entry in logs:
                        # Escape HTML characters in log entries
                        log_entry_escaped = str(log_entry).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        monitoring_html += f"{log_entry_escaped}\n"
                    monitoring_html += "</pre></div>"

                monitoring_html += "</div>"
        monitoring_html += "</div></div>"

        # Client Logs
        monitoring_html += "<div class='admin-card'><h3>Client Logs (Last 50)</h3>"
        client_logs = dashboard_data.get('client_logs', [])
        if not client_logs:
            monitoring_html += "<p>No client logs found</p>"
        else:
            monitoring_html += "<pre class='client-logs-pre'>"
            for log in reversed(client_logs):
                session_info = f" [session={log.get('session_id')}]" if log.get('session_id') else ""
                log_line = f"[{log.get('timestamp')}] [{log.get('severity', '').upper()}] [{log.get('event_type')}] {log.get('participant_id')}: {log.get('message')}{session_info}\n"
                monitoring_html += log_line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            monitoring_html += "</pre>"
        monitoring_html += "</div>"

        monitoring_html += f"<div class='timestamp'>Last updated: {dashboard_data.get('timestamp', '')[:19].replace('T', ' ')}</div></div>"
        return monitoring_html

    # --- GRADIO-SPECIFIC EVENT HANDLERS (ADAPTERS) ---

    async def _gradio_login_handler(self, code: str):
        """Adapter for the login event."""
        result = await self.handle_login(code)
        if result.error:
            return result.error, None, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

        if result.is_admin:
            return "Admin Access Granted", result.participant_id, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        else:
            return result.message, result.participant_id, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    async def _gradio_join_handler(self, participant_id: str, session_code: str):
        """Adapter for the join session event."""
        result = await self.handle_join_session(participant_id, session_code)
        if result.error:
            return result.error, {}

        return result.message, result.user_context

    async def _gradio_send_handler(self, user_context: dict, delay_state: dict):
        """Adapter for the send message event.

        Reads the message from delay_state["pending_message"] (stored by _gradio_pre_send).
        No outputs - input clearing is handled by _gradio_pre_send.
        """
        message = delay_state.get("pending_message", "")
        if not message:
            print(f"[SEND] No pending message in delay_state", flush=True)
            return

        result = await self.handle_send_message(user_context, message)
        if result.error:
            print(f"[SEND] Error: {result.error}", flush=True)
            # Note: We can't restore the message since input is already cleared
            # The error will be visible in the status after refresh
        else:
            print(f"[SEND] Message sent successfully", flush=True)

    async def _gradio_refresh_handler(self, user_context: dict, delay_state: dict = None):
        """Adapter for the UI refresh event. Always returns fresh data from server.

        Returns: (status_display, chat_display, session_info_display, topic_display,
                  typical_replies_display, message_input, end_session_btn)
        """
        # Check if we're in a delay period - if so, return frozen chat with "Waiting" status
        print(f"[REFRESH] delay_state: {delay_state}", flush=True)
        if delay_state and delay_state.get("in_delay", False):
            print(f"[REFRESH] In delay period - returning frozen chat with 'Waiting' status", flush=True)

            # If snapshot is empty (pre-send phase, before _gradio_start_delay populates it),
            # keep all displays unchanged to avoid flashing empty content
            if not delay_state.get("chat_snapshot"):
                print(f"[REFRESH] Snapshot empty (pre-send phase), keeping displays unchanged", flush=True)
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=False)

            if not user_context:
                return "Enter a session code to join.", delay_state.get("chat_snapshot", ""), "", "", "", gr.update(interactive=False), gr.update(interactive=False)

            result = await self.handle_refresh(user_context)
            if result.error:
                return result.error, delay_state.get("chat_snapshot", ""), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            # Get participant role
            session = result.session_info
            participant_id = user_context.get("participant_id")
            if participant_id == session.get("parameters", {}).get("first_speaker"):
                participant_role = "first_speaker"
            elif participant_id == session.get("parameters", {}).get("second_speaker"):
                participant_role = "second_speaker"
            else:
                participant_role = "observer"

            # During delay, always show "Waiting for opponent to respond..."
            # This blinds the user to whether opponent is AI (instant) or human (slow)
            waiting_status = "Waiting for opponent to respond..."

            # Render header with waiting status (not the real status)
            topic = session.get("parameters", {}).get("debate_topic", "N/A")
            header_md = self._render_header_to_markdown(topic, waiting_status, session, participant_role, participant_id)

            # During delay: disable both message input and end session button
            message_input_update = gr.update(interactive=False)
            end_session_interactive = gr.update(interactive=False)

            # Return frozen snapshots for chat and typical replies, but keep session info (commitments) current
            return "", delay_state.get("chat_snapshot", ""), delay_state.get("session_info_snapshot", ""), header_md, delay_state.get("typical_replies_snapshot", ""), message_input_update, end_session_interactive

        # Normal refresh (not in delay)
        if not user_context:
            return "Enter a session code to join.", "", "", "", "", gr.update(interactive=False), gr.update(interactive=False)

        # Get fresh session state (always returns full data, no change detection)
        result = await self.handle_refresh(user_context)

        if result.error:
            self._log_event(
                event_type="system",
                participant_id=user_context.get("participant_id", "SYSTEM"),
                message=f"UI Refresh failed: {result.error}",
                session_id=user_context.get("session_id"),
                severity="error"
            )
            return result.error, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        # Get participant role from session data
        session = result.session_info
        participant_id = user_context.get("participant_id")
        if participant_id == session.get("parameters", {}).get("first_speaker"):
            participant_role = "first_speaker"
        elif participant_id == session.get("parameters", {}).get("second_speaker"):
            participant_role = "second_speaker"
        else:
            participant_role = "observer"

        # Always render and return fresh UI data
        chat_html = self._render_chat_to_html(result.chat_items)
        session_info_md = self._render_session_info_to_markdown(
            result.session_info,
            result.first_speaker_commitments or [],
            result.second_speaker_commitments or [],
            participant_role
        )
        typical_replies_html = self._render_typical_replies_to_html(
            result.typical_replies or [],
            result.session_info,
            participant_role
        )

        # Extract topic from session
        topic = session.get("parameters", {}).get("debate_topic", "N/A")
        header_md = self._render_header_to_markdown(topic, result.status_text, session, participant_role, participant_id)

        # Determine if inputs should be interactive
        session_status = session.get("status")
        in_session = session_status in ["Running.", "Started."]
        can_speak = session_status == "Running." and result.status_text == "Your turn"

        # Both message input and end session button: enabled when in session AND it's user's turn
        message_input_update = gr.update(interactive=in_session and can_speak)
        end_session_interactive = gr.update(interactive=in_session and can_speak)

        return "", chat_html, session_info_md, header_md, typical_replies_html, message_input_update, end_session_interactive

    async def _gradio_admin_refresh_handler(self, participant_id: str):
        """Adapter for the admin refresh event."""
        result = await self.handle_admin_refresh(participant_id)
        if result.error:
            self._log_event(
                event_type="system",
                participant_id=participant_id,
                message=f"Admin Refresh failed: {result.error}",
                severity="error"
            )
            return f"Error: {result.error}"

        return self._render_admin_dashboard_to_html(result.dashboard_data)

    async def _gradio_end_session_handler(self, user_context: dict):
        """Adapter for the end session event."""
        result = await self.handle_end_session(user_context)
        if result.error:
            return result.error

        return result.message or "Session ended successfully"

    async def _gradio_pre_send(self, user_context: dict, delay_state: dict, message: str):
        """
        Immediately sets in_delay=True and captures pre-send state before any async operations.

        This prevents the race condition where the auto-refresh timer fires
        during the send API call and shows the AI response before the delay
        period starts.

        Also captures current session info (commitments) BEFORE the message is sent,
        so we can show pre-send commitments during delay (hiding AI's new commitments).

        The message is stored in delay_state so _gradio_send_handler can access it
        after the input is cleared.

        Returns: (delay_state, message_input, end_session_btn)
        """
        print(f"[PRE_SEND] Setting in_delay=True immediately, message length: {len(message) if message else 0}", flush=True)

        # Capture current session info before sending (to freeze commitments)
        pre_session_info_snapshot = ""
        if user_context and user_context.get("session_id"):
            try:
                result = await self.handle_refresh(user_context)
                if not result.error:
                    session = result.session_info
                    participant_id = user_context.get("participant_id")

                    # Determine participant role
                    if participant_id == session.get("parameters", {}).get("first_speaker"):
                        participant_role = "first_speaker"
                    elif participant_id == session.get("parameters", {}).get("second_speaker"):
                        participant_role = "second_speaker"
                    else:
                        participant_role = "observer"

                    # Capture current commitments (before AI responds)
                    pre_session_info_snapshot = self._render_session_info_to_markdown(
                        session,
                        result.first_speaker_commitments or [],
                        result.second_speaker_commitments or [],
                        participant_role
                    )
                    print(f"[PRE_SEND] Captured pre-send session info snapshot", flush=True)
            except Exception as e:
                print(f"[PRE_SEND] Failed to capture pre-send state: {e}", flush=True)

        new_delay_state = {
            "in_delay": True,
            "chat_snapshot": "",
            "typical_replies_snapshot": "",
            "session_info_snapshot": pre_session_info_snapshot,
            "pending_message": message  # Store message for _gradio_send_handler
        }

        # Clear and disable inputs immediately
        return new_delay_state, gr.update(value="", interactive=False), gr.update(interactive=False)

    async def _gradio_delay_before_refresh(self, user_context: dict, delay_state: dict):
        """
        Applies simulated delay only when opponent is AI (indicated by delay_state).

        This is called after _gradio_start_delay() which sets in_delay=True for AI
        opponents and in_delay=False for human opponents.

        Args:
            user_context: Current user session context with session_id and participant_id
            delay_state: State dict containing in_delay flag set by _gradio_start_delay()

        Returns:
            None
        """
        # Only delay if we're in delay mode (AI opponent)
        if not delay_state or not delay_state.get("in_delay", False):
            print(f"[DELAY] No delay needed (human opponent or not in delay mode)", flush=True)
            return

        print(f"[DELAY] Starting simulated model response sleep ({self.ai_response_delay}s)", flush=True)
        await asyncio.sleep(self.ai_response_delay)
        print(f"[DELAY] Simulated model response complete", flush=True)
        return

    async def _gradio_start_delay(self, user_context: dict, delay_state: dict):
        """
        Captures current chat state and sets delay flag if opponent is AI.

        For AI opponents: Sets in_delay=True and creates a snapshot showing only
        the user's messages (hiding the AI's instant response).

        For human opponents: Sets in_delay=False, letting normal polling handle
        the response when it arrives.

        This helps blind users to whether their opponent is human or AI.
        """
        print(f"[START_DELAY] Called with user_context: {user_context}", flush=True)

        session_id = user_context.get("session_id")
        participant_id = user_context.get("participant_id")

        if not session_id or not participant_id:
            print(f"[START_DELAY] Missing context, returning unchanged", flush=True)
            return delay_state

        # Wait a moment for backend to finish processing
        print(f"[START_DELAY] Waiting for backend to finish processing...", flush=True)
        await asyncio.sleep(0.5)

        # Get current session state
        result = await self.handle_refresh(user_context)
        if result.error:
            print(f"[START_DELAY] Refresh error: {result.error}", flush=True)
            return delay_state

        session = result.session_info

        # Check if opponent is AI
        opponent_is_ai = self._is_opponent_ai(session, participant_id)
        print(f"[START_DELAY] Opponent is AI: {opponent_is_ai}", flush=True)

        if not opponent_is_ai:
            # Human opponent - no delay needed, let normal polling handle it
            print(f"[START_DELAY] Human opponent, no delay needed", flush=True)
            return {"in_delay": False, "chat_snapshot": "", "typical_replies_snapshot": "", "session_info_snapshot": ""}

        # AI opponent - create snapshot hiding AI's response
        # Find messages up to and including user's last message (exclude AI response)
        chat_items = result.chat_items or []

        # Filter to show only messages up to user's last message
        # The AI response (if any) will be the last non-user message
        chat_items_snapshot = []
        for item in chat_items:
            chat_items_snapshot.append(item)
            if item.is_user:
                # After adding user's message, check if next would be AI response
                # We want to include user's message but not what comes after
                pass

        # Simpler approach: exclude the last message if it's from opponent (AI)
        if chat_items and not chat_items[-1].is_user:
            chat_items_snapshot = chat_items[:-1]
        else:
            chat_items_snapshot = chat_items

        print(f"[START_DELAY] Chat items: {len(chat_items)}, snapshot items: {len(chat_items_snapshot)}", flush=True)

        # Render chat snapshot
        chat_html = self._render_chat_to_html(chat_items_snapshot)

        # Use pre-captured session_info_snapshot from delay_state (captured in _gradio_pre_send)
        # This shows commitments BEFORE AI responded, hiding any new commitments from AI
        session_info_snapshot = delay_state.get("session_info_snapshot", "")
        print(f"[START_DELAY] Using pre-captured session_info_snapshot (length: {len(session_info_snapshot)})", flush=True)

        # Hide typical replies during delay (they're based on AI's response, revealing it exists)
        typical_replies_snapshot = ""

        print(f"[START_DELAY] Setting in_delay=True for AI opponent", flush=True)
        return {
            "in_delay": True,
            "chat_snapshot": chat_html,
            "typical_replies_snapshot": typical_replies_snapshot,
            "session_info_snapshot": session_info_snapshot
        }

    async def _gradio_end_delay_and_refresh(self, user_context: dict, delay_state: dict):
        """
        Clears delay flag and performs full refresh to show AI response.

        Returns: (delay_state, status_display, chat_display, session_info_display, topic_display,
                  typical_replies_display, message_input, end_session_btn)
        """
        # Clear delay flag and snapshots
        delay_state = {"in_delay": False, "chat_snapshot": "", "typical_replies_snapshot": "", "session_info_snapshot": ""}

        # Perform normal refresh
        if not user_context:
            return delay_state, "Enter a session code to join.", "", "", "", "", gr.update(interactive=False), gr.update(interactive=False)

        result = await self.handle_refresh(user_context)

        if result.error:
            self._log_event(
                event_type="system",
                participant_id=user_context.get("participant_id", "SYSTEM"),
                message=f"UI Refresh failed: {result.error}",
                session_id=user_context.get("session_id"),
                severity="error"
            )
            return delay_state, result.error, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        # Get participant role
        session = result.session_info
        participant_id = user_context.get("participant_id")
        if participant_id == session.get("parameters", {}).get("first_speaker"):
            participant_role = "first_speaker"
        elif participant_id == session.get("parameters", {}).get("second_speaker"):
            participant_role = "second_speaker"
        else:
            participant_role = "observer"

        # Render all UI components
        chat_html = self._render_chat_to_html(result.chat_items)
        session_info_md = self._render_session_info_to_markdown(
            result.session_info,
            result.first_speaker_commitments or [],
            result.second_speaker_commitments or [],
            participant_role
        )
        typical_replies_html = self._render_typical_replies_to_html(
            result.typical_replies or [],
            result.session_info,
            participant_role
        )
        topic = session.get("parameters", {}).get("debate_topic", "N/A")
        header_md = self._render_header_to_markdown(topic, result.status_text, session, participant_role, participant_id)

        # Determine input states
        session_status = session.get("status")
        in_session = session_status in ["Running.", "Started."]
        can_speak = session_status == "Running." and result.status_text == "Your turn"
        message_input_update = gr.update(interactive=in_session and can_speak)
        end_session_interactive = gr.update(interactive=in_session and can_speak)

        return delay_state, "", chat_html, session_info_md, header_md, typical_replies_html, message_input_update, end_session_interactive

    # --- INTERFACE IMPLEMENTATION ---

    def create_interface(self):
        """Implements the abstract method to build the Gradio UI."""
        with gr.Blocks(title="Persuasui") as demo:
            gr.Markdown("# 💬 Persuasui Debate Client")
            
            # State variables
            participant_id = gr.State()
            user_context = gr.State({})
            delay_state = gr.State({"in_delay": False, "chat_snapshot": "", "typical_replies_snapshot": "", "session_info_snapshot": ""})
            
            # Login section
            with gr.Column(visible=True) as login_section:
                login_code = gr.Textbox(label="Login Code", max_lines=1, submit_btn="Login")
                login_status = gr.Markdown()

            # Main debate interface
            with gr.Column(visible=False) as main_interface:
                with gr.Row():
                    with gr.Column(scale=1):
                        session_code = gr.Textbox(label="Session Code", max_lines=1, submit_btn="Join")
                        refresh_btn = gr.Button("Refresh", variant="secondary")
                        status_display = gr.Markdown("Enter a session code to join.")
                        session_info_display = gr.Markdown(label="Session Info")
                    with gr.Column(scale=2):
                        topic_display = gr.Markdown()
                        chat_display = gr.HTML()
                        typical_replies_display = gr.HTML()
                        message_input = gr.Textbox(label="Your Message", interactive=True, submit_btn="Send")
                        end_session_btn = gr.Button("End Session", variant="stop", interactive=False)
                        send_status = gr.Markdown()

            # Admin interface
            with gr.Column(visible=False) as admin_interface:
                gr.Markdown("## Admin Dashboard")
                admin_refresh_btn = gr.Button("Refresh Admin View")
                admin_display = gr.HTML() # Use HTML for admin dashboard

            # --- Event Wiring ---

            # Login
            login_code.submit(
                fn=self._gradio_login_handler,
                inputs=[login_code],
                outputs=[login_status, participant_id, login_code, main_interface, admin_interface]
            )

            # Join
            session_code.submit(
                fn=self._gradio_join_handler,
                inputs=[participant_id, session_code],
                outputs=[send_status, user_context]
            ).then(
                fn=self._gradio_refresh_handler,
                inputs=[user_context, delay_state],
                outputs=[status_display, chat_display, session_info_display, topic_display, typical_replies_display, message_input, end_session_btn]
            )

            # Send Message
            message_input.submit(
                fn=self._gradio_pre_send,
                inputs=[user_context, delay_state, message_input],
                outputs=[delay_state, message_input, end_session_btn]
            ).then(
                fn=self._gradio_send_handler,
                inputs=[user_context, delay_state],
                outputs=[]
            ).then(
                fn=self._gradio_start_delay,
                inputs=[user_context, delay_state],
                outputs=[delay_state],
                show_progress='hidden'
            ).then(
                fn=self._gradio_delay_before_refresh,
                inputs=[user_context, delay_state],
                outputs=[],
                show_progress='hidden'
            ).then(
                fn=self._gradio_end_delay_and_refresh,
                inputs=[user_context, delay_state],
                outputs=[delay_state, status_display, chat_display, session_info_display, topic_display, typical_replies_display, message_input, end_session_btn],
                show_progress='hidden'
            )

            # End Session
            end_session_btn.click(
                fn=self._gradio_end_session_handler,
                inputs=[user_context],
                outputs=[send_status]
            ).then(
                fn=self._gradio_refresh_handler,
                inputs=[user_context, delay_state],
                outputs=[status_display, chat_display, session_info_display, topic_display, typical_replies_display, message_input, end_session_btn]
            )

            # Refresh button
            refresh_btn.click(
                fn=self._gradio_refresh_handler,
                inputs=[user_context, delay_state],
                outputs=[status_display, chat_display, session_info_display, topic_display, typical_replies_display, message_input, end_session_btn]
            )

            # Auto-refresh timer
            gr.Timer(self.refresh_interval).tick(
                fn=self._gradio_refresh_handler,
                inputs=[user_context, delay_state],
                outputs=[status_display, chat_display, session_info_display, topic_display, typical_replies_display, message_input, end_session_btn]
            )

            # Admin refresh
            admin_refresh_btn.click(
                fn=self._gradio_admin_refresh_handler,
                inputs=[participant_id],
                outputs=[admin_display]
            )

        self.interface = demo

    def launch(self, **kwargs):
        """Implements the abstract method to launch the Gradio app."""
        if not self.interface:
            self.create_interface()
        
        launch_kwargs = {
            'server_name': self.server_name,
            'server_port': self.server_port,
            'show_error': True,
            'css': self.custom_css,
            'js': self.force_light_mode_js,
            **kwargs
        }
        if self.mode == 'dev':
            launch_kwargs['debug'] = True
            self._log_event(
                event_type="system",
                participant_id="SYSTEM",
                message="Running in DEV mode with hot-reloading enabled",
                severity="info"
            )

        self._log_event(
            event_type="system",
            participant_id="SYSTEM",
            message=f"Launching Gradio client on {self.server_name}:{self.server_port}",
            severity="info"
        )
        self.interface.launch(**launch_kwargs)
