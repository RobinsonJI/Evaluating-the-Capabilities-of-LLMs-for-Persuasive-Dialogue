import flet as ft
from flet import Colors
from datetime import datetime
import asyncio
from typing import List, Dict

from client.client import BasePersuasuiClient
from client.models import ChatItem


class FletPersuasuiClient(BasePersuasuiClient):
    """
    The Flet-specific implementation of the Persuasui client.

    This class inherits the core application logic from BasePersuasuiClient
    and implements the methods required to build and run a Flet user interface.
    """
    def __init__(self, **kwargs):
        """
        Initialises the Flet client.

        It calls the base class constructor and then initializes Flet-specific
        attributes.

        Note: Per-session state (participant_id, user_context, UI controls)
        is stored in page.session, not as instance variables, to support
        multiple simultaneous users.
        """
        super().__init__(**kwargs)

    # --- FLET-SPECIFIC RENDERING HELPERS ---

    def _render_chat_to_controls(self, chat_items: List[ChatItem]) -> List[ft.Control]:
        """Takes a list of ChatItem models and returns Flet controls for the chat display."""
        if not chat_items:
            return [ft.Container(
                content=ft.Text(
                    "No messages yet. Join a session to start!",
                    color=Colors.GREY_600,
                    italic=True
                ),
                padding=20,
                alignment=ft.alignment.center
            )]

        controls = []
        for item in chat_items:
            try:
                time_str = datetime.fromisoformat(item.timestamp).strftime('%H:%M:%S')
            except (ValueError, TypeError):
                time_str = ""

            # Determine alignment and color based on who sent the message
            if item.is_user:
                bgcolor = Colors.BLUE_100
                alignment = ft.alignment.center_right
            else:
                bgcolor = Colors.GREY_200
                alignment = ft.alignment.center_left

            message_container = ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text(
                            item.speaker_id,
                            weight=ft.FontWeight.BOLD,
                            size=12
                        ),
                        ft.Text(
                            f"Turn {item.turn_number}",
                            size=10,
                            color=Colors.GREY_600
                        )
                    ], spacing=10),
                    ft.Text(
                        item.content,
                        size=14,
                        selectable=True
                    ),
                    ft.Text(
                        time_str,
                        size=10,
                        color=Colors.GREY_500,
                        italic=True
                    )
                ], spacing=5, tight=True),
                bgcolor=bgcolor,
                border_radius=10,
                padding=10,
                margin=ft.margin.only(bottom=10),
                alignment=alignment
            )

            controls.append(message_container)

        return controls

    def _render_session_info_to_text(self, session_info: dict, participant_id: str) -> str:
        """Takes session data and returns a formatted text string for the session info panel."""
        params = session_info.get("parameters", {})
        p1_id = params.get("first_speaker")
        p2_id = params.get("second_speaker")

        opponent_id = p2_id if participant_id == p1_id else p1_id

        # Determine opponent type based on ID prefix, default to Human
        opponent_type = "AI" if opponent_id and opponent_id.startswith("AI") else "Human"

        both_joined = "Ready" if session_info.get("participant1_joined") and session_info.get("participant2_joined") else "Waiting"

        if participant_id == p1_id:
            stance = "FOR"
        else:
            stance = "AGAINST"

        return f"""Session: {session_info.get('session_id')}
Topic: {params.get('debate_topic', 'N/A')}
Your Role: {participant_id} ({stance})
Opponent: {opponent_id} ({opponent_type})
Status: {session_info.get('status')} ({both_joined})
Turn: {session_info.get('current_turn', 'N/A')}
Messages: {len(session_info.get('dialogue_history', []))}"""

    def _render_admin_dashboard_to_controls(self, dashboard_data: dict) -> List[ft.Control]:
        """Takes admin dashboard data and generates Flet controls for the monitoring panel."""
        controls = []

        # Recent Activity Section
        activity_column = ft.Column([], spacing=5, scroll=ft.ScrollMode.AUTO, height=300)
        recent_activities = dashboard_data.get('recent_activities', [])

        if not recent_activities:
            activity_column.controls.append(ft.Text("No recent activity", color=Colors.GREY_600))
        else:
            for activity in reversed(recent_activities[-15:]):
                session_id_info = f" (Session: {activity.get('session_id')})" if activity.get('session_id') else ''
                activity_column.controls.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Text(activity.get('timestamp', ''), size=10, color=Colors.GREY_500),
                            ft.Text(
                                f"{activity.get('event_type', '').upper()}",
                                weight=ft.FontWeight.BOLD,
                                size=11
                            ),
                            ft.Text(
                                f"{activity.get('participant_id', '')}: {activity.get('message', '')}{session_id_info}",
                                size=12
                            )
                        ], spacing=2, tight=True),
                        bgcolor=Colors.GREY_100,
                        border_radius=5,
                        padding=8,
                        margin=ft.margin.only(bottom=5)
                    )
                )

        controls.append(ft.Container(
            content=ft.Column([
                ft.Text("Recent Activity", size=18, weight=ft.FontWeight.BOLD),
                activity_column
            ]),
            bgcolor=Colors.WHITE,
            border_radius=10,
            padding=15,
            margin=ft.margin.only(bottom=10)
        ))

        # Participants Section
        all_participants = dashboard_data.get('participants', [])
        human_participants = [p for p in all_participants if not p.get('is_ai', False)]
        ai_participants = [p for p in all_participants if p.get('is_ai', False)]

        # Human Participants
        human_column = ft.Column([], spacing=5, scroll=ft.ScrollMode.AUTO, height=200)
        if not human_participants:
            human_column.controls.append(ft.Text("No human participants found", color=Colors.GREY_600))
        else:
            for p in human_participants:
                status_label = "Offline"
                status_color = Colors.GREY_400
                if p.get('is_authenticated'):
                    if p.get('current_session'):
                        status_label = "Active"
                        status_color = Colors.GREEN
                    else:
                        status_label = "Online"
                        status_color = Colors.BLUE

                human_column.controls.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Text(p.get('id'), weight=ft.FontWeight.BOLD, size=14),
                                ft.Container(
                                    content=ft.Text(status_label, size=10, color=Colors.WHITE),
                                    bgcolor=status_color,
                                    border_radius=5,
                                    padding=ft.padding.symmetric(horizontal=8, vertical=2)
                                )
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            ft.Text(f"Type: Human", size=11),
                            ft.Text(f"Session: {p.get('current_session', 'None')}", size=11),
                            ft.Text(f"Auth Code: {p.get('auth_code', 'N/A')}", size=11)
                        ], spacing=3, tight=True),
                        bgcolor=Colors.GREY_100,
                        border_radius=5,
                        padding=10,
                        margin=ft.margin.only(bottom=5)
                    )
                )

        # AI Models
        ai_column = ft.Column([], spacing=5, scroll=ft.ScrollMode.AUTO, height=200)
        if not ai_participants:
            ai_column.controls.append(ft.Text("No AI models found", color=Colors.GREY_600))
        else:
            for p in ai_participants:
                status_label = "Active" if p.get('current_session') else "Online"
                status_color = Colors.GREEN if p.get('current_session') else Colors.BLUE

                ai_column.controls.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Text(p.get('id'), weight=ft.FontWeight.BOLD, size=14),
                                ft.Container(
                                    content=ft.Text(status_label, size=10, color=Colors.WHITE),
                                    bgcolor=status_color,
                                    border_radius=5,
                                    padding=ft.padding.symmetric(horizontal=8, vertical=2)
                                )
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            ft.Text(f"Type: AI Model", size=11),
                            ft.Text(f"Session: {p.get('current_session', 'Available')}", size=11),
                            ft.Text(f"Messages: {p.get('message_count', 0)}", size=11)
                        ], spacing=3, tight=True),
                        bgcolor=Colors.GREY_100,
                        border_radius=5,
                        padding=10,
                        margin=ft.margin.only(bottom=5)
                    )
                )

        participants_row = ft.Row([
            ft.Container(
                content=ft.Column([
                    ft.Text("Human Participants", size=16, weight=ft.FontWeight.BOLD),
                    human_column
                ]),
                bgcolor=Colors.WHITE,
                border_radius=10,
                padding=15,
                expand=1
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("AI Models", size=16, weight=ft.FontWeight.BOLD),
                    ai_column
                ]),
                bgcolor=Colors.WHITE,
                border_radius=10,
                padding=15,
                expand=1
            )
        ], spacing=10)

        controls.append(participants_row)

        # All Sessions
        sessions_column = ft.Column([], spacing=5, scroll=ft.ScrollMode.AUTO, height=300)
        sessions = dashboard_data.get('sessions', [])

        if not sessions:
            sessions_column.controls.append(ft.Text("No sessions found", color=Colors.GREY_600))
        else:
            for s in sessions:
                p1 = s.get('participants', {}).get('participant1', {})
                p2 = s.get('participants', {}).get('participant2', {})
                status = s.get('status', 'unknown').lower().replace('.', '')

                status_colors = {
                    'started': Colors.BLUE,
                    'running': Colors.GREEN,
                    'finished': Colors.GREY_500
                }
                status_color = status_colors.get(status, Colors.GREY_400)

                sessions_column.controls.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Text(f"Session {s.get('session_id')}", weight=ft.FontWeight.BOLD, size=14),
                                ft.Container(
                                    content=ft.Text(status.upper(), size=10, color=Colors.WHITE),
                                    bgcolor=status_color,
                                    border_radius=5,
                                    padding=ft.padding.symmetric(horizontal=8, vertical=2)
                                )
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            ft.Text(f"Topic: {s.get('topic')}", size=12),
                            ft.Text("Participants:", size=11, weight=ft.FontWeight.BOLD),
                            ft.Text(
                                f"{p1.get('id')} ({'AI' if p1.get('is_ai') else 'Human'}) - {'Joined' if p1.get('joined') else 'Not joined'}",
                                size=11
                            ),
                            ft.Text(
                                f"{p2.get('id')} ({'AI' if p2.get('is_ai') else 'Human'}) - {'Joined' if p2.get('joined') else 'Not joined'}",
                                size=11
                            )
                        ], spacing=3, tight=True),
                        bgcolor=Colors.GREY_100,
                        border_radius=5,
                        padding=10,
                        margin=ft.margin.only(bottom=5)
                    )
                )

        controls.append(ft.Container(
            content=ft.Column([
                ft.Text("All Sessions", size=18, weight=ft.FontWeight.BOLD),
                sessions_column
            ]),
            bgcolor=Colors.WHITE,
            border_radius=10,
            padding=15
        ))

        # Timestamp
        controls.append(
            ft.Text(
                f"Last updated: {dashboard_data.get('timestamp', '')[:19].replace('T', ' ')}",
                size=10,
                color=Colors.GREY_500,
                text_align=ft.TextAlign.CENTER
            )
        )

        return controls

    # --- FLET-SPECIFIC EVENT HANDLERS (ADAPTERS) ---

    async def _flet_login_handler(self, e, page):
        """Adapter for the login event."""
        controls = page.session.get("controls")
        code = controls["login_code_field"].value
        self.logger.info(f"Login attempt with code: '{code[:2] if code else ''}...'")
        result = await self.handle_login(code)

        if result.error:
            self.logger.warning(f"Login failed: {result.error}")
            controls["login_status"].value = result.error
            controls["login_status"].color = Colors.RED
        else:
            self.logger.info(f"Participant '{result.participant_id}' logged in successfully.")
            page.session.set("participant_id", result.participant_id)

            if result.is_admin:
                controls["login_status"].value = "Admin Access Granted"
                controls["login_status"].color = Colors.GREEN
                controls["login_section"].visible = False
                controls["main_interface"].visible = False
                controls["admin_interface"].visible = True
                # Trigger initial admin refresh
                await self._flet_admin_refresh_handler(None, page)
            else:
                controls["login_status"].value = result.message
                controls["login_status"].color = Colors.GREEN
                controls["login_section"].visible = False
                controls["main_interface"].visible = True
                controls["admin_interface"].visible = False

        page.update()

    async def _flet_join_handler(self, e, page):
        """Adapter for the join session event."""
        controls = page.session.get("controls")
        participant_id = page.session.get("participant_id")
        session_code = controls["session_code_field"].value

        self.logger.info(f"Participant '{participant_id}' attempting to join session '{session_code}'.")
        result = await self.handle_join_session(participant_id, session_code)

        if result.error:
            self.logger.warning(f"Join session failed for '{participant_id}': {result.error}")
            controls["send_status"].value = result.error
            controls["send_status"].color = Colors.RED
        else:
            self.logger.info(f"Participant '{participant_id}' joined session '{session_code}' successfully.")
            page.session.set("user_context", result.user_context)
            controls["send_status"].value = result.message
            controls["send_status"].color = Colors.GREEN

            # Trigger a refresh after joining
            await self._flet_refresh_handler(None, page)

        page.update()

    async def _flet_send_handler(self, e, page):
        """Adapter for the send message event."""
        controls = page.session.get("controls")
        user_context = page.session.get("user_context", {})
        message = controls["message_input"].value
        participant_id = user_context.get("participant_id", "Unknown")

        self.logger.info(f"Participant '{participant_id}' sending message: '{message[:30] if message else ''}...'")
        result = await self.handle_send_message(user_context, message)

        if result.error:
            self.logger.warning(f"Send message failed for '{participant_id}': {result.error}")
            controls["send_status"].value = result.error
            controls["send_status"].color = Colors.RED
        else:
            self.logger.info(f"Participant '{participant_id}' sent message successfully.")
            controls["send_status"].value = result.message
            controls["send_status"].color = Colors.GREEN
            controls["message_input"].value = ""  # Clear input box on success

            # Trigger a refresh after sending
            await self._flet_refresh_handler(None, page)

        page.update()

    async def _flet_refresh_handler(self, e, page):
        """Adapter for the UI refresh event."""
        controls = page.session.get("controls")
        user_context = page.session.get("user_context", {})

        if not user_context:
            controls["status_display"].value = "Enter a session code to join."
            page.update()
            return

        # Use the 'since' parameter for efficient polling
        last_poll_time = page.session.get("last_poll_time")
        result = await self.handle_refresh(user_context, since=last_poll_time)
        page.session.set("last_poll_time", datetime.now().isoformat())

        if result.error:
            self.logger.error(f"UI Refresh failed: {result.error}")
            controls["status_display"].value = result.error
            controls["status_display"].color = Colors.RED
            page.update()
            return

        if not result.changed:
            # If nothing changed, don't update the UI to prevent flicker
            return

        # Update chat display
        chat_controls = self._render_chat_to_controls(result.chat_items)
        controls["chat_display"].controls = chat_controls

        # Auto-scroll to bottom after updating chat
        if chat_controls:
            controls["chat_display"].scroll_to(offset=-1, duration=100)

        # Update session info
        session_info_text = self._render_session_info_to_text(
            result.session_info,
            user_context["participant_id"]
        )
        controls["session_info_display"].value = session_info_text

        # Update status and topic
        controls["status_display"].value = result.status_text
        controls["status_display"].color = Colors.BLUE
        controls["topic_display"].value = result.topic_and_stance

        page.update()

    async def _flet_admin_refresh_handler(self, e, page):
        """Adapter for the admin refresh event."""
        controls = page.session.get("controls")
        participant_id = page.session.get("participant_id")

        result = await self.handle_admin_refresh(participant_id)

        if result.error:
            self.logger.error(f"Admin Refresh failed: {result.error}")
            controls["admin_display"].controls = [ft.Text(f"Error: {result.error}", color=Colors.RED)]
        else:
            self.logger.info("Admin dashboard refreshed.")
            admin_controls = self._render_admin_dashboard_to_controls(result.dashboard_data)
            controls["admin_display"].controls = admin_controls

        page.update()

    # --- AUTO-REFRESH TASK ---

    async def _auto_refresh_task(self, page):
        """Background task for auto-refreshing the UI."""
        while True:
            await asyncio.sleep(self.refresh_interval)

            controls = page.session.get("controls")
            user_context = page.session.get("user_context", {})

            # Refresh main interface if user is in a session
            if user_context and controls["main_interface"].visible:
                await self._flet_refresh_handler(None, page)

            # Refresh admin dashboard if admin interface is visible
            if controls["admin_interface"].visible:
                await self._flet_admin_refresh_handler(None, page)

    # --- INTERFACE IMPLEMENTATION ---

    def create_interface(self):
        """Implements the abstract method to build the Flet UI."""

        def build_ui(page: ft.Page):
            page.title = "Persuasui Debate Client"
            page.theme_mode = ft.ThemeMode.LIGHT
            page.padding = 20
            page.scroll = ft.ScrollMode.AUTO

            # Initialize per-session state
            page.session.set("participant_id", None)
            page.session.set("user_context", {})
            page.session.set("last_poll_time", None)
            page.session.set("controls", {})

            # Title
            title = ft.Text("💬 Persuasui Debate Client", size=32, weight=ft.FontWeight.BOLD)

            # --- LOGIN SECTION ---
            login_code_field = ft.TextField(
                label="Login Code",
                autofocus=True,
                expand=True
            )
            login_btn = ft.ElevatedButton(
                "Login",
                on_click=lambda e: asyncio.create_task(self._flet_login_handler(e, page))
            )
            login_status = ft.Text("", size=14)

            login_section = ft.Column(
                [
                    ft.Container(
                        content=ft.Row([
                            login_code_field,
                            login_btn
                        ], spacing=10),
                        width=400
                    ),
                    login_status
                ],
                spacing=10,
                visible=True
            )

            # --- MAIN INTERFACE ---
            session_code_field = ft.TextField(
                label="Session Code",
                expand=True
            )
            join_btn = ft.ElevatedButton(
                "Join Session",
                on_click=lambda e: asyncio.create_task(self._flet_join_handler(e, page))
            )
            status_display = ft.Text(
                "Enter a session code to join.",
                size=14,
                color=Colors.BLUE
            )
            session_info_display = ft.Text(
                "",
                size=12,
                selectable=True
            )

            left_column = ft.Column(
                [
                    ft.Row([
                        session_code_field,
                        join_btn
                    ], spacing=10),
                    ft.Divider(),
                    status_display,
                    ft.Divider(),
                    session_info_display
                ],
                spacing=10,
                expand=1
            )

            topic_display = ft.Text("", size=14, weight=ft.FontWeight.BOLD)
            chat_display = ft.Column(
                [],
                spacing=5,
                scroll=ft.ScrollMode.AUTO,
                height=400,
                expand=True
            )
            message_input = ft.TextField(
                label="Your Message",
                multiline=True,
                min_lines=2,
                max_lines=5,
                expand=True,
                on_submit=lambda e: asyncio.create_task(self._flet_send_handler(e, page))
            )
            send_btn = ft.ElevatedButton(
                "Send",
                on_click=lambda e: asyncio.create_task(self._flet_send_handler(e, page))
            )
            send_status = ft.Text("", size=12)

            right_column = ft.Column(
                [
                    topic_display,
                    ft.Container(
                        content=chat_display,
                        bgcolor=Colors.GREY_50,
                        border_radius=10,
                        padding=10,
                        expand=True
                    ),
                    ft.Row([
                        message_input,
                        send_btn
                    ], spacing=10),
                    send_status
                ],
                spacing=10,
                expand=2
            )

            main_interface = ft.Row(
                [left_column, right_column],
                spacing=20,
                visible=False,
                vertical_alignment=ft.CrossAxisAlignment.START,
                expand=True
            )

            # --- ADMIN INTERFACE ---
            admin_refresh_btn = ft.ElevatedButton(
                "Refresh Admin View",
                on_click=lambda e: asyncio.create_task(self._flet_admin_refresh_handler(e, page))
            )
            admin_display = ft.Column(
                [],
                spacing=10,
                scroll=ft.ScrollMode.AUTO,
                expand=True
            )

            admin_interface = ft.Column(
                [
                    ft.Text("Admin Dashboard", size=24, weight=ft.FontWeight.BOLD),
                    admin_refresh_btn,
                    ft.Divider(),
                    admin_display
                ],
                spacing=10,
                visible=False,
                expand=True
            )

            # Store control references in session
            page.session.set("controls", {
                "login_code_field": login_code_field,
                "login_status": login_status,
                "login_section": login_section,
                "session_code_field": session_code_field,
                "status_display": status_display,
                "session_info_display": session_info_display,
                "topic_display": topic_display,
                "chat_display": chat_display,
                "message_input": message_input,
                "send_status": send_status,
                "main_interface": main_interface,
                "admin_display": admin_display,
                "admin_interface": admin_interface
            })

            # Add all sections to page
            page.add(
                title,
                ft.Divider(),
                login_section,
                main_interface,
                admin_interface
            )

            # Start auto-refresh task
            async def auto_refresh():
                await self._auto_refresh_task(page)

            page.run_task(auto_refresh)

        return build_ui

    def launch(self, **kwargs):
        """Implements the abstract method to launch the Flet app."""
        if self.mode == 'dev':
            self.logger.info("Running in DEV mode (note: Flet doesn't support hot-reload when run as Python module).")

        self.logger.info(f"Launching Flet client on port {self.server_port}")
        self.logger.info(f"Navigate to: http://localhost:{self.server_port}")

        ft.app(
            target=self.create_interface(),
            port=self.server_port,
            view=None,  # Don't auto-open browser
            **kwargs
        )
