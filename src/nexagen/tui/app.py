"""Interactive TUI application for the nexagen agent."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Input, Static
from textual.binding import Binding
from rich.text import Text

from nexagen.constants import DEFAULT_MODEL, DEFAULT_PERMISSION_MODE


class MessageDisplay(Static):
    """Displays a single message with role styling."""

    def __init__(self, role: str, content: str, is_error: bool = False):
        self.msg_role = role
        self.msg_content = content
        self.msg_is_error = is_error
        super().__init__()

    def compose(self) -> ComposeResult:
        pass

    def render(self) -> Text:
        if self.msg_role == "user":
            return Text.from_markup(f"[bold blue]You:[/bold blue] {self.msg_content}")
        elif self.msg_role == "assistant":
            return Text.from_markup(
                f"[bold green]Agent:[/bold green] {self.msg_content}"
            )
        elif self.msg_role == "tool":
            if self.msg_is_error:
                return Text.from_markup(
                    f"[bold red]Tool Error:[/bold red] {self.msg_content}"
                )
            return Text.from_markup(
                f"[dim]Tool Result:[/dim] {self.msg_content[:200]}"
            )
        elif self.msg_role == "status":
            return Text.from_markup(
                f"[bold yellow]\u280b[/bold yellow] {self.msg_content}"
            )
        return Text(self.msg_content)


class StepProgress(Static):
    """Shows a completed step with checkmark."""

    def __init__(self, step_num: int, summary: str, completed: bool = True):
        self.step_num = step_num
        self.summary = summary
        self.completed = completed
        super().__init__()

    def render(self) -> Text:
        icon = "\u2713" if self.completed else "\u280b"
        style = "green" if self.completed else "yellow"
        return Text.from_markup(
            f"[{style}]{icon}[/{style}] Step {self.step_num}: {self.summary}"
        )


class NexagenApp(App):
    """Interactive TUI for nexagen agent."""

    CSS = """
    #chat-log {
        height: 1fr;
        border: solid green;
        padding: 1;
    }
    #input-area {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    def __init__(
        self,
        provider: str = DEFAULT_MODEL,
        tools: list[str] | None = None,
        permission_mode: str = DEFAULT_PERMISSION_MODE,
    ):
        super().__init__()
        self.provider = provider
        self.tools = tools
        self.permission_mode = permission_mode
        self.step_count = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat-log")
        yield Static(
            f"Provider: {self.provider} | Mode: {self.permission_mode}",
            id="status-bar",
        )
        yield Input(placeholder="Type your message...", id="input-area")
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        input_widget = self.query_one("#input-area", Input)
        input_widget.value = ""

        chat_log = self.query_one("#chat-log", VerticalScroll)

        # Show user message
        chat_log.mount(MessageDisplay("user", user_text))

        # Show thinking indicator
        thinking = MessageDisplay("status", "Thinking...")
        chat_log.mount(thinking)

        if user_text.lower() in ("exit", "quit"):
            self.exit()
            return

        try:
            from nexagen.agent import Agent
            from nexagen.conversation import Conversation

            if not hasattr(self, "_agent"):
                self._agent = Agent(
                    provider=self.provider,
                    tools=self.tools,
                    permission_mode=self.permission_mode,
                )
                self._conversation = Conversation()

            # Remove thinking indicator
            thinking.remove()

            self.step_count = 0

            async for message in self._agent.run(user_text, self._conversation):
                if message.role == "assistant":
                    if message.tool_calls:
                        # Show step progress
                        self.step_count += 1
                        summary = message.summary or message.text or "Working..."
                        if len(summary) > 80:
                            summary = summary[:80] + "..."
                        chat_log.mount(StepProgress(self.step_count, summary))
                    elif message.text:
                        # Final response
                        chat_log.mount(MessageDisplay("assistant", message.text))
                elif message.role == "tool":
                    if message.is_error:
                        chat_log.mount(
                            MessageDisplay("tool", message.text or "", is_error=True)
                        )

            chat_log.scroll_end()

        except Exception as e:
            thinking.remove()
            chat_log.mount(MessageDisplay("tool", f"Error: {e}", is_error=True))

    def action_clear(self) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.remove_children()

    def action_quit(self) -> None:
        self.exit()


def run_tui(
    provider: str = DEFAULT_MODEL,
    tools: list[str] | None = None,
    permission_mode: str = DEFAULT_PERMISSION_MODE,
):
    """Launch the nexagen TUI."""
    app = NexagenApp(provider=provider, tools=tools, permission_mode=permission_mode)
    app.run()
