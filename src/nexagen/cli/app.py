from __future__ import annotations
import asyncio
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from nexagen.constants import DEFAULT_MODEL, DEFAULT_PERMISSION_MODE


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="nexagen")
def main():
    """nexagen — Universal LLM Agent SDK"""
    pass


@main.command()
@click.argument("prompt")
@click.option("--provider", "-p", default=DEFAULT_MODEL, help="LLM provider (e.g., ollama/qwen3)")
@click.option("--tools", "-t", default=None, help="Comma-separated tool names (e.g., file_read,bash)")
@click.option("--permission-mode", "-m", default=DEFAULT_PERMISSION_MODE, help="Permission mode: readonly, safe, full")
@click.option("--system-prompt", "-s", default=None, help="Custom system prompt")
def run(prompt: str, provider: str, tools: str | None, permission_mode: str, system_prompt: str | None):
    """Run a single prompt through the agent."""
    tool_list = tools.split(",") if tools else None

    async def _run():
        from nexagen.agent import Agent

        try:
            agent = Agent(
                provider=provider,
                tools=tool_list,
                system_prompt=system_prompt,
                permission_mode=permission_mode,
            )
        except (ValueError, ImportError) as e:
            console.print(f"[bold red]Error creating agent:[/bold red] {e}")
            return

        try:
            async for message in agent.run(prompt):
                if message.role == "assistant" and message.text:
                    console.print(Panel(message.text, title="Agent", border_style="green"))
                elif message.role == "tool":
                    style = "red" if message.is_error else "dim"
                    label = "Tool Error" if message.is_error else "Tool Result"
                    output = message.text or ""
                    if len(output) > 500:
                        output = output[:500] + "\n..."
                    console.print(f"[{style}]{label}:[/{style}] {output}")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Agent error:[/bold red] {e}")

    asyncio.run(_run())


@main.command()
@click.option("--provider", "-p", default=DEFAULT_MODEL, help="LLM provider")
@click.option("--tools", "-t", default=None, help="Comma-separated tool names")
@click.option("--permission-mode", "-m", default=DEFAULT_PERMISSION_MODE, help="Permission mode")
def chat(provider: str, tools: str | None, permission_mode: str):
    """Interactive chat with the agent."""
    tool_list = tools.split(",") if tools else None

    console.print(Panel(
        f"[bold]nexagen[/bold] interactive chat\n"
        f"Provider: {provider}\n"
        f"Tools: {tools or 'none'}\n"
        f"Mode: {permission_mode}\n\n"
        f"Type 'exit' or 'quit' to end.",
        title="nexagen",
        border_style="blue",
    ))

    async def _chat():
        from nexagen.agent import Agent
        from nexagen.conversation import Conversation

        try:
            agent = Agent(
                provider=provider,
                tools=tool_list,
                permission_mode=permission_mode,
            )
        except (ValueError, ImportError) as e:
            console.print(f"[bold red]Error creating agent:[/bold red] {e}")
            return

        conv = Conversation()

        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ")
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye!")
                break

            if user_input.strip().lower() in ("exit", "quit"):
                console.print("Goodbye!")
                break

            if not user_input.strip():
                continue

            try:
                async for message in agent.run(user_input, conversation=conv):
                    if message.role == "assistant" and message.text:
                        console.print(f"[bold green]Agent:[/bold green] {message.text}")
                    elif message.role == "tool":
                        if message.is_error:
                            console.print(f"[red]Error:[/red] {message.text}")
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                console.print("[dim]You can try again or type 'exit' to quit.[/dim]")

    asyncio.run(_chat())


@main.command()
@click.option("--provider", "-p", default=DEFAULT_MODEL, help="LLM provider (e.g., openai_compat/gpt-4o-mini)")
@click.option("--base-url", "-b", default=None, help="Custom base URL (e.g., http://127.0.0.1:8081)")
@click.option("--tools", "-t", default=None, help="Comma-separated tool names (e.g., file_read,bash,grep)")
@click.option("--permission-mode", "-m", default=DEFAULT_PERMISSION_MODE, help="Permission mode: readonly, safe, full")
def tui(provider: str, base_url: str | None, tools: str | None, permission_mode: str):
    """Launch interactive TUI with step-by-step progress display."""
    tool_list = tools.split(",") if tools else None

    if base_url:
        from nexagen.models import ProviderConfig
        config = ProviderConfig.from_string(provider)
        config.base_url = base_url
        from nexagen.tui.app import NexagenApp
        app = NexagenApp(provider=config, tools=tool_list, permission_mode=permission_mode)
        app.run()
    else:
        from nexagen.tui.app import run_tui
        run_tui(provider=provider, tools=tool_list, permission_mode=permission_mode)


if __name__ == "__main__":
    main()
