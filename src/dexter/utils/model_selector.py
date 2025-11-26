import os

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from typing import Optional

from dexter.model import MODEL_PROVIDER


# Model definitions with display names and actual model identifiers
MODELS = [
    {
        "display_name": f"{os.getenv('LLM_API_OPENAI_MODEL', 'gpt-5-mini')} / {os.getenv('LLM_API_OPENAI_STRONG_MODEL', 'gpt-5-pro')}",
        "model_provider": "openai",
        "description": "OpenAI",
    },
    {
        "display_name": f"{os.getenv('LLM_API_ANTHROPIC_MODEL', 'claude-haiku-4-5')} / {os.getenv('LLM_API_ANTHROPIC_STRONG_MODEL', 'claude-sonnet-4-5')}",
        "model_provider": "anthropic",
        "description": "Anthropic",
    },
    {
        "display_name": f"{os.getenv('LLM_API_GOOGLE_MODEL', 'gemini-3')} / {os.getenv('LLM_API_GOOGLE_STRONG_MODEL', 'gemini-3-pro')}",
        "model_provider": "google",
        "description": "Google",
    },
]


def select_model_provider(
    current_model_provider: Optional[MODEL_PROVIDER] = None,
) -> Optional[str]:
    """
    Display an interactive model provider selector UI similar to the screenshot.

    Args:
        current_model_provider: The currently selected model provider (if any)

    Returns:
        The selected model provider, or None if cancelled
    """
    selected_index = 0
    if current_model_provider:
        # Find the index of the current model
        for i, model in enumerate(MODELS):
            if model["model_provider"] == current_model_provider:
                selected_index = i
                break

    # Create key bindings
    kb = KeyBindings()

    def get_formatted_text():
        """Generate the formatted text for the menu."""
        fragments = []

        # Title
        fragments.append(("class:title", "Select model provider\n"))
        fragments.append(
            (
                "class:subtitle",
                "Switch between LLM model providers. Applies to this session and future sessions.\n",
            )
        )
        fragments.append(("", "\n"))

        # Model options
        for i, model in enumerate(MODELS):
            prefix = "> " if i == selected_index else "  "
            # Show checkmark if this is the currently selected model
            checkmark = (
                " ✓"
                if current_model_provider
                and current_model_provider == model["model_provider"]
                else ""
            )

            # Number
            number = f"{i + 1}.  "

            # Model provider name
            name = model["display_name"]

            # Description
            desc = f" · {model['description']}"

            # Style based on selection
            if i == selected_index:
                fragments.append(
                    (
                        "class:model.selected",
                        f"{prefix}{number}{name}{desc}{checkmark}\n",
                    )
                )
            else:
                fragments.append(
                    ("class:model", f"{prefix}{number}{name}{desc}{checkmark}\n")
                )

        fragments.append(("", "\n"))
        fragments.append(("class:footer", "Enter to confirm · Esc to exit"))

        return FormattedText(fragments)

    # Create formatted text control with a callable that captures current state
    formatted_text_control = FormattedTextControl(get_formatted_text, show_cursor=False)
    text_window = Window(
        content=formatted_text_control,
        wrap_lines=False,
    )

    layout = Layout(HSplit([text_window]))

    # Style matching the screenshot
    style = Style.from_dict(
        {
            "title": "#58A6FF bold",  # Light blue
            "subtitle": "#888888",  # Grey
            "model": "#58A6FF",  # Light blue
            "model.selected": "#a5cfff bold",  # Lighter blue bold
            "footer": "#888888",  # Grey
        }
    )

    # Key bindings
    @kb.add("up")
    @kb.add("k")
    def move_up(event):
        nonlocal selected_index
        selected_index = max(0, selected_index - 1)
        # The callable will be called again on next render with updated selected_index
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def move_down(event):
        nonlocal selected_index
        selected_index = min(len(MODELS) - 1, selected_index + 1)
        # The callable will be called again on next render with updated selected_index
        event.app.invalidate()

    @kb.add("enter")
    def confirm(event):
        event.app.exit(result=MODELS[selected_index]["model_provider"])

    @kb.add("escape")
    def cancel(event):
        event.app.exit(result=None)

    # Create and run the application
    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
    )

    result = app.run()
    return result
