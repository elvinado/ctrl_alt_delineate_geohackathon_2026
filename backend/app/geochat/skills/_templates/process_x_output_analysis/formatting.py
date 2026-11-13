"""Template for the fixed Markdown layout of one process output analysis."""
from typing import Any

TITLE = "<PROCESS_NAME>"
# First line of every rendered answer. ``intent.py`` recognises follow-up questions by it.
RESULT_MARKER = TITLE + " — results"


def render_markdown(result: dict[str, Any]) -> str:
    """Lay the handler's evidence out in one stable order.

    Keep the order fixed (title, verdict, short reading, headline table, findings, next steps,
    evidence read). Print the handler's numbers unchanged and show identifiers as `code`.
    """
    raise NotImplementedError("Render the handler result as Markdown")
