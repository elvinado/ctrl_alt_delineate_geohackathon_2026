"""Template for the fixed Markdown layout of one process pre-flight check."""
from typing import Any

TITLE = "<PROCESS_NAME>"
PREFLIGHT_MARKER = TITLE + " — pre-flight check"


def render_markdown(result: dict[str, Any]) -> str:
    """Lay the validation result out: readiness, issues and fixes, inputs, plan, next step."""
    raise NotImplementedError("Render the handler result as Markdown")
