"""Template for recognising questions about one process's results."""
from typing import Any


def parse_request(message: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Return handler parameters (for example ``focus``) when the question is about this process, else ``None``.

    Accept a question that names the process, or a follow-up to this skill's own report. Leave
    ``job_id`` out unless the user wrote one: the agent then reads the most recent run.
    """
    return None
