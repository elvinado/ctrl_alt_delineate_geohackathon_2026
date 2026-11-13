"""Template for recognising requests to prepare or run one process."""
from typing import Any


def parse_request(message: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Return handler parameters when the user asks to prepare, check or describe this process, else ``None``."""
    return None
