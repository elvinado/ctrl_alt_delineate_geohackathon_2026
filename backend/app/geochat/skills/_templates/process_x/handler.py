"""Template for a single processing-operation skill."""
from typing import Any


def execute(*, project_id: str, parameters: dict[str, Any], db: Any) -> dict[str, Any]:
    """Validate and execute one named processing operation.

    Replace this template with calls to the existing backend processing services.
    Do not execute arbitrary user-provided Python source here.
    """
    raise NotImplementedError("Implement this processing skill using the approved backend service")
