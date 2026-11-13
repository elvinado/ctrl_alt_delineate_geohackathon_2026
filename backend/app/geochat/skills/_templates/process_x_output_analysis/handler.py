"""Template for analysing outputs of one processing operation."""
from typing import Any


def execute(*, project_id: str, parameters: dict[str, Any], db: Any) -> dict[str, Any]:
    """Retrieve and prepare outputs for analysis by the GeoChat agent.

    Replace this template with calls to approved job and dataset services.
    Keep numerical/file inspection deterministic where possible; let the LLM
    explain the returned evidence rather than fabricate measurements.
    """
    raise NotImplementedError("Implement output analysis using approved backend services")
