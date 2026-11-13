"""Deterministic interpretation of requests to prepare or run Process C (volume products, stages 10-11)."""
from __future__ import annotations

import re
from typing import Any

# How users name this process.
_PROCESS = re.compile(
    r"\bprocess[\s_-]*c\b|\bvolume[- ]products?\b|\bseg-?y export\b|"
    r"\bstages?\s*10\s*(?:-|–|to|and|&)\s*11\b|process_c_volume_products",
    re.IGNORECASE)
_RUN = re.compile(
    r"\b(?:run|rerun|re-run|start|launch|submit|execute|kick off|trigger|prepare|configure|set up|setup|plan|"
    r"ready|pre-?flight|requirements?|prerequisites?|needs?|requires?|required|inputs?|can (?:i|we))\b")
_DESCRIBE = re.compile(r"\b(?:what is|what's|what does|describe|how does|what happens|what will|purpose of|about)\b")
_RESULT_NOUN = re.compile(r"\b(results?|outputs?|outcomes?|findings?|verdict|scores?)\b")
_RESULT_VERB = re.compile(r"\b(explain|analy[sz]e|interpret|summari[sz]e|review|assess|evaluate)\b")
_JOB_ID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE)

# The upstream job this process builds on (a completed Process B run).
SOURCE_PARAMETER = "source_inference_job_id"
SOURCE_PYTHON_SCRIPT = "process_B_inference.py"


def parse_request(message: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Return handler parameters for a request to prepare, check or describe this process, else ``None``.

    Two keys starting with an underscore are instructions for the agent, not handler parameters:
    ``_source_job`` asks it to fill in the latest completed upstream job, and ``_project_datasets``
    asks it to pass the active project's registered datasets.
    """
    text = re.sub(r"\s+", " ", (message or "").casefold()).strip()
    if not text or not _PROCESS.search(text):
        return None
    wants_run = bool(_RUN.search(text))
    if _RESULT_NOUN.search(text) and not re.search(r"\b(?:run|rerun|re-run|start|launch|submit|execute)\b", text):
        return None  # reading results belongs to the output-analysis skill
    if _RESULT_VERB.search(text) and not wants_run:
        return None
    if not (wants_run or _DESCRIBE.search(text)):
        return None
    parameters: dict[str, Any] = {}
    job = _JOB_ID.search(message or "")
    if job:
        parameters[SOURCE_PARAMETER] = job.group(0).lower()
    else:
        parameters["_source_job"] = {"parameter": SOURCE_PARAMETER, "python_script": SOURCE_PYTHON_SCRIPT}
    return parameters
