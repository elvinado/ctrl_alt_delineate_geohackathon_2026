"""Deterministic interpretation of questions about Process C (volume products, stages 10-11) results."""
from __future__ import annotations

import re
from typing import Any

from app.geochat.skills.process_c_volume_products_output_analysis.formatting import RESULT_MARKER

# How users name this process.
_PROCESS = re.compile(
    r"\bprocess[\s_-]*c\b|\bvolume[- ]products?\b|\bseg-?y export\b|"
    r"\bstages?\s*10\s*(?:-|–|to|and|&)\s*11\b|process_c_volume_products",
    re.IGNORECASE)
_PROCESS_LETTER = "c"
_ANY_PROCESS_LETTER = re.compile(r"\bprocess[\s_-]*([a-f])\b(?!\s+(?:new|seismic|well|dataset|file|volume|survey|single|few|lot|different)\b)")

# A request to read results, as opposed to a request to run the process.
_RESULT = re.compile(
    r"\b(explain|analy[sz]e|analysis|interpret|summari[sz]e|summary|review|assess|evaluate|judge|results?|outputs?|"
    r"outcomes?|findings?|verdict|report|scores?|good enough|trust|understand|what happened|walk me through|"
    r"tell me about|how (?:good|well|reliable|accurate)|what (?:did|does|do) .*(?:find|show|mean|produce|say))\b")
_RESULT_NOUN = re.compile(r"\b(results?|outputs?|outcomes?|findings?|verdict|scores?)\b")
_RUN = re.compile(
    r"\b(?:how (?:do|can|to) (?:i |we )?(?:run|start|launch)|"
    r"(?:run|rerun|re-run|start|launch|submit|execute|kick off|trigger)\s+(?:the\s+|a\s+|another\s+|new\s+)*"
    r"(?:process|stages?|foundations?|inference|volume products?|geobody|well[- ]geobody|prospect)|"
    r"prepare|configure|set up|setup|ready (?:to|for)|pre-?flight|requirements?|prerequisites?|needs?|requires?|required)\b")
_JOB_ID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE)
# Follow-up questions that ask for the overall reading again.
_OVERVIEW = re.compile(
    r"\b(good enough|fix first|improve|next steps?|overall|verdict|bottom line|what should (?:we|i)|what has to be fixed|"
    r"can (?:we|i) (?:trust|use|continue|move on)|find anything|meaningful|any real)\b")
# Follow-ups that belong to another part of Ask Geo (data inventory, projects, job history, viewers).
_ELSEWHERE = re.compile(
    r"\b(datasets?|projects?|uploads?|registered|uploaded|available|processing jobs?|job history|workflows?|scripts?|"
    r"open|view|display|switch|(?:have|has|with) (?:logs|data|check ?shots?|deviation))\b")

# First match wins, so the specific views come before the broad ones.
_FOCUS: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pattern)) for name, pattern in [
    ("all", r"\b(?:everything|full (?:report|details?|analysis|picture)|all (?:the )?details|complete (?:report|analysis)|in full)\b"),
    ("glossary", r"\bglossary\b|\bjargon\b|\bterms?\b|what does [\w -]{1,40} mean"),
    ("validation", r"\bblind\b|validat|\bfrozen\b|\bfreeze\b|\btruth\b"),
    ("audit", r"\baudit|provenance|integrity|reproduc|lineage|\bhash"),
    ("statistics", r"statistic|\bstats\b|distribution|value ranges?|\bsmooth|percentile|coverage"),
    ("figures", r"\b(?:figures?|plots?|images?|charts?|pictures?|sections?|dashboard)\b"),
    ("quality", r"\btrust|quality|reliab|\bscores?\b|\brating|which cubes|how good|accura"),
    ("products", r"seg-?y|\bfiles?\b|\bload|export|deliverables?|download|\bheaders?\b|\bbyte"),
    ("wells", r"\bwells?\b"),
]
]

# The handler's own suggested questions, mapped to the view that answers them.
_QUESTION_FOCUS: dict[str, str] = {
    "which cubes can i load and trust, and for what": "quality",
    "how do i load the seg-y files correctly": "products",
    "has this been tested on a blind well": "validation",
    "what did the audit find": "audit",
    "why does the shale-volume cube look so smooth": "statistics",
    "what should we do before sharing these cubes": "overview",
}


def _entity(message: str) -> dict[str, Any]:
    """This process has no one-item card."""
    return {}



def _normalise(message: str) -> str:
    return re.sub(r"\s+", " ", (message or "").casefold()).strip()


def _follows_this_report(history: list[dict[str, Any]] | None) -> bool:
    """True when the previous assistant message was this skill's rendered report."""
    for item in reversed(history or []):
        if item.get("role") == "assistant":
            return RESULT_MARKER in str(item.get("content") or "")
    return False


def parse_request(message: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Return handler parameters for a question about this process's results, else ``None``.

    Without ``history`` only questions that name the process are accepted. With ``history`` a
    follow-up to this skill's own report is accepted too, when it asks for one of the views.
    ``job_id`` is set only when the user wrote one; the agent otherwise uses the latest run.
    """
    text = _normalise(message)
    if not text:
        return None
    named = bool(_PROCESS.search(text))
    letters = {match.group(1) for match in _ANY_PROCESS_LETTER.finditer(text)}
    names_another = bool(letters - {_PROCESS_LETTER})
    if names_another and not named:
        return None
    topic = _PROCESS.sub(" ", text)
    question = text.rstrip(" ?.!")
    focus = _QUESTION_FOCUS.get(question)
    if focus is None:
        focus = next((name for name, pattern in _FOCUS if pattern.search(topic)), None)
    entity = _entity(message)

    if named:
        if _RUN.search(text) and not _RESULT_NOUN.search(text):
            return None  # a request to run or prepare the process belongs to the run skill
        if not (_RESULT.search(text) or focus or entity or _JOB_ID.search(text)):
            return None
    elif _follows_this_report(history):
        if question not in _QUESTION_FOCUS and _ELSEWHERE.search(text):
            return None
        if focus is None and _OVERVIEW.search(text):
            focus = "overview"
        if focus is None and not entity:
            return None
    else:
        return None

    parameters: dict[str, Any] = {"focus": focus or "overview", **entity}
    job = _JOB_ID.search(message or "")
    if job:
        parameters["job_id"] = job.group(0).lower()
    return parameters
