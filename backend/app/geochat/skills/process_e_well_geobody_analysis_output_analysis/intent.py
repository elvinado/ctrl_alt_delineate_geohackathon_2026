"""Deterministic interpretation of questions about Process E (well-geobody analysis, stage 13) results."""
from __future__ import annotations

import re
from typing import Any

from app.geochat.skills.process_e_well_geobody_analysis_output_analysis.formatting import RESULT_MARKER

# How users name this process.
_PROCESS = re.compile(
    r"\bprocess[\s_-]*e\b|\bwell[- ]geobody(?: analysis)?\b|\bstage\s*13\b|"
    r"process_e_well_geobody_analysis|\bwells? (?:intersect|cross|penetrat|meet)\w* (?:the |a |any )?(?:geo)?bod",
    re.IGNORECASE)
_PROCESS_LETTER = "e"
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
    ("properties", r"porosity|saturation|\bphit\b|\bswt\b|\bpropert"),
    ("depth", r"\bdeep|\bdepths?\b|\bthick|time[- ]depth|\bt-?z\b|\btvd"),
    ("provenance", r"provenance|pre-?flight|\bchecks?\b|\bfiles?\b|source jobs?|output tables?"),
    ("wells", r"\bwells?\b|intersect|\bcross|penetrat|\bpaths?\b|\bconfirm|same body"),
]
]

# The handler's own suggested questions, mapped to the view that answers them.
_QUESTION_FOCUS: dict[str, str] = {
    "did any well confirm a geobody": "wells",
    "why does every well meet the same body": "wells",
    "how deep is the body, and how thick": "depth",
    "why are there no porosity or saturation numbers": "properties",
    "tell me about one well": "wells",
    "what should we fix to make this analysis meaningful": "overview",
}

_WELL = re.compile(r"\b(?:well\s+)?([a-z]{1,4}-\d+[a-z]?(?:-?st\d*)?)\b", re.IGNORECASE)


def _entity(message: str) -> dict[str, Any]:
    """A well named in the question, for the one-well card."""
    match = _WELL.search(message or "")
    return {"well": match.group(1)} if match else {}



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
