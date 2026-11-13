"""LLM-driven skill selection and response orchestration."""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from app.geochat.skills.registry import SKILL_REGISTRY, Skill, execute_skill
from app.geochat.skills.synonyms import normalize_skill_parameters
from app.geochat.skills.data_inventory.handler import execute as get_inventory
from app.geochat.skills.data_inventory.formatting import render_markdown as render_inventory
from app.geochat.skills.data_inventory.intent import parse_inventory_request, parse_well_name_filter
from app.geochat.skills.well_survey_inventory.intent import parse_entity_inventory_request
from app.geochat.skills.well_survey_inventory.handler import execute as get_entity_inventory
from app.geochat.skills.well_survey_inventory.formatting import render_markdown as render_entity_inventory
import re

logger = logging.getLogger(__name__)


class AgentAction(BaseModel):
    action: str = "answer"
    skill: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    clarification: str | None = None


def _parse_action(content: str | None) -> AgentAction:
    raw = (content or "{}").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        action = AgentAction.model_validate(json.loads(raw))
    except (ValueError, TypeError, ValidationError):
        return AgentAction()
    if action.action not in {"answer", "use_skill", "clarify", "open_upload", "view_dataset", "open_seismic_viewer", "open_well_viewer"}:
        return AgentAction()
    if action.action == "use_skill" and action.skill not in SKILL_REGISTRY:
        return AgentAction()
    return action

# --------------------------------------------------------------------------------------------
# Process skills (process_* and process_*_output_analysis)
#
# Nothing below names a workflow. A process skill is recognised by its registered name prefix, and
# everything specific to it lives in its own folder: SKILL.md (when to use it), handler.py (the
# evidence), intent.py (which questions it answers) and formatting.py (how the answer is laid out).
# --------------------------------------------------------------------------------------------
_JOB_ID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE)
_FINISHED = {"succeeded", "success", "completed"}
_NARRATIVE_SECTIONS = ("How to talk to the user", "Expert reading rules", "Safety")
_NARRATIVE_KEYS = (
    "status", "focus", "verdict", "at_a_glance", "takeaways", "scorecard", "cube_guide", "next_steps",
    "well_card", "body_card", "prospect_card",
)


def _is_output_analysis(skill: Skill) -> bool:
    return skill.name.startswith("process_") and skill.name.endswith("_output_analysis")


def _process_skills() -> list[Skill]:
    """Process skills that can parse a request themselves; result readers first, then run skills."""
    skills = [skill for skill in SKILL_REGISTRY.values() if skill.name.startswith("process_") and skill.intent]
    return sorted(skills, key=lambda skill: (not _is_output_analysis(skill), skill.name))


def _skill_module(skill: Skill, name: str) -> Any:
    try:
        return importlib.import_module(f"app.geochat.skills.{skill.name}.{name}")
    except ImportError:
        return None


def _skill_title(skill: Skill) -> str:
    title = getattr(_skill_module(skill, "formatting"), "TITLE", None)
    return str(title) if title else skill.name.replace("_", " ")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _job_records(db: Session) -> list[dict[str, Any]]:
    """Every processing job folder in the configured storage roots, newest first. Read-only."""
    from app.storage import configured_storage_roots

    records: dict[str, dict[str, Any]] = {}
    for root in configured_storage_roots(db):
        jobs_root = root / "processing_jobs"
        if not jobs_root.is_dir():
            continue
        for job_dir in jobs_root.iterdir():
            if job_dir.is_symlink() or not job_dir.is_dir() or job_dir.name in records:
                continue
            metadata = _read_json(job_dir / "job.json")
            result = _read_json(job_dir / "result.json")
            try:
                modified = job_dir.stat().st_mtime
            except OSError:
                modified = 0.0
            records[job_dir.name] = {
                "job_id": job_dir.name,
                "metadata": metadata,
                "status": str(result.get("status") or metadata.get("status") or "unknown").lower(),
                "order": (str(metadata.get("created_at") or result.get("created_at") or ""), modified),
            }
    return sorted(records.values(), key=lambda record: record["order"], reverse=True)


def _latest_job(db: Session, matches: Any, messages: list[dict[str, str]]) -> tuple[str | None, int]:
    """Pick the job a question is about: one named in the conversation, else the newest finished run.

    Returns the job ID and how many matching runs exist, so the answer can say when it chose one.
    """
    candidates = [record for record in _job_records(db) if matches(record["metadata"])]
    if not candidates:
        return None, 0
    known = {record["job_id"].lower(): record["job_id"] for record in candidates}
    for message in reversed(messages):
        for mentioned in reversed(_JOB_ID_RE.findall(message.get("content") or "")):
            if mentioned.lower() in known:
                return known[mentioned.lower()], len(candidates)
    finished = [record for record in candidates if record["status"] in _FINISHED]
    return (finished or candidates)[0]["job_id"], len(candidates)


def _project_dataset_descriptors(db: Session, project_id: str) -> list[dict[str, Any]]:
    """Registered datasets of one project, as the descriptors a pre-flight check validates."""
    from sqlalchemy import select
    from app.models import Dataset, Survey, Well

    descriptors = []
    for record in db.scalars(select(Dataset).where(Dataset.project_id == project_id)).all():
        well = db.get(Well, record.well_id) if getattr(record, "well_id", None) else None
        survey = db.get(Survey, record.survey_id) if getattr(record, "survey_id", None) else None
        try:
            metadata = json.loads(record.metadata_json or "{}")
        except (TypeError, ValueError):
            metadata = {}
        descriptors.append({
            "dataset_id": record.id, "kind": record.kind, "format": record.format,
            "name": record.original_filename or record.name,
            "well_name": well.name if well else None, "survey_name": survey.name if survey else None,
            "metadata": metadata if isinstance(metadata, dict) else {},
        })
    return descriptors


def _skill_guidance(skill: Skill) -> str:
    """The parts of SKILL.md that govern wording, so the reading prompt stays small."""
    parts = []
    for heading in _NARRATIVE_SECTIONS:
        match = re.search(rf"^## {re.escape(heading)}\s*\n(.+?)(?=^## |\Z)", skill.instructions, re.MULTILINE | re.DOTALL)
        if match:
            parts.append(f"## {heading}\n{match.group(1).strip()}")
    return "\n\n".join(parts) or skill.instructions


def explain_skill_result(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    skill: Skill,
    result: dict[str, Any],
    question: str,
    generation: dict[str, Any],
) -> tuple[str | None, Any]:
    """Ask the language model for the short plain-language reading of one skill result.

    The handler's evidence stays authoritative and is rendered separately; this only adds a few
    sentences that answer the user's question. Returns ``(None, None)`` when the model is
    unavailable, so the deterministic answer is still delivered.
    """
    evidence = {key: result[key] for key in _NARRATIVE_KEYS if result.get(key) not in (None, "", [], {})}
    evidence["key_findings"] = [
        {"level": item.get("level"), "observed": item.get("observed"), "plain_meaning": item.get("plain_meaning")}
        for item in (result.get("key_findings") or [])[:6] if isinstance(item, dict)
    ]
    details = result.get("details")
    if details and str(result.get("focus")) not in {"overview", "all"}:
        encoded = json.dumps(details, ensure_ascii=False, default=str)
        if len(encoded) <= 6000:
            evidence["details"] = details
    instructions = (
        "You are writing the short reading that sits above the result tables of a completed processing run. "
        "Write one paragraph of 3 to 6 plain sentences. Answer the user's question first. Use only facts found in the "
        "EVIDENCE JSON and quote its numbers exactly; never add a number, well, file or cause that is not there. "
        "Say 'the run shows' for facts and 'this usually means' for interpretation. Mention at most two caveats. "
        "Do not repeat the verdict sentence word for word, and do not list tables, findings or next steps: the user "
        "sees those below your paragraph. No headings, no bullet points, no tables, no code formatting."
    )
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_prompt}\n\n{instructions}\n\nSKILL GUIDANCE:\n{_skill_guidance(skill)}"},
                {"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE JSON:\n" + json.dumps(evidence, ensure_ascii=False, default=str)},
            ],
            temperature=min(float(generation.get("temperature", 0.2) or 0.0), 0.3),
            max_tokens=min(int(generation.get("max_tokens", 320) or 320), 320),
            top_p=generation.get("top_p", 1.0),
        )
    except (OpenAIError, OSError, RuntimeError, TypeError, ValueError):
        logger.warning("Plain-language reading unavailable for %s; returning the deterministic answer only", skill.name)
        return None, None
    text = (completion.choices[0].message.content or "").strip()
    # Keep it a paragraph: the fixed layout owns headings, lists and tables.
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith(("#", "|", "```"))]
    text = " ".join(re.sub(r"^(?:[-*+]|\d+\.)\s+", "", line) for line in lines)[:1500].strip()
    return (text or None), completion


_EVERY_PROCESS = re.compile(r"\b(?:each|every|all|whole|entire)\b[^.?!]*\b(?:process(?:es)?|stages?|pipeline)\b")
_RESULT_WORDS = re.compile(r"\b(?:explain|results?|outputs?|outcomes?|findings?|summar\w*|verdicts?|interpret|review|how (?:did|good))\b")
_VERDICT_WORDS = {
    "clean": "🟢 Clean", "usable_with_caveats": "🟡 Usable with caveats", "needs_attention": "🔴 Needs attention",
    "incomplete": "🔴 Incomplete", "did_not_finish": "🔴 Did not finish",
}
_LEVEL_ICONS = {"critical": "🔴", "caution": "🟡", "good": "🟢", "info": "🔵"}


def _one_line(value: Any) -> str:
    """Plain text that is safe inside a Markdown table cell or list item."""
    text = " ".join(str(value or "").replace("|", "/").split())
    return re.sub(r"(?<![`\w])([A-Za-z0-9][\w./-]*_[\w./-]*[A-Za-z0-9])(?![`\w])", r"`\1`", text)


def _every_process_reply(db: Session, messages: list[dict[str, str]]) -> str:
    """One verdict per registered process, from the latest run of each. Deterministic; no model call."""
    rows, sections, missing = [], [], []
    for skill in (item for item in _process_skills() if _is_output_analysis(item)):
        title = _skill_title(skill)
        supports_job = getattr(_skill_module(skill, "handler"), "supports_job", None)
        job_id = _latest_job(db, supports_job, messages)[0] if callable(supports_job) else None
        if not job_id:
            rows.append(f"| {title} | — | ⚪ No run found | Run it from the Processing tab to get results. |")
            missing.append(title)
            continue
        try:
            result = execute_skill(skill.name, db=db, project_id=None, parameters={"job_id": job_id})
        except Exception:
            logger.exception("Could not summarise %s", skill.name)
            rows.append(f"| {title} | `{job_id}` | ⚪ Not readable | The outputs of this run could not be read. |")
            continue
        verdict = result.get("verdict") if isinstance(result.get("verdict"), dict) else {}
        level = str(verdict.get("level") or "")
        word = _VERDICT_WORDS.get(level, "⚪ Not analysed" if not result.get("applicable") else f"⚪ {level.replace('_', ' ')}")
        rows.append(f"| {title} | `{job_id}` | {word} | {_one_line(verdict.get('plain') or result.get('summary'))} |")
        points = [item for item in (result.get("takeaways") or []) if isinstance(item, dict)][:2]
        if points:
            sections.append(f"## {title}")
            sections.append("\n".join(
                f"- {_LEVEL_ICONS.get(str(item.get('level')), '🔵')} {_one_line(item.get('observed'))} "
                f"**Meaning:** {_one_line(item.get('plain_meaning'))}" for item in points))
    if not rows:
        return "No process result readers are registered, so there is nothing to summarise."
    lines = [
        "# Results of every process — latest run of each", "",
        "| Process | Latest run | Verdict | What the run says |", "|---|---|---|---|", *rows,
    ]
    if sections:
        lines.extend(["", *[part for section in sections for part in (section, "")]])
    lines.extend(["---", "Ask for one process to see its tables and a plain-language reading, for example "
                  "**Explain the results of Process A**. Each verdict is read from that run's own files; nothing is recomputed."])
    return "\n".join(lines)


def _render_skill_result(skill: Skill, result: dict[str, Any]) -> str:
    if skill.renderer:
        return skill.renderer(result)
    return "```json\n" + json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n```"


def _process_skill_reply(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    generation: dict[str, Any],
    skill: Skill,
    parameters: dict[str, Any],
    messages: list[dict[str, str]],
    project_id: str | None,
    db: Session,
) -> tuple[str, str | None, str | None, Any, Any]:
    """Run one process skill and return its answer in the skill's own fixed layout."""
    parameters = dict(parameters)
    title = _skill_title(skill)
    if not project_id and skill.scope != "global":
        return (f"Select a project first so I can check **{title}** against its data.", skill.name, None, None, None)

    if _is_output_analysis(skill):
        supports_job = getattr(_skill_module(skill, "handler"), "supports_job", None)
        chosen_for_user, runs = False, 0
        if parameters.get("job_id") and str(parameters["job_id"]) not in {record["job_id"] for record in _job_records(db)}:
            parameters.pop("job_id")  # a planner may pass words such as "latest"; fall back to the newest run
        if not parameters.get("job_id"):
            if not callable(supports_job):
                return (f"Tell me the job ID of the **{title}** run you want explained.", skill.name, None, None, None)
            parameters["job_id"], runs = _latest_job(db, supports_job, messages)
            chosen_for_user = True
            if not parameters["job_id"]:
                return (
                    f"I could not find a run of **{title}** in this workspace, so there are no results to explain yet. "
                    "Run it from the Processing tab first, or give me the job ID if it lives elsewhere.",
                    skill.name, None, None, None,
                )
        try:
            result = execute_skill(skill.name, db=db, project_id=project_id, parameters=parameters)
        except ValueError as exc:
            return str(exc), skill.name, None, None, None
        except Exception:
            logger.exception("Output analysis failed for %s", skill.name)
            return (
                f"I couldn't read the outputs of that **{title}** run due to an internal error. "
                "Please try again; if it persists, check the backend logs.",
                skill.name, None, None, None,
            )
        completion = None
        if result.get("applicable"):
            result["narrative"], completion = explain_skill_result(
                client, model=model, system_prompt=system_prompt, skill=skill, result=result,
                question=messages[-1]["content"], generation=generation,
            )
        content = _render_skill_result(skill, result)
        if chosen_for_user and runs > 1:
            content += f"\n\nThis is the most recent of {runs} runs of this process. Give me a job ID to read a different one."
        return content, skill.name, None, None, completion

    # A run skill: pre-flight only. The handler validates and plans; it never submits.
    source = parameters.pop("_source_job", None)
    wants_datasets = parameters.pop("_project_datasets", False)
    if isinstance(source, dict) and source.get("parameter") and not parameters.get(source["parameter"]):
        script = str(source.get("python_script") or "")
        job_id, _ = _latest_job(db, lambda metadata: Path(str(metadata.get("python_script") or "")).name == script, messages)
        if job_id:
            parameters[source["parameter"]] = job_id
    if wants_datasets and project_id and not parameters.get("datasets"):
        try:
            parameters["datasets"] = _project_dataset_descriptors(db, project_id)
        except Exception:
            logger.exception("Could not list project datasets for %s", skill.name)
    try:
        result = execute_skill(skill.name, db=db, project_id=project_id, parameters=parameters)
    except ValueError as exc:
        return str(exc), skill.name, None, None, None
    except Exception:
        logger.exception("Pre-flight check failed for %s", skill.name)
        return (
            f"I couldn't complete the pre-flight check for **{title}** due to an internal error. "
            "Please try again; if it persists, check the backend logs.",
            skill.name, None, None, None,
        )
    if result.get("ui_action"):
        return (str(result.get("message") or "Processing request submitted."), skill.name, str(result["ui_action"]), result.get("ui_action_data") or {}, None)
    return _render_skill_result(skill, result), skill.name, None, None, None


def run_agent(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    project_id: str | None,
    db: Session,
    generation: dict[str, Any],
) -> tuple[str, str | None, str | None, Any, Any]:
    """Select a skill or a supported frontend action, then generate a response."""
    # Handle common inventory and viewer intents deterministically. These actions
    # should not depend on the LLM's prose generation or tool-routing reliability.
    latest = messages[-1]["content"].strip()
    history_text = "\n".join(item["content"] for item in messages[:-1])
    normalized = latest.casefold()

    # Process skills first: a question that names a process (or follows that process's own report)
    # is answered by the skill that owns it, in its fixed layout. Each skill's intent.py decides.
    if _EVERY_PROCESS.search(normalized) and _RESULT_WORDS.search(normalized) and _process_skills():
        return _every_process_reply(db, messages), None, None, None, None
    for skill in _process_skills():
        try:
            process_parameters = skill.intent(latest, messages[:-1])
        except Exception:
            logger.exception("Intent parser failed for %s", skill.name)
            continue
        if process_parameters is not None:
            return _process_skill_reply(
                client, model=model, system_prompt=system_prompt, generation=generation, skill=skill,
                parameters=process_parameters, messages=messages, project_id=project_id, db=db,
            )

    # Processing history is workspace-wide and should be rendered deterministically
    # so every requested column is present and sourced from persisted job records.
    wants_job_history = bool(re.search(
        r"\b(list|show|view|display|review)\b.*\b(completed|finished|previous|past|processing)\s+(?:processing\s+)?jobs?\b|"
        r"\b(completed|finished)\s+processing\s+jobs?\b|"
        r"\bprocessing\s+(?:job\s+)?history\b|\bwhat processing jobs have (?:completed|finished)\b",
        normalized,
    ))
    if wants_job_history:
        try:
            from app.geochat.skills.processing_job_history.handler import execute as get_job_history
            history = get_job_history(db=db, project_id=None, parameters={})
        except Exception:
            logger.exception("Could not retrieve completed processing-job history")
            return (
                "I couldn't retrieve processing-job history due to an internal error. Please try again or check the backend logs.",
                "processing_job_history", None, None, None,
            )
        if not history["items"]:
            return "No completed processing jobs were found in the workspace.", "processing_job_history", None, None, None

        def table_cell(value: Any) -> str:
            return str(value if value not in (None, "") else "—").replace("|", r"\|").replace("\n", " ").strip()

        lines = [
            "**Completed processing jobs**",
            "",
            "| Python file | ID | Status | Description or summary | Outputs/results |",
            "|---|---|---|---|---|",
        ]
        for item in history["items"]:
            outputs = ", ".join(item["outputs"]) if item["outputs"] else "—"
            lines.append(
                "| " + " | ".join(table_cell(value) for value in (
                    item["python_file"], item["id"], item["status"], item["summary"], outputs,
                )) + " |"
            )
        lines.extend(["", f"Showing {history['count']} completed jobs (newest first)."])
        return "\n".join(lines), "processing_job_history", None, None, None

    # Workspace project management is global: it must work independently of the
    # currently selected project and must not be routed through project-scoped skills.
    wants_project_list = bool(re.search(
        r"\b(list|show|view)\b.*\b(projects|my projects)\b|\b(what|which) projects (are )?(available|there)|\bwhat (are )?(my|the) projects\b|\bprojects do i have\b",
        normalized,
    ))
    wants_project_create = bool(re.search(
        r"\b(create|make|start|new)\b.*\b(project)\b|\bnew project\b", normalized
    ))
    switch_match = re.search(
        r"\b(?:switch|change|go)\s+(?:over\s+)?to\s+(?:the\s+)?(?:project\s+)?(.+?)\s*[.!?]*$|"
        r"\bopen\s+(?:the\s+)?project\s+(.+?)\s*[.!?]*$",
        latest, re.IGNORECASE,
    )
    switch_name = next((part.strip().strip(" \"'`.,!? ") for part in switch_match.groups() if part), "") if switch_match else ""
    wants_project_switch = bool(switch_name)
    if wants_project_list or wants_project_create or wants_project_switch:
        if wants_project_create:
            name_match = re.search(
                r"\b(?:called|named)\s+['\"]?(.+?)['\"]?\s*$|\bproject\s+['\"]([^'\"]+)['\"]",
                latest, re.IGNORECASE,
            )
            name = next((part.strip().strip(" .!?\"'") for part in name_match.groups() if part), "") if name_match else ""
            if not name:
                direct_name = re.search(r"\b(?:create|make|start|new)\s+(?:a\s+)?project\s+(.+)$", latest, re.IGNORECASE)
                if direct_name:
                    name = direct_name.group(1).strip().strip(" .!?\"'")
            if not name:
                return (
                    "What would you like to name the new project?",
                    "project_management", None, None, None,
                )
            try:
                created = execute_skill(
                    "project_management", db=db, project_id=None,
                    parameters={"operation": "create", "name": name},
                )["project"]
            except ValueError as exc:
                return str(exc), "project_management", None, None, None
            return (
                f"Created project **{created['name']}**. Opening its Project tab now.",
                "project_management", "switch_project",
                {"project_id": created["id"], "project_name": created["name"]}, None,
            )
        if wants_project_switch:
            try:
                result = execute_skill(
                    "project_management", db=db, project_id=None,
                    parameters={"operation": "switch", "name": switch_name},
                )
            except ValueError as exc:
                return str(exc), "project_management", None, None, None
            target = result.get("project")
            if target:
                return (
                    f"Switching to **{target['name']}** and opening its Project tab.",
                    "project_management", "switch_project",
                    {"project_id": target["id"], "project_name": target["name"]}, None,
                )
            if result.get("matches"):
                options = ", ".join(f"**{name}**" for name in result["matches"])
                return f"More than one project matches “{switch_name}”: {options}. Please specify the full project name.", "project_management", None, None, None
            return f"I couldn't find a project matching **{switch_name}**. Available projects: {result.get('available', 'none')}.", "project_management", None, None, None
        try:
            projects = execute_skill(
                "project_management", db=db, project_id=None,
                parameters={"operation": "list"},
            )["projects"]
        except ValueError as exc:
            return str(exc), "project_management", None, None, None
        if not projects:
            return "No projects have been created in this workspace yet.", "project_management", None, None, None
        lines = ["**Workspace projects**", ""]
        lines.extend(f"- **{item['name']}** — `{item['id']}`" for item in projects)
        return "\n".join(lines), "project_management", None, None, None

    # Keep the help response grounded in the live skill registry and implemented UI actions.
    asks_capabilities = normalized.strip() == "help" or bool(re.search(
        r"\b(what can you do|what can ask geo do|what can i ask|how can you help|how do i use ask geo|help me get started|show me (your )?(capabilities|features)|list (your )?(capabilities|features))\b",
        normalized,
    ))
    if asks_capabilities:
        # Capability discovery is project-independent and does not access project data.
        capability_skill = SKILL_REGISTRY["ask_geo_capabilities"]
        if capability_skill.handler is None:
            return "The Ask Geo capability catalog is not available.", "ask_geo_capabilities", None, None, None
        guide = capability_skill.handler(db=db, project_id=project_id or "", parameters={})
        lines = ["**What Ask Geo can do**", ""]
        for item in guide["capabilities"]:
            lines.append(f"- **{item['name'].replace('_', ' ')}:** {item['description']}")
        lines.extend(["", "**Open and work with data**"])
        for item in guide["frontend_actions"]:
            lines.append(f"- **{item['name'].replace('_', ' ')}:** {item['description']}")
        lines.extend(["", "**Example questions**"])
        lines.extend(f"- {question}" for question in guide["examples"])
        lines.extend(["", "**Current limits**"])
        lines.extend(f"- {limit}" for limit in guide["limitations"])
        lines.append("\nSelect a project for project-specific data and processing operations.")
        return "\n".join(lines), "ask_geo_capabilities", None, None, None

    wants_processing_catalog = bool(
        re.search(r"\b(process(?:ing)?|script|model|workflow)\b", normalized)
        and re.search(r"\b(available|availability|what|which|list|scan|discover|catalog|can\s+i\s+run|to\s+run)\b", normalized)
    )
    if wants_processing_catalog:
        if not project_id:
            return (
                "Select a project first so I can scan its configured processing scripts.",
                "processing_catalog", None, None, None,
            )
        try:
            catalog = execute_skill(
                "processing_catalog", db=db, project_id=project_id, parameters={}
            )
        except ValueError as exc:
            return str(exc), "processing_catalog", None, None, None

        if not catalog.get("directory_exists"):
            return (
                f"The processing scripts directory was not found: `{catalog['scripts_directory']}`.",
                "processing_catalog", None, None, None,
            )
        # Keep this response intentionally narrow: the processing catalogue is
        # the source of truth, and the frontend turns each script name into a
        # direct action that selects that script in the Processing tab.
        scripts = catalog.get("scripts") or catalog.get("workers") or []
        lines = ["| Scripts | Description |", "|---|---|"]
        if scripts:
            for item in scripts:
                name = str(item.get("name") or "").strip()
                description = str(item.get("description") or "No description available").strip()
                if name:
                    safe_name = name.replace("|", "\\|")
                    safe_description = description.replace("|", "\\|").replace("\n", " ")
                    lines.append(f"| `{safe_name}` | {safe_description} |")
        else:
            lines.append("| No Python scripts found | — |")
        return "\n".join(lines), "processing_catalog", None, None, None

    # Entity inventory is separate from dataset inventory: "what well do I have?"
    # lists Well records, while "what well data do I have?" lists associated datasets.
    entity_filters = parse_entity_inventory_request(latest)
    if entity_filters:
        if not project_id:
            return (
                "Select a project first so I can list its registered wells or surveys.",
                "well_survey_inventory", None, None, None,
            )
        try:
            entity_inventory = get_entity_inventory(db, project_id, entity_filters)
        except Exception:
            logger.exception("Deterministic well/survey inventory query failed")
            return (
                "I couldn't retrieve the project well/survey list due to an internal error. "
                "Please try again; if it persists, check the backend logs.",
                "well_survey_inventory", None, None, None,
            )
        return render_entity_inventory(entity_inventory), "well_survey_inventory", None, None, None

    wants_view = bool(re.search(r"\b(view|open|display|visuali[sz]e|plot)\b", normalized))
    inventory_filters = parse_inventory_request(latest)
    wants_inventory = inventory_filters is not None
    requested_type = inventory_filters.get("data_type") if inventory_filters else None

    # Resolve follow-ups such as "view that file" from the preceding conversation.
    follow_up_view = wants_view and bool(re.search(r"\b(that|it|this)\s+(file|dataset)\b|\b(open|view)\s+(that|it)\b", normalized))
    prior_filename = None
    if follow_up_view:
        filenames = re.findall(r"[\w .-]+\.(?:txt|las|sgy|segy)\b", history_text, flags=re.IGNORECASE)
        if filenames:
            prior_filename = filenames[-1].strip().rstrip(".,;:")
            wants_view = True

    # Only accept well identifiers such as M-1; never treat the word "data"
    # in "well data" as an entity name.
    requested_well = (inventory_filters or {}).get("well_name") or parse_well_name_filter(normalized)
    requested_survey = (inventory_filters or {}).get("survey_name")

    if wants_inventory and (not wants_view or "available" in normalized or "inventory" in normalized or isinstance(requested_type, list)):
        if not project_id:
            return (
                "Select a project first so I can check its registered datasets and well data.",
                "data_inventory", None, None, None,
            )
        filters: dict[str, Any] = dict(inventory_filters or {})
        if requested_well:
            filters["well_name"] = requested_well
        if requested_survey:
            filters["survey_name"] = requested_survey
        try:
            if prior_filename:
                # Exact filename matching avoids accidentally returning a similarly
                # named dataset when a user asks about a specific file.
                inventory = get_inventory(db, project_id, {})
                inventory["datasets"] = [
                    row for row in inventory["datasets"]
                    if (row.get("filename") or row.get("name") or "").casefold() == prior_filename.casefold()
                ]
                inventory["count"] = len(inventory["datasets"])
            else:
                inventory = get_inventory(db, project_id, filters)
        except ValueError as exc:
            return str(exc), "data_inventory", None, None, None
        except Exception:
            logger.exception("Deterministic dataset inventory query failed")
            return (
                "I couldn't retrieve the dataset inventory due to an internal error. "
                "Please try again; if it persists, check the backend logs.",
                "data_inventory", None, None, None,
            )
        if requested_well and prior_filename:
            inventory["datasets"] = [
                row for row in inventory["datasets"]
                if (row.get("well") or "").casefold() == requested_well.casefold()
            ]
            inventory["count"] = len(inventory["datasets"])
        return render_inventory(inventory), "data_inventory", None, None, None

    if project_id and wants_view and (requested_type or prior_filename):
        inventory = get_inventory(db, project_id, {})
        records = inventory["datasets"]
        if prior_filename:
            records = [row for row in records if (row.get("filename") or row.get("name") or "").casefold() == prior_filename.casefold()]
        if requested_type:
            records = [row for row in records if row.get("kind") == requested_type]
        if requested_well:
            records = [row for row in records if (row.get("well") or "").casefold() == requested_well.casefold()]
        records = [row for row in records if row.get("file_accessible")]
        if records:
            record = records[0]
            return "", None, "view_dataset", {
                "dataset_name": record.get("filename") or record.get("name"),
                "dataset_type": record.get("kind"),
                "well_name": record.get("well"),
            }, None
        return (f"I couldn't find an accessible {requested_type or 'matching'} dataset" + (f" for well {requested_well}" if requested_well else "") + " in this project.", None, None, None, None)

    skill_catalog = [
        {"name": skill.name, "description": skill.description}
        for skill in SKILL_REGISTRY.values()
    ]
    synonyms_path = Path(__file__).resolve().parent / "skills" / "SYNONYMS.md"
    synonyms = synonyms_path.read_text(encoding="utf-8") if synonyms_path.is_file() else ""
    router_prompt = (
        "You are the action planner for a geoscience assistant. Choose whether to answer "
        "directly, use exactly one listed skill, ask for clarification, or request a frontend "
        "upload or viewer navigation workflow. Do not answer the user's question in this step. Return ONLY JSON matching: "
        '{"action":"answer|use_skill|clarify|open_upload|view_dataset","skill":null,"parameters":{},"clarification":null}. '
        "Use open_upload when the user explicitly asks to upload, add, import, or browse for "
        "geological data/files. Do not use it for questions about existing datasets. "
        "For a clear viewing request, select the appropriate viewer action immediately; do not add examples or explanatory suggestions. "
        "Use view_dataset when the user asks to view/display/open a specific dataset. Set dataset_type to segy, las, deviation, or checkshot based on the filename or context. "
        ""
        "If the requested viewer or dataset type is genuinely unclear, choose clarify and ask one short question; only then give one or two supported example requests. "
        "Only route to viewer actions for these existing capabilities: SEG-Y seismic trace preview; LAS well-log curve preview; deviation/checkshot TXT table preview. "
        "Do not claim support for 3D seismic volumes, maps, crossplots, arbitrary CSV plotting, or other visualizations. "
        "For a specific dataset, put its exact or closest-mentioned filename/name in parameters.dataset_name. "
        "For a general request, leave dataset_name null and set parameters.dataset_type to segy, las, deviation, or checkshot when clear. "
        "For processing workflows, choose the most specific registered process_* skill using its own instructions. "
        "Do not hardcode workflow names or parameters in the shared router; each workflow's SKILL.md defines its trigger, required inputs, exclusions, and follow-up behavior. "
        "Never use a processing execution skill to analyze an existing output unless its instructions explicitly say so; prefer a separately registered output-analysis skill when available. "
        "For a process_*_output_analysis skill, set parameters.focus to the view the question asks for when one is clear, add the well, body or prospect the user named, and leave job_id out unless the user wrote one: the application then reads the most recent run. "
        "Use processing_catalog when the user asks what processing scripts, processes, models, or workflows are available, or asks to scan/list what can be run. This is a read-only scan of the active scripts directory; it must not execute discovered files. "
        "Use processing_job_history when the user asks to list, show, or review completed/finished processing jobs or processing history. This is a workspace-wide read-only history query and must not submit, modify, or delete jobs. "
        "Use data_inventory for questions about project-specific datasets, files, well data, or survey data. "
        "Use well_survey_inventory when the user asks to list the project well entities or survey entities themselves. "
        "Use geoscience_knowledge for conceptual geoscience questions. "
        "Do not select a skill for ordinary conversation. Never invent parameter values. "
        f"Available skills: {json.dumps(skill_catalog, ensure_ascii=False)}\n\n"
        "DOMAIN SYNONYMS AND CANONICAL TERMS:\n" + synonyms
    )
    decision_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": router_prompt},
            *messages[-8:],
        ],
        temperature=0,
        max_tokens=300,
    )
    decision = _parse_action(decision_response.choices[0].message.content)

    if decision.action == "use_skill" and decision.skill == "well_survey_inventory":
        if not project_id:
            return (
                "Select a project first so I can list its registered wells or surveys.",
                "well_survey_inventory", None, None, None,
            )
        try:
            entity_inventory = execute_skill(
                "well_survey_inventory",
                db=db,
                project_id=project_id,
                parameters=decision.parameters,
            )
        except ValueError as exc:
            return str(exc), "well_survey_inventory", None, None, None
        except Exception:
            logger.exception("Planner-routed well/survey inventory query failed")
            return (
                "I couldn't retrieve the project well/survey list due to an internal error. "
                "Please try again; if it persists, check the backend logs.",
                "well_survey_inventory", None, None, None,
            )
        return render_entity_inventory(entity_inventory), "well_survey_inventory", None, None, None

    # If the planner recognizes an inventory request that the deterministic
    # phrase parser did not, still execute and render it directly. Never rely on
    # a second LLM completion to format authoritative inventory data.
    if decision.action == "use_skill" and decision.skill == "data_inventory":
        if not project_id:
            return (
                "Select a project first so I can check its registered datasets and well data.",
                "data_inventory", None, None, None,
            )
        try:
            inventory = execute_skill(
                "data_inventory",
                db=db,
                project_id=project_id,
                parameters=normalize_skill_parameters("data_inventory", decision.parameters),
            )
        except ValueError as exc:
            return str(exc), "data_inventory", None, None, None
        except Exception:
            logger.exception("Planner-routed dataset inventory query failed")
            return (
                "I couldn't retrieve the dataset inventory due to an internal error. "
                "Please try again; if it persists, check the backend logs.",
                "data_inventory", None, None, None,
            )
        return render_inventory(inventory), "data_inventory", None, None, None

    # A process skill chosen by the planner gets the same fixed layout as one chosen by its own parser.
    if decision.action == "use_skill" and decision.skill and decision.skill.startswith("process_") and SKILL_REGISTRY[decision.skill].handler:
        return _process_skill_reply(
            client, model=model, system_prompt=system_prompt, generation=generation, skill=SKILL_REGISTRY[decision.skill],
            parameters=normalize_skill_parameters(decision.skill, decision.parameters), messages=messages,
            project_id=project_id, db=db,
        )

    selected_skill = None
    ui_action = ("view_dataset" if decision.action in {"view_dataset", "open_seismic_viewer", "open_well_viewer"} else "open_upload" if decision.action == "open_upload" else None)
    ui_action_data = decision.parameters if ui_action == "view_dataset" else None
    skill_context = ""
    if decision.action == "open_upload":
        skill_context = (
            "The user requested a data upload. Tell them the Input Data upload dialog is being opened. "
            "Do not claim that any file has already been uploaded."
        )
    elif decision.action == "clarify":
        skill_context = "The planner requests clarification. Ask the user: " + (decision.clarification or "Please clarify what you would like to do.")
    elif decision.action == "use_skill" and decision.skill:
        skill = SKILL_REGISTRY[decision.skill]
        selected_skill = skill.name
        skill_context = "SELECTED SKILL INSTRUCTIONS:\n" + skill.instructions
        if skill.handler:
            try:
                result = execute_skill(
                    skill.name,
                    db=db,
                    project_id=project_id,
                    parameters=normalize_skill_parameters(skill.name, decision.parameters),
                )
                if result.get("requires_input"):
                    return str(result.get("message") or "Please provide the required processing input."), skill.name, None, None, None
                if result.get("ui_action"):
                    return (str(result.get("message") or "Processing request submitted."), skill.name, str(result["ui_action"]), result.get("ui_action_data") or {}, None)
                skill_context += "\n\nAUTHORITATIVE SKILL RESULT (JSON):\n" + json.dumps(result, ensure_ascii=False, default=str)
            except ValueError as exc:
                skill_context += "\n\nSKILL COULD NOT RUN: " + str(exc)
        else:
            skill_context += "\n\nNo application handler is needed for this knowledge skill. Answer using your domain knowledge."

    final_system = system_prompt
    if skill_context:
        final_system += "\n\nAGENT SKILL CONTEXT:\n" + skill_context
    if decision.action == "clarify":
        final_system += "\nDo not guess; ask the requested clarification question."

    # Viewer navigation is a UI action, not a conversational answer. Avoid a
    # second LLM call: it adds latency and can fail after routing already succeeded.
    if ui_action == "view_dataset":
        return "", selected_skill, ui_action, ui_action_data, None
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": final_system}, *messages],
        **generation,
    )
    content = completion.choices[0].message.content or ""
    return content, selected_skill, ui_action, ui_action_data, completion
