"""Discover self-contained skills from their SKILL.md and optional handler.py, formatting.py and intent.py."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import re
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    handler: Callable[..., dict[str, Any]] | None = None
    scope: str = "project"
    # Optional companions, discovered by file name like the handler:
    # formatting.render_markdown(result) -> str and intent.parse_request(message, history) -> dict | None.
    renderer: Callable[[dict[str, Any]], str] | None = None
    intent: Callable[..., dict[str, Any] | None] | None = None


SKILLS_DIR = Path(__file__).resolve().parent
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _description(markdown: str, fallback: str) -> str:
    match = re.search(r"^## Description\s*\n(.+?)(?:\n\s*\n|\n## )", markdown, re.MULTILINE | re.DOTALL)
    return " ".join(match.group(1).split()) if match else fallback


def _optional(directory: Path, module_name: str, attribute: str) -> Callable[..., Any] | None:
    """Return ``<skill>/<module_name>.py:<attribute>`` when the skill provides it, else ``None``."""
    if not (directory / f"{module_name}.py").is_file():
        return None
    module = importlib.import_module(f"app.geochat.skills.{directory.name}.{module_name}")
    found = getattr(module, attribute, None)
    return found if callable(found) else None


def discover_skills() -> dict[str, Skill]:
    """Load skills from immediate child directories; underscore dirs are templates/internal."""
    discovered: dict[str, Skill] = {}
    for directory in sorted(SKILLS_DIR.iterdir()):
        if not directory.is_dir() or directory.name.startswith("_") or not _NAME_RE.fullmatch(directory.name):
            continue
        instructions_path = directory / "SKILL.md"
        if not instructions_path.is_file():
            continue
        instructions = instructions_path.read_text(encoding="utf-8")
        handler = None
        if (directory / "handler.py").is_file():
            module = importlib.import_module(f"app.geochat.skills.{directory.name}.handler")
            handler = getattr(module, "execute", None)
            if not callable(handler):
                raise RuntimeError(f"Skill {directory.name} has handler.py but no callable execute()")
        renderer = _optional(directory, "formatting", "render_markdown")
        intent = _optional(directory, "intent", "parse_request")
        discovered[directory.name] = Skill(
            name=directory.name,
            description=_description(instructions, directory.name.replace("_", " ")),
            instructions=instructions,
            handler=handler,
            renderer=renderer,
            intent=intent,
            scope="global" if re.search(r"^## Scope\s*\n\s*`?global`?\s*$", instructions, re.MULTILINE | re.IGNORECASE) else "project",
        )
    return discovered


SKILL_REGISTRY = discover_skills()


def get_skill_descriptions() -> list[dict[str, str]]:
    return [{"name": skill.name, "description": skill.description} for skill in SKILL_REGISTRY.values()]


def get_skill(name: str) -> Skill | None:
    return SKILL_REGISTRY.get(name)


def execute_skill(name: str, *, db: Session, project_id: str | None, parameters: dict[str, Any]) -> dict[str, Any]:
    skill = get_skill(name)
    if skill is None:
        raise ValueError(f"Unknown skill: {name}")
    if skill.handler is None:
        return {"execution": "not_required", "note": "This is an instruction-only skill."}
    if not project_id and skill.scope != "global":
        raise ValueError("A project must be selected before using this skill.")
    return skill.handler(db=db, project_id=project_id, parameters=parameters)
