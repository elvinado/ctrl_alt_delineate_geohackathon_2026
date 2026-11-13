"""Workspace-level project creation and listing."""
from __future__ import annotations

import uuid
from sqlalchemy.orm import Session

from app.models import Project


def execute(*, db: Session, project_id: str | None, parameters: dict) -> dict:
    operation = str(parameters.get("operation", "")).strip().lower()
    if operation == "list":
        projects = db.query(Project).order_by(Project.created_at.desc()).all()
        return {
            "projects": [
                {"id": item.id, "name": item.name, "description": item.description,
                 "created_at": item.created_at.isoformat() if item.created_at else None}
                for item in projects
            ]
        }
    if operation == "switch":
        requested = str(parameters.get("name", "")).strip()
        if not requested:
            raise ValueError("Please specify which project to switch to.")
        projects = db.query(Project).order_by(Project.created_at.desc()).all()
        normalize = lambda value: " ".join(value.casefold().split())
        target = normalize(requested)
        exact = [item for item in projects if normalize(item.name) == target]
        matches = exact or [item for item in projects if target in normalize(item.name)]
        if not matches:
            available = ", ".join(item.name for item in projects) or "none"
            return {"project": None, "available": available}
        if len(matches) > 1:
            return {"project": None, "matches": [item.name for item in matches]}
        item = matches[0]
        return {"project": {"id": item.id, "name": item.name, "description": item.description}}
    if operation == "create":
        name = str(parameters.get("name", "")).strip()
        if not name:
            raise ValueError("Please provide a name for the new project.")
        if len(name) > 200:
            raise ValueError("Project names must be 200 characters or fewer.")
        project = Project(
            id=str(uuid.uuid4()), name=name,
            description=(str(parameters["description"]).strip() or None)
            if parameters.get("description") is not None else None,
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        return {"project": {"id": project.id, "name": project.name, "description": project.description}}
    raise ValueError("Project management supports create, list, and switch operations.")
