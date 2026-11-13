"""Read-only inventory of well and survey entities in the active project."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Survey, Well


def execute(db: Session, project_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
    entity_type = str(parameters.get("entity_type") or "").casefold()
    if entity_type == "wells":
        records = db.scalars(
            select(Well).where(Well.project_id == project_id).order_by(Well.name, Well.id)
        ).all()
        items = [{"name": row.name, "uwi": row.uwi} for row in records]
    elif entity_type == "surveys":
        records = db.scalars(
            select(Survey).where(Survey.project_id == project_id).order_by(Survey.name, Survey.id)
        ).all()
        items = [{"name": row.name, "description": row.description, "crs": row.crs} for row in records]
    else:
        raise ValueError("entity_type must be 'wells' or 'surveys'.")

    return {
        "project_id": project_id,
        "entity_type": entity_type,
        "count": len(items),
        "items": items,
    }
