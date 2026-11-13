"""Read-only project dataset inventory skill."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Dataset, Well, Survey
from app.storage import find_dataset_path
from app.geochat.skills.synonyms import canonical_dataset_kind
from app.geochat.skills.data_inventory.descriptions import dataset_description


def execute(db: Session, project_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Return project-scoped dataset records and physical file accessibility."""
    statement = select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at, Dataset.original_filename, Dataset.id)
    data_type = parameters.get("data_type")
    if data_type:
        requested = data_type if isinstance(data_type, list) else [data_type]
        allowed = [canonical_dataset_kind(str(item)) for item in requested]
        if any(item is None for item in allowed):
            raise ValueError(
                "Unsupported dataset type. Use seismic, well_log, checkshot, or deviation."
            )
        statement = statement.where(Dataset.kind.in_(allowed))

    records = db.scalars(statement).all()
    datasets: list[dict[str, Any]] = []
    for record in records:
        well = db.get(Well, record.well_id) if getattr(record, "well_id", None) else None
        survey = db.get(Survey, record.survey_id) if getattr(record, "survey_id", None) else None
        well_name = well.name if well else None
        survey_name = survey.name if survey else None

        requested_well = parameters.get("well_name")
        requested_survey = parameters.get("survey_name")
        if requested_well and (well_name or "").casefold() != str(requested_well).casefold():
            continue
        if requested_survey and (survey_name or "").casefold() != str(requested_survey).casefold():
            continue

        file_path = find_dataset_path(record, db)
        description = dataset_description(record.kind, record.format, file_path)

        datasets.append({
            "dataset_id": record.id,
            "kind": record.kind,
            "name": record.name,
            "filename": record.original_filename,
            "format": record.format,
            "description": description,
            "well": well_name,
            "survey": survey_name,
            "status": record.status,
            "registered": True,
            "file_accessible": file_path is not None,
            "size_bytes": record.size_bytes,
        })

    return {"project_id": project_id, "count": len(datasets), "datasets": datasets}

