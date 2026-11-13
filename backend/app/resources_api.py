import json
import os
import uuid
import math
from io import BytesIO
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Project, Survey, Well, Dataset, SeismicDataset, CheckshotDataset, DeviationDataset, Curve, WellLogDataset
from app.schemas_common import (DatasetLinkLocal, ProjectCreate, ProjectPatch, ProjectOut, SurveyCreate, SurveyPatch, SurveyOut, WellCreate, WellPatch, WellOut, DatasetPatch, DatasetOut)
from app.storage import ingest_file, resolve_dataset_path, find_dataset_path, register_linked_file, allowed_link_roots
from app.file_headers import read_file_header
from app.curve_data import read_curve_samples
from app.well_tables import read_well_table
from app.dataset_data import read_seismic_data, read_well_data, read_table_data, read_excel_well_list
from app.seismic_viewer import geometry_metadata, read_seismic_section
from app.runtime_settings import get_effective_settings

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "service": "geohackathon-data-api"}

@router.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    p = Project(id=str(uuid.uuid4()), name=payload.name, description=payload.description)
    db.add(p); db.commit(); db.refresh(p)
    return p

@router.get("/projects", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return db.query(Project).order_by(Project.created_at.desc()).all()

@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, db: Session = Depends(get_db)):
    p = db.get(Project, project_id)
    if not p: raise HTTPException(404, "Project not found")
    return p

@router.patch("/projects/{project_id}", response_model=ProjectOut)
def patch_project(project_id: str, payload: ProjectPatch, db: Session = Depends(get_db)):
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    changes = payload.model_dump(exclude_unset=True)
    if "name" in changes and changes["name"] is None:
        raise HTTPException(422, "name cannot be null")
    for field, value in changes.items():
        setattr(p, field, value)
    db.commit()
    db.refresh(p)
    return p


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_db)):
    """Delete a project and its owned resources; remove files after commit."""
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    datasets = db.query(Dataset).filter(Dataset.project_id == project_id).all()
    paths = []
    for dataset in datasets:
        if getattr(dataset, "storage_mode", "managed") != "linked":
            path = find_dataset_path(dataset, db)
            if path is not None:
                paths.append(path)
    # ORM cascades remove joined-inheritance subtype rows and curve children.
    db.delete(project)
    db.commit()
    for path in paths:
        path.unlink(missing_ok=True)
    return Response(status_code=204)

def survey_out(s):
    return {"id": s.id, "project_id": s.project_id, "name": s.name,
            "description": s.description, "crs": s.crs,
            "metadata": json.loads(s.metadata_json), "created_at": s.created_at}

@router.get("/surveys")
def list_surveys(project_id: str | None = None, db: Session = Depends(get_db)):
    if project_id and not db.get(Project, project_id):
        raise HTTPException(404, "Project not found")
    query = db.query(Survey)
    if project_id: query = query.filter(Survey.project_id == project_id)
    return [survey_out(s) for s in query.order_by(Survey.created_at.desc()).all()]

@router.get("/surveys/{survey_id}", response_model=SurveyOut)
def get_survey(survey_id: str, db: Session = Depends(get_db)):
    s = db.get(Survey, survey_id)
    if not s:
        raise HTTPException(404, "Survey not found")
    return survey_out(s)


@router.patch("/surveys/{survey_id}", response_model=SurveyOut)
def patch_survey(survey_id: str, payload: SurveyPatch, db: Session = Depends(get_db)):
    s = db.get(Survey, survey_id)
    if not s:
        raise HTTPException(404, "Survey not found")
    changes = payload.model_dump(exclude_unset=True)
    if "name" in changes and changes["name"] is None:
        raise HTTPException(422, "name cannot be null")
    for field, value in changes.items():
        if field == "metadata":
            s.metadata_json = json.dumps(value if value is not None else {})
        else:
            setattr(s, field, value)
    db.commit()
    db.refresh(s)
    return survey_out(s)


@router.delete("/surveys/{survey_id}", status_code=204)
def delete_survey(survey_id: str, db: Session = Depends(get_db)):
    """Delete a survey only when no datasets depend on it."""
    survey = db.get(Survey, survey_id)
    if not survey:
        raise HTTPException(404, "Survey not found")
    if db.query(SeismicDataset.id).filter(SeismicDataset.survey_id == survey_id).first():
        raise HTTPException(409, "Survey has seismic datasets attached; delete them first")
    db.delete(survey)
    db.commit()
    return Response(status_code=204)


@router.post("/projects/{project_id}/surveys", status_code=201)
def create_survey(
    project_id: str,
    payload: SurveyCreate,
    db: Session = Depends(get_db),
):
    if not db.get(Project, project_id):
        raise HTTPException(404, "Project not found")

    s = Survey(
        id=str(uuid.uuid4()),
        project_id=project_id,
        name=payload.name,
        description=payload.description,
        crs=payload.crs,
        metadata_json=json.dumps(payload.metadata),
    )

    db.add(s)
    db.commit()
    db.refresh(s)

    return {
        "id": s.id,
        "project_id": s.project_id,
        "name": s.name,
        "description": s.description,
        "crs": s.crs,
        "metadata": json.loads(s.metadata_json),
        "created_at": s.created_at,
    }

@router.get("/surveys/{survey_id}/datasets")
def list_survey_datasets(survey_id: str, db: Session = Depends(get_db)):
    if not db.get(Survey, survey_id): raise HTTPException(404, "Survey not found")
    return dataset_list(db.query(SeismicDataset).filter(SeismicDataset.survey_id == survey_id).all())

@router.post("/projects/{project_id}/wells", response_model=WellOut, status_code=201)
def create_well(project_id: str, payload: WellCreate, db: Session = Depends(get_db)):
    if not db.get(Project, project_id): raise HTTPException(404, "Project not found")
    w = Well(id=str(uuid.uuid4()), project_id=project_id, name=payload.name, uwi=payload.uwi,
             x=payload.x, y=payload.y, rt=payload.rt, td=payload.td, crs=payload.crs, metadata_json=json.dumps(payload.metadata))
    db.add(w); db.commit(); db.refresh(w)
    return {"id": w.id, "project_id": w.project_id, "name": w.name, "uwi": w.uwi, "x": w.x,
            "y": w.y, "rt": w.rt, "td": w.td, "crs": w.crs, "metadata": json.loads(w.metadata_json), "created_at": w.created_at}



@router.post("/projects/{project_id}/wells/import")
async def import_well_list(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Import/upsert well name, RT, TD, X and Y from an Excel well-list workbook."""
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    filename = (file.filename or "").lower()
    if not filename.endswith((".xlsx", ".xlsm")):
        raise HTTPException(422, "Well list must be an Excel .xlsx or .xlsm file")

    max_bytes = get_effective_settings(db).ingestion.max_upload_size_mb * 1024 * 1024
    try:
        from openpyxl import load_workbook
        raw = await file.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise HTTPException(413, "Well-list workbook exceeds the configured upload limit")
        workbook = load_workbook(BytesIO(raw), read_only=True, data_only=True)
        sheet = workbook.active
        required = {"well name": "name", "rt": "rt", "td": "td", "x": "x", "y": "y"}
        header_row = None
        columns = {}
        for row_number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            normalized = {str(value).strip().casefold(): index for index, value in enumerate(row) if value is not None}
            if all(key in normalized for key in required):
                header_row, columns = row_number, {field: normalized[label] for label, field in required.items()}
                break
            if row_number >= 50:
                break
        if header_row is None:
            raise HTTPException(422, "Could not find a header row containing Well Name, RT, TD, X and Y")
        parsed = []
        errors = []
        seen = set()
        for row_number, row in enumerate(sheet.iter_rows(min_row=header_row + 1, values_only=True), start=header_row + 1):
            values = {field: (row[index] if index < len(row) else None) for field, index in columns.items()}
            name = str(values["name"] or "").strip()
            if not name:
                continue
            key = name.casefold()
            if key in seen:
                errors.append({"row": row_number, "well": name, "error": "Duplicate well name in workbook"})
                continue
            seen.add(key)
            record = {"name": name}
            row_error = None
            for field in ("rt", "td", "x", "y"):
                value = values[field]
                if value is None or str(value).strip() == "":
                    record[field] = None
                    continue
                try:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError
                    record[field] = number
                except (TypeError, ValueError):
                    row_error = f"{field.upper()} must be a finite number"
                    break
            if row_error:
                errors.append({"row": row_number, "well": name, "error": row_error})
            else:
                parsed.append(record)
        workbook.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Unable to read Excel well list: {type(exc).__name__}: {exc}") from exc

    if not parsed:
        raise HTTPException(422, "No valid well rows were found in the workbook")

    # Persist the source workbook as a project-level table dataset.
    file.file.seek(0)
    dataset = await ingest_file(
        db, file, "table", project_id, None, None,
        file.filename, {"table_type": "well_list", "imported_columns": ["name", "rt", "td", "x", "y"]},
    )

    existing = {
        well.name.strip().casefold(): well
        for well in db.query(Well).filter(Well.project_id == project_id).all()
    }
    created = updated = 0
    for item in parsed:
        key = item["name"].casefold()
        well = existing.get(key)
        if well is None:
            well = Well(
                id=str(uuid.uuid4()), project_id=project_id, name=item["name"],
                rt=item["rt"], td=item["td"], x=item["x"], y=item["y"],
                crs=None, metadata_json=json.dumps({"well_list_dataset_id": dataset.id}),
            )
            db.add(well)
            existing[key] = well
            created += 1
        else:
            for field in ("rt", "td", "x", "y"):
                if item[field] is not None:
                    setattr(well, field, item[field])
            metadata = json.loads(well.metadata_json or "{}")
            metadata["well_list_dataset_id"] = dataset.id
            well.metadata_json = json.dumps(metadata)
            updated += 1
    db.commit()
    return {
        "dataset": dataset_out(dataset),
        "summary": {"rows_read": len(parsed) + len(errors), "created": created, "updated": updated,
                    "skipped": len(errors), "errors": errors},
    }


@router.get("/wells")
def list_wells(project_id: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Well)
    if project_id: query = query.filter(Well.project_id == project_id)
    return [well_out(w) for w in query.order_by(Well.created_at.desc()).all()]

@router.get("/wells/{well_id}")
def get_well(well_id: str, db: Session = Depends(get_db)):
    w = db.get(Well, well_id)
    if not w: raise HTTPException(404, "Well not found")
    return well_out(w)

@router.patch("/wells/{well_id}", response_model=WellOut)
def patch_well(well_id: str, payload: WellPatch, db: Session = Depends(get_db)):
    w = db.get(Well, well_id)
    if not w:
        raise HTTPException(404, "Well not found")
    changes = payload.model_dump(exclude_unset=True)
    if "name" in changes and changes["name"] is None:
        raise HTTPException(422, "name cannot be null")
    for field, value in changes.items():
        if field == "metadata":
            w.metadata_json = json.dumps(value if value is not None else {})
        else:
            setattr(w, field, value)
    db.commit()
    db.refresh(w)
    return well_out(w)


@router.delete("/wells/{well_id}", status_code=204)
def delete_well(well_id: str, db: Session = Depends(get_db)):
    """Delete a well only when no datasets depend on it."""
    well = db.get(Well, well_id)
    if not well:
        raise HTTPException(404, "Well not found")
    dependent = any((
        db.query(WellLogDataset.id).filter(WellLogDataset.well_id == well_id).first(),
        db.query(CheckshotDataset.id).filter(CheckshotDataset.well_id == well_id).first(),
        db.query(DeviationDataset.id).filter(DeviationDataset.well_id == well_id).first(),
    ))
    if dependent:
        raise HTTPException(409, "Well has datasets attached; delete them first")
    db.delete(well)
    db.commit()
    return Response(status_code=204)


def well_out(w):
    return {"id": w.id, "project_id": w.project_id, "name": w.name, "uwi": w.uwi, "x": w.x,
            "y": w.y, "rt": w.rt, "td": w.td, "crs": w.crs, "metadata": json.loads(w.metadata_json), "created_at": w.created_at}

def dataset_out(d):
    """Serialize the common dataset fields plus its subtype-specific association."""
    survey_id = d.survey_id if isinstance(d, SeismicDataset) else None
    well_id = d.well_id if isinstance(d, (WellLogDataset, CheckshotDataset, DeviationDataset)) else None
    status = d.status
    if getattr(d, "storage_mode", "managed") == "linked":
        try:
            from app.storage import validate_linked_path
            if not d.source_path:
                status = "unavailable"
            else:
                validate_linked_path(d.source_path)
        except HTTPException:
            status = "unavailable"
    return {
        "id": d.id, "project_id": d.project_id, "survey_id": survey_id, "well_id": well_id,
        "kind": d.kind, "dataset_type": type(d).__name__, "name": d.name,
        "original_filename": d.original_filename, "format": d.format, "storage_root": d.storage_root,
        "storage_mode": getattr(d, "storage_mode", "managed"),
        "source_path": getattr(d, "source_path", None), "crs": d.crs,
        "size_bytes": d.size_bytes, "checksum_sha256": d.checksum_sha256, "status": status,
        "metadata": json.loads(d.metadata_json), "validation": json.loads(d.validation_json),
        "created_at": d.created_at,
    }

def dataset_list(records):
    return {"count": len(records), "items": [dataset_out(d) for d in records]}

@router.post("/datasets", response_model=DatasetOut, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    kind: str = Form(...),
    project_id: str | None = Form(default=None),
    survey_id: str | None = Form(default=None),
    well_id: str | None = Form(default=None),
    name: str | None = Form(default=None),
    crs: str | None = Form(default=None),
    metadata_json: str = Form(default="{}"),
    db: Session = Depends(get_db),
):
    try:
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict): raise ValueError("must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(422, f"metadata_json must be a JSON object: {exc}") from exc
    record = await ingest_file(db, file, kind, project_id, survey_id, well_id, name, metadata, crs)
    return dataset_out(record)

@router.get("/datasets/link-local/browse")
def browse_link_local_directory(path: str | None = Query(default=None)):
    """Browse immediate children inside configured server-side link roots."""
    roots = allowed_link_roots()
    if path is None or not path.strip():
        return {
            "current_path": None,
            "roots": [{"name": root.name or str(root), "path": str(root)} for root in roots if root.is_dir()],
            "entries": [],
            "parent_path": None,
        }

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise HTTPException(422, "Directory path must be absolute")
    try:
        directory = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(404, "Directory is unavailable") from exc
    if not directory.is_dir():
        raise HTTPException(422, "Selected path is not a directory")
    root = next((item for item in roots if directory == item or item in directory.parents), None)
    if root is None:
        raise HTTPException(403, "Directory is outside configured link roots")

    entries = []
    try:
        children = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        for child in children:
            try:
                resolved = child.resolve(strict=True)
                if root != resolved and root not in resolved.parents:
                    continue
                if child.is_dir():
                    entries.append({"name": child.name, "path": str(resolved), "kind": "directory"})
                elif child.is_file() and child.suffix.lower() in {".sgy", ".segy", ".las", ".txt"} and os.access(child, os.R_OK):
                    entries.append({"name": child.name, "path": str(resolved), "kind": "file", "extension": child.suffix.lower()})
            except OSError:
                continue
    except PermissionError as exc:
        raise HTTPException(403, "Permission denied while browsing this directory") from exc

    parent = directory.parent if directory != root else None
    return {
        "current_path": str(directory),
        "roots": [{"name": item.name or str(item), "path": str(item)} for item in roots if item.is_dir()],
        "parent_path": str(parent) if parent else None,
        "entries": entries,
    }


@router.post("/datasets/link-local", response_model=DatasetOut, status_code=201)
def link_local_dataset(payload: DatasetLinkLocal, db: Session = Depends(get_db)):
    """Register an existing server-side dataset without copying its contents."""
    record = register_linked_file(
        db, source_path=payload.source_path, kind=payload.kind,
        project_id=payload.project_id, survey_id=payload.survey_id,
        well_id=payload.well_id, name=payload.name,
        metadata=payload.metadata, crs=payload.crs,
    )
    return dataset_out(record)


@router.get("/datasets")
def list_datasets(
    kind: str | None = None, project_id: str | None = None, survey_id: str | None = None,
    well_id: str | None = None, limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0), db: Session = Depends(get_db)
):
    if survey_id and well_id:
        raise HTTPException(422, "survey_id and well_id filters are mutually exclusive")
    query = db.query(Dataset)
    if kind: query = query.filter(Dataset.kind == kind)
    if project_id: query = query.filter(Dataset.project_id == project_id)
    if survey_id:
        query = query.join(SeismicDataset, SeismicDataset.id == Dataset.id).filter(SeismicDataset.survey_id == survey_id)
    if well_id:
        query = query.filter(Dataset.id.in_(
            db.query(WellLogDataset.id).filter(WellLogDataset.well_id == well_id).union(
                db.query(CheckshotDataset.id).filter(CheckshotDataset.well_id == well_id)
            ).union(db.query(DeviationDataset.id).filter(DeviationDataset.well_id == well_id))
        ))
    total = query.count()
    records = query.order_by(Dataset.created_at.desc()).offset(offset).limit(limit).all()
    return {"count": total, "limit": limit, "offset": offset, "items": [dataset_out(d) for d in records]}

@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    d = db.get(Dataset, dataset_id)
    if not d: raise HTTPException(404, "Dataset not found")
    return dataset_out(d)

@router.patch("/datasets/{dataset_id}", response_model=DatasetOut)
def patch_dataset(dataset_id: str, payload: DatasetPatch, db: Session = Depends(get_db)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    changes = payload.model_dump(exclude_unset=True)
    if "name" in changes and changes["name"] is None:
        raise HTTPException(422, "name cannot be null")

    if "survey_id" in changes or "well_id" in changes:
        if isinstance(dataset, SeismicDataset):
            if "well_id" in changes:
                raise HTTPException(422, "Seismic datasets cannot have well_id; use survey_id")
            survey_id = changes.get("survey_id")
            survey = db.get(Survey, survey_id) if survey_id else None
            if survey_id and survey is None:
                raise HTTPException(404, "Survey not found")
            if survey and survey.project_id != dataset.project_id:
                raise HTTPException(422, "Survey does not belong to dataset project")
            dataset.survey_id = survey_id
        else:
            if "survey_id" in changes:
                raise HTTPException(422, "Well-associated datasets cannot have survey_id; survey-level activities belong to Project")
            if not isinstance(dataset, (WellLogDataset, CheckshotDataset, DeviationDataset)):
                raise HTTPException(422, "Unsupported dataset subtype")
            well_id = changes.get("well_id")
            well = db.get(Well, well_id) if well_id else None
            if well_id and well is None:
                raise HTTPException(404, "Well not found")
            if well is None:
                raise HTTPException(422, f"{dataset.kind} datasets require well_id")
            if well.project_id != dataset.project_id:
                raise HTTPException(422, "Well does not belong to dataset project")
            dataset.well_id = well_id

    for field, value in changes.items():
        if field in {"survey_id", "well_id"}:
            continue
        if field == "metadata":
            dataset.metadata_json = json.dumps(value if value is not None else {})
        else:
            setattr(dataset, field, value)
    db.commit()
    db.refresh(dataset)
    return dataset_out(dataset)


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a dataset registration; only managed files are physically removed."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    safe_path = None
    if getattr(dataset, "storage_mode", "managed") != "linked":
        safe_path = find_dataset_path(dataset, db)

    db.query(Curve).filter(Curve.dataset_id == dataset_id).delete(synchronize_session=False)
    db.delete(dataset)
    db.commit()

    if safe_path is not None:
        safe_path.unlink(missing_ok=True)
    return Response(status_code=204)


@router.get("/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str, db: Session = Depends(get_db)):
    d = db.get(Dataset, dataset_id)
    if not d: raise HTTPException(404, "Dataset not found")
    path = resolve_dataset_path(d, db)
    return FileResponse(path, filename=d.original_filename, media_type="application/octet-stream")


@router.get("/datasets/{dataset_id}/file-info")
def dataset_file_info(dataset_id: str, db: Session = Depends(get_db)):
    """Confirm the source file exists and return its stored-file identity."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    path = resolve_dataset_path(dataset, db)
    return {
        "dataset_id": dataset.id,
        "exists": True,
        "filename": dataset.original_filename,
        "format": dataset.format,
        "size_bytes": path.stat().st_size,
        "checksum_sha256": dataset.checksum_sha256,
        "download_path": f"/api/v1/datasets/{dataset.id}/download",
    }


def curve_out(curve: Curve) -> dict:
    """Serialize a catalogued curve definition."""
    return {
        "id": curve.id,
        "dataset_id": curve.dataset_id,
        "mnemonic": curve.mnemonic,
        "unit": curve.unit,
        "description": curve.description,
        "position": curve.position,
        "is_index": curve.is_index,
        "metadata": json.loads(curve.metadata_json),
        "created_at": curve.created_at,
    }


@router.get("/datasets/{dataset_id}/curves")
def list_dataset_curves(dataset_id: str, db: Session = Depends(get_db)):
    """List curve definitions extracted from a dataset, in source-file order."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    if not isinstance(dataset, WellLogDataset):
        raise HTTPException(422, "Curve cataloguing currently supports LAS datasets only")
    curves = (
        db.query(Curve)
        .filter(Curve.dataset_id == dataset_id)
        .order_by(Curve.position)
        .all()
    )
    return {"dataset_id": dataset_id, "count": len(curves), "items": [curve_out(c) for c in curves]}


@router.get("/curves/{curve_id}")
def get_curve(curve_id: str, db: Session = Depends(get_db)):
    """Retrieve one curve definition by its stable identifier."""
    curve = db.get(Curve, curve_id)
    if not curve:
        raise HTTPException(404, "Curve not found")
    return curve_out(curve)


@router.get("/curves/{curve_id}/data")
def get_curve_data(
    curve_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=10000),
    index_min: float | None = Query(default=None),
    index_max: float | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Return a bounded page of numerical samples for a catalogued LAS curve."""
    curve = db.get(Curve, curve_id)
    if not curve:
        raise HTTPException(status_code=404, detail="Curve not found")
    dataset = db.get(Dataset, curve.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Source dataset not found")
    if not isinstance(dataset, WellLogDataset):
        raise HTTPException(
            status_code=422,
            detail="Curve sample retrieval currently supports LAS datasets only",
        )
    if index_min is not None and index_max is not None and index_min > index_max:
        raise HTTPException(
            status_code=422,
            detail="index_min must be less than or equal to index_max",
        )

    path = resolve_dataset_path(dataset, db)
    try:
        return read_curve_samples(
            path,
            curve,
            offset=offset,
            limit=limit,
            index_min=index_min,
            index_max=index_max,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/data")
def dataset_data(
    dataset_id: str,
    max_traces: int = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    """Return the frontend data payload for a dataset, parsed from its stored source file.

    SEG-Y responses include up to ``max_traces`` traces (default 500); traceCount
    reports the total number of traces and scannedTraceCount reports the number
    included in the response. LAS and table payloads are returned in full within
    parser safety limits.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    path = resolve_dataset_path(dataset, db)
    try:
        if dataset.kind == "seismic":
            return {"dataset_id": dataset.id, **read_seismic_data(path, max_traces=max_traces)}
        if dataset.kind == "well_log":
            return {"dataset_id": dataset.id, **read_well_data(path)}
        if dataset.kind in {"deviation", "checkshot"}:
            return {"dataset_id": dataset.id, **read_table_data(path, dataset.kind)}
        if dataset.kind == "table" and dataset.format in {"xlsx", "xlsm"}:
            return {"dataset_id": dataset.id, **read_excel_well_list(path)}
        raise HTTPException(status_code=422, detail=f"Unsupported dataset kind: {dataset.kind}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/seismic/geometry")
def seismic_geometry(
    dataset_id: str,
    inline_byte: int | None = Query(default=None, ge=1, le=240),
    crossline_byte: int | None = Query(default=None, ge=1, le=240),
    db: Session = Depends(get_db),
):
    """Return full-volume geometry metadata and ranked inline/crossline byte mappings."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.kind != "seismic":
        raise HTTPException(status_code=422, detail="Dataset is not seismic")
    path = resolve_dataset_path(dataset, db)
    try:
        return {"dataset_id": dataset.id, **geometry_metadata(path, inline_byte, crossline_byte)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/seismic/section")
def seismic_section(
    dataset_id: str,
    mode: str = Query(..., pattern="^(inline|crossline|timeslice)$"),
    target: int = Query(..., ge=0),
    inline_byte: int = Query(default=189, ge=1, le=240),
    crossline_byte: int = Query(default=193, ge=1, le=240),
    db: Session = Depends(get_db),
):
    """Retrieve one indexed inline/crossline or a bounded time-slice grid."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.kind != "seismic":
        raise HTTPException(status_code=422, detail="Dataset is not seismic")
    path = resolve_dataset_path(dataset, db)
    try:
        return {"dataset_id": dataset.id, **read_seismic_section(path, mode, target, inline_byte, crossline_byte)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/header")
def dataset_header(dataset_id: str, db: Session = Depends(get_db)):
    """Read SEG-Y textual/binary headers or LAS well/curve headers."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.format.lower() not in {"sgy", "segy", "las"}:
        raise HTTPException(
            status_code=422,
            detail="Header parsing currently supports SEG-Y (.sgy/.segy) and LAS (.las) files only",
        )
    path = resolve_dataset_path(dataset, db)
    try:
        parsed = read_file_header(path, dataset.format)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "dataset_id": dataset.id,
        "filename": dataset.original_filename,
        "stored_format": dataset.format,
        "stored_crs": dataset.crs,
        "header": parsed,
    }


@router.get("/datasets/{dataset_id}/table")
def dataset_table(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=10000),
    db: Session = Depends(get_db),
):
    """Return paginated parsed rows for deviation/checkshot text datasets."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.kind not in {"deviation", "checkshot"}:
        raise HTTPException(
            status_code=422,
            detail="Tabular data retrieval currently supports deviation and checkshot datasets only",
        )
    if dataset.format.lower().lstrip(".") != "txt":
        raise HTTPException(status_code=422, detail="Deviation and checkshot data must be stored as .txt")
    try:
        parsed = read_well_table(
            resolve_dataset_path(dataset, db), dataset.kind, offset=offset, limit=limit
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"dataset_id": dataset.id, **parsed}


@router.get("/datasets/{dataset_id}/validation")
def dataset_validation(dataset_id: str, db: Session = Depends(get_db)):
    d = db.get(Dataset, dataset_id)
    if not d: raise HTTPException(404, "Dataset not found")
    return {"dataset_id": d.id, "status": d.status, "validation": json.loads(d.validation_json)}
