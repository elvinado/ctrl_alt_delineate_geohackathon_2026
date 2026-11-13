"""Dataset ingestion and safe file resolution."""
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from fastapi import HTTPException, UploadFile
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import (CheckshotDataset, Dataset, DeviationDataset, Project, SeismicDataset,
                        Survey, Well, WellLogDataset, Curve, TableDataset)
from app.runtime_settings import get_effective_settings
from app.file_headers import read_las_header
from app.well_tables import read_well_table

class LinkedDatasetSettings(BaseSettings):
    """Infrastructure settings for linked dataset access.

    Read the backend .env file as well as process environment variables. Values
    exported by the process environment take precedence, matching the other
    bootstrap settings in the application.
    """

    dataset_allowed_roots: str = ""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


KIND_CONFIG = {
    "seismic": {"formats": {"sgy", "segy"}, "model": SeismicDataset, "association": "survey"},
    "well_log": {"formats": {"las"}, "model": WellLogDataset, "association": "well"},
    "checkshot": {"formats": {"txt"}, "model": CheckshotDataset, "association": "well"},
    "deviation": {"formats": {"txt"}, "model": DeviationDataset, "association": "well"},
    "table": {"formats": {"xlsx", "xlsm"}, "model": TableDataset, "association": None},
}


def _sync_default_scripts(storage_root: Path) -> None:
    """Refresh bundled scripts without deleting user-added runtime files.

    ``backend/scripts`` is the source of truth for shipped defaults. Matching
    files are refreshed on every storage initialization; files that exist only
    in ``site_data/scripts`` are preserved.
    """
    source = Path(__file__).resolve().parent.parent / "scripts"
    target = storage_root / "scripts"
    if not source.is_dir():
        return

    target.mkdir(parents=True, exist_ok=True)
    resolved_target = target.resolve()
    for source_path in source.rglob("*"):
        if source_path.is_symlink() or not source_path.is_file():
            continue

        relative_path = source_path.relative_to(source)
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        # These stage YAML files are user-editable runtime configuration. Seed
        # them on first initialization, but never overwrite saved UI edits.
        if destination.name in {
            "A_foundations.yaml", "B_inference.yaml", "C_volume_products.yaml",
            "D_geobody.yaml", "E_well_geobody_analysis.yaml", "F_prospect.yaml",
        } and destination.exists():
            continue

        # Never follow a runtime symlink outside the scripts directory.
        if destination.is_symlink():
            raise OSError(f"Refusing to overwrite symlink in scripts directory: {destination}")
        try:
            destination.parent.resolve().relative_to(resolved_target)
        except ValueError as exc:
            raise OSError(f"Invalid default script destination: {destination}") from exc

        # Replace atomically so a running API never observes a partially copied file.
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                with source_path.open("rb") as source_file:
                    shutil.copyfileobj(source_file, temporary)
            shutil.copystat(source_path, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def ensure_storage_root(db: Session | None = None) -> Path:
    """Return the first configured storage root that can be created and written to.

    The primary root is preferred. If it is unavailable (for example, a home
    directory is not mounted on a compute node), the configured fallback is used.
    Required application directories (uploads and geochat) are initialized in
    that root. Bundled scripts are refreshed on each initialization while
    runtime-only files are retained.
    """
    owns_session = db is None
    session = db or SessionLocal()
    try:
        storage = get_effective_settings(session).storage
        candidates = [Path(storage.data_root).expanduser(), Path(storage.fallback_data_root).expanduser()]
        seen: set[Path] = set()
        errors: list[str] = []
        for candidate in candidates:
            root = candidate.resolve()
            if root in seen:
                continue
            seen.add(root)
            try:
                # Initialize persistent application directories for every mode.
                # GeoChat transcripts must exist even before the first chat request.
                (root / "uploads").mkdir(parents=True, exist_ok=True)
                (root / "geochat").mkdir(parents=True, exist_ok=True)
                # Check actual write access, not only whether the directory exists.
                probe = root / f".storage-write-check-{uuid.uuid4().hex}"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                _sync_default_scripts(root)
                return root
            except OSError as exc:
                errors.append(f"{root}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Neither configured storage path is writable. " + "; ".join(errors),
        )
    finally:
        if owns_session:
            session.close()


def safe_name(name: str | None) -> str:
    return Path((name or "upload.bin").replace("\x00", "")).name or "upload.bin"


def configured_storage_roots(db: Session | None = None) -> list[Path]:
    """Return configured roots in preference order without requiring them to exist."""
    owns_session = db is None
    session = db or SessionLocal()
    try:
        storage = get_effective_settings(session).storage
        roots = [Path(storage.data_root).expanduser().resolve(), Path(storage.fallback_data_root).expanduser().resolve()]
        return list(dict.fromkeys(roots))
    finally:
        if owns_session:
            session.close()


def find_dataset_path(dataset: Dataset, db: Session | None = None) -> Path | None:
    """Find a dataset file in either configured root, including a previously used fallback."""
    recorded_root = Path(dataset.storage_root).expanduser().resolve()
    roots = [recorded_root, *configured_storage_roots(db)]
    for root in dict.fromkeys(roots):
        path = (root / dataset.storage_key).resolve()
        if root not in path.parents:
            raise HTTPException(500, "Invalid stored path")
        if path.is_file():
            return path
    return None


def allowed_link_roots() -> list[Path]:
    """Return explicitly configured shared-filesystem roots for linked datasets.

    Linked paths are deployment-specific and must never be inferred from a
    developer home directory or a guessed HPC mount. Configure the same
    DATASET_ALLOWED_ROOTS on API and worker nodes when using linked datasets.
    Prefer managed uploads for files that must be portable across environments.
    """
    configured = LinkedDatasetSettings().dataset_allowed_roots
    raw_roots = [value.strip() for value in configured.split(os.pathsep) if value.strip()]
    return list(dict.fromkeys(Path(value).expanduser().resolve() for value in raw_roots))


def validate_linked_path(raw_path: str, *, require_file: bool = True) -> Path:
    """Resolve a linked file and enforce the configured input_data boundary."""
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        raise HTTPException(422, "Linked dataset path must be absolute")
    try:
        path = candidate.resolve(strict=require_file)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(404, f"Linked dataset file is unavailable: {candidate}") from exc
    roots = allowed_link_roots()
    if not any(path == root or root in path.parents for root in roots):
        raise HTTPException(403, "Linked dataset must be inside a configured input_data directory")
    if require_file and (not path.is_file() or not os.access(path, os.R_OK)):
        raise HTTPException(422, "Linked dataset must be a readable regular file")
    return path


def resolve_dataset_path(dataset: Dataset, db: Session | None = None) -> Path:
    """Resolve linked sources directly or managed files from application storage."""
    if getattr(dataset, "storage_mode", "managed") == "linked":
        if not dataset.source_path:
            raise HTTPException(404, "Linked dataset has no source path")
        return validate_linked_path(dataset.source_path)
    path = find_dataset_path(dataset, db)
    if path is None:
        raise HTTPException(404, "Stored dataset file is missing from both configured storage paths")
    return path


def _file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def register_linked_file(db: Session, *, source_path: str, kind: str, project_id: str,
                         survey_id: str | None = None, well_id: str | None = None,
                         name: str | None = None, metadata: dict | None = None,
                         crs: str | None = None) -> Dataset:
    """Register an existing server-side file without copying or owning it."""
    if kind not in KIND_CONFIG:
        raise HTTPException(422, f"Unsupported dataset kind. Allowed: {', '.join(KIND_CONFIG)}")
    path = validate_linked_path(source_path)
    project_id, survey, well = validate_associations(db, kind, project_id, survey_id, well_id)
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in KIND_CONFIG[kind]["formats"]:
        expected = ", ".join(f".{value}" for value in sorted(KIND_CONFIG[kind]["formats"]))
        raise HTTPException(422, f"Dataset kind '{kind}' requires format {expected}; received '.{suffix}'")

    size, checksum = _file_sha256(path)
    dataset_id = str(uuid.uuid4())
    original = safe_name(path.name)
    subtype_association = ({"survey_id": survey_id} if kind == "seismic" else
                           {"well_id": well_id} if KIND_CONFIG[kind]["association"] == "well" else {})
    record = KIND_CONFIG[kind]["model"](
        id=dataset_id, project_id=project_id, **subtype_association,
        kind=kind, name=(name or original)[:255], original_filename=original[:255],
        format=suffix, crs=(crs or (survey.crs if survey else None) or (well.crs if well else None)),
        storage_root=str(path.parent), storage_mode="linked", source_path=str(path),
        storage_key=f"linked/{dataset_id}.{suffix}", size_bytes=size,
        checksum_sha256=checksum, status="validating",
        metadata_json=json.dumps(metadata or {}), validation_json="{}",
    )
    db.add(record)
    try:
        db.flush()
        validation = {"checks": ["extension_matches_kind", "non_empty", "source_file_linked", "allowed_root"], "warnings": []}
        if size == 0:
            raise HTTPException(422, "Empty files are not accepted")
        if kind == "well_log":
            try:
                header = read_las_header(path)
            except ValueError as exc:
                raise HTTPException(422, f"Unable to parse LAS file: {exc}") from exc
            index_mnemonic = str(header.get("index_mnemonic") or "").strip().casefold()
            for position, info in enumerate(header.get("curves", [])):
                mnemonic = str(info.get("mnemonic") or "").strip()
                if mnemonic:
                    db.add(Curve(id=str(uuid.uuid4()), dataset_id=record.id, mnemonic=mnemonic,
                        unit=(str(info.get("unit") or "").strip() or None),
                        description=(str(info.get("description") or "").strip() or None),
                        position=position, is_index=mnemonic.casefold() == index_mnemonic,
                        metadata_json=json.dumps({"header_value": info.get("value")})))
            validation["checks"].append("las_header_parsed")
        elif kind in {"deviation", "checkshot"}:
            try:
                parsed = read_well_table(path, kind, offset=0, limit=1)
            except ValueError as exc:
                raise HTTPException(422, f"Unable to parse {kind} file: {exc}") from exc
            record.metadata_json = json.dumps({**(metadata or {}), "parsed_table": {
                "columns": parsed["columns"], "row_count": parsed["row_count"], "parser": "well_table_v1"}})
            validation["checks"].append("table_parsed")
        record.status = "ready"
        record.validation_json = json.dumps(validation)
        db.commit()
        db.refresh(record)
        return record
    except Exception:
        db.rollback()
        raise


def validate_associations(db: Session, kind: str, project_id: str | None,
                          survey_id: str | None, well_id: str | None) -> tuple[str, Survey | None, Well | None]:
    """Validate only the association allowed by the concrete dataset subtype."""
    if kind not in KIND_CONFIG:
        raise HTTPException(422, f"Unsupported dataset kind. Allowed: {', '.join(KIND_CONFIG)}")

    required = KIND_CONFIG[kind]["association"]
    if required is None and (survey_id or well_id):
        raise HTTPException(422, "Table datasets are project-level and do not accept survey_id or well_id")
    if required == "survey" and well_id:
        raise HTTPException(422, "Seismic datasets use survey_id, not well_id")
    if required == "well" and survey_id:
        raise HTTPException(422, f"{kind} datasets use well_id; survey association belongs at project level")

    survey = db.get(Survey, survey_id) if survey_id else None
    well = db.get(Well, well_id) if well_id else None
    if survey_id and survey is None:
        raise HTTPException(404, "Survey not found")
    if well_id and well is None:
        raise HTTPException(404, "Well not found")

    inferred_project = survey.project_id if survey else (well.project_id if well else None)
    project_id = project_id or inferred_project
    if not project_id:
        raise HTTPException(422, "project_id is required unless inferable from survey_id or well_id")
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    if survey and survey.project_id != project_id:
        raise HTTPException(422, "Survey does not belong to project")
    if well and well.project_id != project_id:
        raise HTTPException(422, "Well does not belong to project")
    return project_id, survey, well


async def ingest_file(db: Session, file: UploadFile, kind: str, project_id: str | None,
                       survey_id: str | None, well_id: str | None, name: str | None,
                       metadata: dict, crs: str | None = None) -> Dataset:
    if kind not in KIND_CONFIG:
        raise HTTPException(422, f"Unsupported dataset kind. Allowed: {', '.join(KIND_CONFIG)}")
    project_id, survey, well = validate_associations(db, kind, project_id, survey_id, well_id)
    original = safe_name(file.filename)
    suffix = Path(original).suffix.lower().lstrip(".")
    if suffix not in KIND_CONFIG[kind]["formats"]:
        expected = ", ".join(f".{x}" for x in sorted(KIND_CONFIG[kind]["formats"]))
        raise HTTPException(422, f"Dataset kind '{kind}' requires format {expected}; received '.{suffix}'")

    config = get_effective_settings(db)
    max_bytes = config.ingestion.max_upload_size_mb * 1024 * 1024
    dataset_id = str(uuid.uuid4())
    root = ensure_storage_root(db)
    destination = root / "uploads" / f"{dataset_id}.{suffix}"
    digest = hashlib.sha256()
    size = 0
    try:
        with destination.open("xb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"File exceeds configured limit of {config.ingestion.max_upload_size_mb} MB")
                digest.update(chunk)
                output.write(chunk)
        if size == 0:
            raise HTTPException(422, "Empty files are not accepted")

        kind_model = KIND_CONFIG[kind]["model"]
        subtype_association = ({"survey_id": survey_id} if kind == "seismic" else
                           {"well_id": well_id} if KIND_CONFIG[kind]["association"] == "well" else {})
        record = kind_model(
            id=dataset_id, project_id=project_id,
            **subtype_association,
            kind=kind, name=(name or original)[:255], original_filename=original[:255],
            format=suffix, crs=(crs or (survey.crs if survey else None) or (well.crs if well else None)),
            storage_root=str(root.resolve()), storage_key=str(Path("uploads") / destination.name), size_bytes=size,
            checksum_sha256=digest.hexdigest(), status="validating",
            metadata_json=json.dumps(metadata), validation_json="{}",
        )
        db.add(record)
        db.flush()

        validation = {"checks": ["extension_matches_kind", "non_empty", "source_file_stored"], "warnings": []}
        if kind == "well_log":
            try:
                header = read_las_header(destination)
            except ValueError as exc:
                raise HTTPException(422, f"Unable to parse LAS file: {exc}") from exc
            index_mnemonic = str(header.get("index_mnemonic") or "").strip().casefold()
            for position, info in enumerate(header.get("curves", [])):
                mnemonic = str(info.get("mnemonic") or "").strip()
                if not mnemonic:
                    continue
                db.add(Curve(
                    id=str(uuid.uuid4()), dataset_id=record.id, mnemonic=mnemonic,
                    unit=(str(info.get("unit") or "").strip() or None),
                    description=(str(info.get("description") or "").strip() or None),
                    position=position, is_index=mnemonic.casefold() == index_mnemonic,
                    metadata_json=json.dumps({"header_value": info.get("value")}),
                ))
            validation["checks"].append("las_header_parsed")
        elif kind in {"deviation", "checkshot"}:
            try:
                parsed = read_well_table(destination, kind, offset=0, limit=1)
            except ValueError as exc:
                raise HTTPException(422, f"Unable to parse {kind} file: {exc}") from exc
            record.metadata_json = json.dumps({**metadata, "parsed_table": {
                "columns": parsed["columns"], "row_count": parsed["row_count"], "parser": "well_table_v1"
            }})
            validation["checks"].append("table_parsed")
        # SEG-Y binary/text header validation is intentionally available through /header;
        # ingestion records the file safely without forcing full-volume reads.
        record.status = "ready"
        record.validation_json = json.dumps(validation)
        db.commit()
        db.refresh(record)
        return record
    except Exception:
        db.rollback()
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
