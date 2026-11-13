"""CRUD API for user-managed Python processing scripts in the active data root."""
from datetime import datetime, timezone
import os
import re
import hashlib
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.storage import ensure_storage_root

router = APIRouter()
SCRIPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.py$")
MAX_SCRIPT_BYTES = 2 * 1024 * 1024


class ScriptUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(..., description="Complete UTF-8 Python source code")


def _scripts_dir(db: Session) -> Path:
    """Resolve scripts under the currently active configured storage root."""
    return ensure_storage_root(db) / "scripts"


def _script_path(db: Session, script_id: str) -> Path:
    if not SCRIPT_ID_RE.fullmatch(script_id) or script_id in {".py", "..py"}:
        raise HTTPException(status_code=422, detail="script_id must be a valid .py filename")
    root = _scripts_dir(db).resolve()
    path = (root / script_id).resolve()
    if root not in path.parents:
        raise HTTPException(status_code=422, detail="Invalid script path")
    return path


def _script_info(path: Path) -> dict:
    stat = path.stat()
    return {
        "id": path.name,
        "name": path.name,
        "size_bytes": stat.st_size,
        "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    }


@router.get("/scripts", summary="List available Python processing scripts")
def list_scripts(db: Session = Depends(get_db)):
    scripts_dir = _scripts_dir(db)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    scripts = [
        _script_info(path)
        for path in scripts_dir.iterdir()
        if path.is_file() and path.suffix == ".py" and SCRIPT_ID_RE.fullmatch(path.name)
    ]
    return {"items": sorted(scripts, key=lambda item: item["name"].lower()), "count": len(scripts)}


@router.get("/scripts/{script_id}", summary="Read a processing script")
def get_script(script_id: str, db: Session = Depends(get_db)):
    path = _script_path(db, script_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Processing script not found")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Script is not valid UTF-8 text") from exc
    return {**_script_info(path), "content": content}


@router.post("/scripts", status_code=status.HTTP_201_CREATED, summary="Upload a processing script")
async def upload_script(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = Path(file.filename or "").name
    path = _script_path(db, filename)
    if not filename.endswith(".py") or not SCRIPT_ID_RE.fullmatch(filename):
        raise HTTPException(status_code=422, detail="Upload must have a valid .py filename")
    root = _scripts_dir(db).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise HTTPException(status_code=409, detail="A script with this filename already exists")
    content = await file.read(MAX_SCRIPT_BYTES + 1)
    if len(content) > MAX_SCRIPT_BYTES:
        raise HTTPException(status_code=413, detail=f"Script exceeds {MAX_SCRIPT_BYTES} bytes")
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Script must be UTF-8 text") from exc
    # Exclusive create prevents accidentally overwriting a concurrently uploaded script.
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="A script with this filename already exists") from exc
    return _script_info(path)


@router.put("/scripts/{script_id}", summary="Replace a processing script's contents")
def update_script(script_id: str, payload: ScriptUpdate, db: Session = Depends(get_db)):
    path = _script_path(db, script_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Processing script not found")
    content = payload.content.encode("utf-8")
    if len(content) > MAX_SCRIPT_BYTES:
        raise HTTPException(status_code=413, detail=f"Script exceeds {MAX_SCRIPT_BYTES} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**_script_info(path), "content": payload.content}


@router.delete("/scripts/{script_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a processing script")
def delete_script(script_id: str, db: Session = Depends(get_db)):
    path = _script_path(db, script_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Processing script not found")
    path.unlink()
    return None


# Only these shipped GeoGG stage configurations are editable through the UI.
PROCESSING_CONFIGS = {
    "A_foundations": "A_foundations.yaml",
    "B_inference": "B_inference.yaml",
    "C_volume_products": "C_volume_products.yaml",
    "D_geobody": "D_geobody.yaml",
    "E_well_geobody_analysis": "E_well_geobody_analysis.yaml",
    "F_prospect": "F_prospect.yaml",
}
CONFIG_ROOT_KEYS = {
    "A_foundations": None,
    "B_inference": "inference",
    "C_volume_products": "volume_products",
    "D_geobody": "geobody",
    "E_well_geobody_analysis": "well_geobody_analysis",
    "F_prospect": "prospect",
}


class ProcessingConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: dict
    expected_revision: str = Field(..., min_length=64, max_length=64)


def _config_path(db: Session, config_id: str) -> Path:
    filename = PROCESSING_CONFIGS.get(config_id)
    if not filename:
        raise HTTPException(status_code=404, detail="Unknown processing configuration")
    root = (_scripts_dir(db)).resolve()
    candidate = root / filename
    if candidate.is_symlink():
        raise HTTPException(status_code=422, detail="Refusing to edit a symlinked processing configuration")
    path = candidate.resolve()
    if root not in path.parents:
        raise HTTPException(status_code=422, detail="Invalid processing configuration path")
    return path


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="PyYAML is required by the backend to edit processing configurations") from exc
    return yaml


def _read_processing_config(path: Path, config_id: str) -> tuple[str, dict, str]:
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Processing configuration file not found")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Configuration is not valid UTF-8 text") from exc
    yaml = _yaml_module()
    try:
        config = yaml.safe_load(content) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"Configuration YAML is invalid: {exc}") from exc
    if not isinstance(config, dict):
        raise HTTPException(status_code=422, detail="Configuration YAML root must be a mapping")
    required_root = CONFIG_ROOT_KEYS[config_id]
    if required_root and not isinstance(config.get(required_root), dict):
        raise HTTPException(status_code=422, detail=f"Configuration must contain a '{required_root}' mapping")
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Configuration contains values that cannot be represented safely as JSON: {exc}") from exc
    revision = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, config, revision


@router.get("/processing-configs", summary="List editable GeoGG stage configurations")
def list_processing_configs(db: Session = Depends(get_db)):
    scripts_dir = _scripts_dir(db)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for config_id, filename in PROCESSING_CONFIGS.items():
        path = _config_path(db, config_id)
        if path.is_file():
            _, _, revision = _read_processing_config(path, config_id)
            items.append({"id": config_id, "name": filename, "revision": revision})
    return {"items": items, "count": len(items)}


@router.get("/processing-configs/{config_id}", summary="Read a GeoGG stage configuration")
def get_processing_config(config_id: str, db: Session = Depends(get_db)):
    path = _config_path(db, config_id)
    content, config, revision = _read_processing_config(path, config_id)
    return {"id": config_id, "name": path.name, "content": content, "config": config, "revision": revision}


@router.put("/processing-configs/{config_id}", summary="Save a GeoGG stage configuration")
def update_processing_config(config_id: str, payload: ProcessingConfigUpdate, db: Session = Depends(get_db)):
    path = _config_path(db, config_id)
    _, _, current_revision = _read_processing_config(path, config_id)
    if current_revision != payload.expected_revision:
        raise HTTPException(status_code=409, detail="Configuration changed on the backend since it was loaded. Reload before saving.")
    required_root = CONFIG_ROOT_KEYS[config_id]
    if required_root and not isinstance(payload.config.get(required_root), dict):
        raise HTTPException(status_code=422, detail=f"Configuration must contain a '{required_root}' mapping")
    yaml = _yaml_module()
    try:
        # Round-trip the existing document so explanatory YAML comments and
        # formatting survive UI edits. PyYAML remains the safe validation gate.
        try:
            from ruamel.yaml import YAML
            from ruamel.yaml.comments import CommentedMap, CommentedSeq
            round_trip = YAML()
            round_trip.preserve_quotes = True
            with path.open("r", encoding="utf-8") as source:
                document = round_trip.load(source) or CommentedMap()

            def merge_yaml(existing, updated):
                if isinstance(updated, dict):
                    if not isinstance(existing, dict):
                        existing = CommentedMap()
                    for old_key in list(existing):
                        if old_key not in updated:
                            del existing[old_key]
                    for key, value in updated.items():
                        existing[key] = merge_yaml(existing.get(key), value)
                    return existing
                if isinstance(updated, list):
                    if isinstance(existing, list):
                        existing[:] = [merge_yaml(existing[index] if index < len(existing) else None, value)
                                        for index, value in enumerate(updated)]
                        return existing
                    return CommentedSeq(updated)
                return updated

            document = merge_yaml(document, payload.config)
            from io import StringIO
            buffer = StringIO()
            round_trip.dump(document, buffer)
            content = buffer.getvalue()
        except ImportError:
            content = yaml.safe_dump(payload.config, sort_keys=False, allow_unicode=True, default_flow_style=False)

        # Verify the exact persisted representation with the safe YAML loader.
        parsed = yaml.safe_load(content)
        if not isinstance(parsed, dict):
            raise ValueError("YAML root must be a mapping")
        if required_root and not isinstance(parsed.get(required_root), dict):
            raise ValueError(f"'{required_root}' must be a mapping")
        json.dumps(parsed, allow_nan=False)
    except Exception as exc:
        # Includes PyYAML and ruamel parser/serializer exceptions.
        raise HTTPException(status_code=422, detail=f"Configuration cannot be serialized as safe YAML: {exc}") from exc
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    saved_content, saved_config, revision = _read_processing_config(path, config_id)
    return {"id": config_id, "name": path.name, "content": saved_content, "config": saved_config, "revision": revision}
