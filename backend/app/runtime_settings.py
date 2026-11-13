import json
from sqlalchemy.orm import Session
from app.models import AppSetting
from app.schemas_settings import (
    IngestionSettings,
    ProcessingSettings,
    SettingsOut,
    StorageSettings,
    StorageSettingsPatch,
    GeoChatSettings,
    LEGACY_DATA_ROOT,
    LEGACY_FALLBACK_DATA_ROOT,
    default_data_root,
    default_fallback_data_root,
)

DEFAULTS = {
    "ingestion": IngestionSettings().model_dump(),
    "processing": ProcessingSettings().model_dump(),
    "storage": StorageSettings().model_dump(),
    "geochat": GeoChatSettings().model_dump(),
}
VERSION_KEY = "__version__"


def get_effective_settings(db: Session) -> SettingsOut:
    rows = {row.key: row.value_json for row in db.query(AppSetting).all()}
    values = {}
    for section, default in DEFAULTS.items():
        raw = json.loads(rows[section]) if section in rows else default
        if section == "ingestion":
            values[section] = IngestionSettings.model_validate(raw)
        elif section == "processing":
            values[section] = ProcessingSettings.model_validate(raw)
        elif section == "storage":
            values[section] = StorageSettings.model_validate(raw)
        else:
            values[section] = GeoChatSettings.model_validate(raw)
    version = int(json.loads(rows[VERSION_KEY])) if VERSION_KEY in rows else 1
    return SettingsOut(
        ingestion=values["ingestion"],
        processing=values["processing"],
        storage=values["storage"],
        geochat=values["geochat"],
        version=version,
    )


def update_settings(db: Session, patch: dict, updated_by: str | None = None) -> SettingsOut:
    current = get_effective_settings(db)
    expected = patch.get("expected_version")
    if expected is not None and expected != current.version:
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="Settings have changed; reload and retry.")

    for section in ("ingestion", "processing", "storage", "geochat"):
        section_patch = patch.get(section)
        if section_patch is None:
            continue
        current_value = getattr(current, section).model_dump()
        if hasattr(section_patch, "model_dump"):
            section_patch = section_patch.model_dump(exclude_unset=True)
        current_value.update(section_patch)
        model = {
            "ingestion": IngestionSettings,
            "processing": ProcessingSettings,
            "storage": StorageSettings,
            "geochat": GeoChatSettings,
        }[section]
        validated = model.model_validate(current_value).model_dump()

        row = db.get(AppSetting, section)
        encoded = json.dumps(validated)
        if row:
            row.value_json = encoded
            row.updated_by = updated_by
        else:
            db.add(AppSetting(key=section, value_json=encoded, updated_by=updated_by))

    new_version = current.version + 1
    version_row = db.get(AppSetting, VERSION_KEY)
    if version_row:
        version_row.value_json = json.dumps(new_version)
        version_row.updated_by = updated_by
    else:
        db.add(AppSetting(key=VERSION_KEY, value_json=json.dumps(new_version), updated_by=updated_by))
    db.commit()
    return get_effective_settings(db)


def migrate_legacy_storage_defaults(db: Session) -> bool:
    """Replace only the old hardcoded defaults; preserve user-customized paths."""
    row = db.get(AppSetting, "storage")
    if row is None:
        return False

    try:
        raw = json.loads(row.value_json)
    except (TypeError, json.JSONDecodeError):
        return False

    changed = False
    new_data_root = default_data_root()
    new_fallback_data_root = default_fallback_data_root()
    if raw.get("data_root") == LEGACY_DATA_ROOT and raw.get("data_root") != new_data_root:
        raw["data_root"] = new_data_root
        changed = True
    if (
        raw.get("fallback_data_root") == LEGACY_FALLBACK_DATA_ROOT
        and raw.get("fallback_data_root") != new_fallback_data_root
    ):
        raw["fallback_data_root"] = new_fallback_data_root
        changed = True

    if changed:
        row.value_json = json.dumps(StorageSettings.model_validate(raw).model_dump())
        db.commit()
    return changed
