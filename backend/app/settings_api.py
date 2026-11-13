"""Runtime settings endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas_settings import SettingsOut, SettingsPatch
from app.runtime_settings import get_effective_settings, update_settings

router = APIRouter()


@router.get("", response_model=SettingsOut)
def read_settings(db: Session = Depends(get_db)) -> SettingsOut:
    """Return the effective runtime configuration."""
    return get_effective_settings(db)


@router.patch("", response_model=SettingsOut)
def patch_settings(
    payload: SettingsPatch, db: Session = Depends(get_db)
) -> SettingsOut:
    """Validate and persist supplied runtime settings."""
    data = payload.model_dump(exclude_unset=True)
    return update_settings(db, data)
