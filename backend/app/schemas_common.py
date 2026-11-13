from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None

class ProjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None

class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    description: str | None
    created_at: datetime

class SurveyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    crs: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class SurveyPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    crs: str | None = None
    metadata: dict[str, Any] | None = None

class SurveyOut(BaseModel):
    id: str
    project_id: str
    name: str
    description: str | None
    crs: str | None
    metadata: dict[str, Any]
    created_at: datetime

class WellCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    uwi: str | None = None
    x: float | None = None
    y: float | None = None
    rt: float | None = None
    td: float | None = None
    crs: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class WellPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    uwi: str | None = None
    x: float | None = None
    y: float | None = None
    rt: float | None = None
    td: float | None = None
    crs: str | None = None
    metadata: dict[str, Any] | None = None

class WellOut(BaseModel):
    id: str
    project_id: str
    name: str
    uwi: str | None
    x: float | None
    y: float | None
    rt: float | None
    td: float | None
    crs: str | None
    metadata: dict[str, Any]
    created_at: datetime

class DatasetPatch(BaseModel):
    """Mutable descriptive fields only; kind and file identity are immutable."""
    name: str | None = Field(default=None, min_length=1, max_length=255)
    survey_id: str | None = None
    well_id: str | None = None
    crs: str | None = None
    metadata: dict[str, Any] | None = None

class DatasetOut(BaseModel):
    id: str
    project_id: str
    survey_id: str | None
    well_id: str | None
    kind: str
    dataset_type: str
    name: str
    original_filename: str
    format: str
    storage_root: str
    storage_mode: str = "managed"
    source_path: str | None = None
    crs: str | None
    size_bytes: int
    checksum_sha256: str
    status: str
    metadata: dict[str, Any]
    validation: dict[str, Any]
    created_at: datetime

class DatasetLinkLocal(BaseModel):
    project_id: str
    kind: str
    source_path: str = Field(min_length=1, max_length=2000)
    name: str | None = Field(default=None, max_length=255)
    survey_id: str | None = None
    well_id: str | None = None
    crs: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class DatasetUploadMetadata(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
