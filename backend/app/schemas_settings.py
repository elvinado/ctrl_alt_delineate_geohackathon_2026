from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_FORMATS = {"sgy", "segy", "las", "csv", "txt", "xlsx", "xls", "npy", "npz", "h5", "hdf5", "zarr", "vtk"}
# Default storage locations for this deployment. These are used only when the
# database has no saved storage settings; editable settings remain authoritative.
DEFAULT_DATA_ROOT = (
    "/home/elvinado/syncthing/Doctorate_UTM_2023/"
    "27 APGCE 2026 Geohackathon/site_data"
)
DEFAULT_FALLBACK_DATA_ROOT = "/data/epic20/APGCE/apgce_team_10/site_data"

# Previous first-run value found in the existing database. Only this exact value
# is migrated, so user-customized storage paths are not overwritten.
LEGACY_DATA_ROOT = (
    "/home/elvinado/syncthing/Doctorate_UTM_2023/"
    "27 APGCE 2026 Geohackathon/Mocking/backend/site_data"
)
LEGACY_FALLBACK_DATA_ROOT = DEFAULT_FALLBACK_DATA_ROOT


def default_data_root() -> str:
    return str(Path(DEFAULT_DATA_ROOT).expanduser().resolve())


def default_fallback_data_root() -> str:
    return str(Path(DEFAULT_FALLBACK_DATA_ROOT).expanduser().resolve())


def normalize_absolute_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("Storage paths must be absolute paths")
    return str(path.resolve())


class IngestionSettings(BaseModel):
    max_upload_size_mb: int = Field(default=16384, ge=1, le=65536)
    allowed_formats: list[str] = Field(default_factory=lambda: ["sgy", "segy", "las", "csv", "txt"])
    validate_file_format: bool = True

    @field_validator("allowed_formats")
    @classmethod
    def check_formats(cls, values):
        normalized = [v.lower().lstrip(".") for v in values]
        invalid = sorted(set(normalized) - ALLOWED_FORMATS)
        if invalid:
            raise ValueError(f"Unsupported formats: {', '.join(invalid)}")
        if not normalized:
            raise ValueError("At least one format must be enabled")
        return sorted(set(normalized))


class ProcessingSettings(BaseModel):
    default_timeout_seconds: int = Field(default=3600, ge=1, le=86400)
    default_runner: str = Field(default="pyrunner.local.sh", min_length=1, max_length=128)
    fallback_runner: str = Field(default="pyrunner.sbatch.sh", min_length=1, max_length=128)

    @field_validator("default_runner", "fallback_runner")
    @classmethod
    def runner_filename_only(cls, value: str) -> str:
        if Path(value).name != value or not value.endswith(".sh") or value in {".", ".."}:
            raise ValueError("Runner must be a .sh filename located directly in scripts/")
        return value

    @model_validator(mode="after")
    def runners_must_differ(self):
        if self.default_runner == self.fallback_runner:
            raise ValueError("Primary and fallback runners must be different")
        return self


class StorageSettings(BaseModel):
    data_root: str = Field(default_factory=default_data_root)
    fallback_data_root: str = Field(default_factory=default_fallback_data_root)

    @field_validator("data_root", "fallback_data_root")
    @classmethod
    def absolute_paths_only(cls, value: str) -> str:
        return normalize_absolute_path(value)


class StorageSettingsPatch(BaseModel):
    data_root: str | None = None
    fallback_data_root: str | None = None

    @field_validator("data_root", "fallback_data_root")
    @classmethod
    def absolute_paths_only(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_absolute_path(value)

    @model_validator(mode="after")
    def reject_explicit_null_paths(self):
        for field in ("data_root", "fallback_data_root"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


class GeoChatSettings(BaseModel):
    base_url: str = Field(default="http://172.28.34.1:8000/v1", min_length=1, max_length=500)
    model: str = Field(default="google/gemma-4-26B-A4B-it", min_length=1, max_length=200)
    system_prompt: str = Field(
        default="You are a helpful geoscience assistant. Be clear about uncertainty and do not invent evidence.",
        min_length=1,
        max_length=12000,
    )
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    top_p: float = Field(default=1.0, gt=0, le=1)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    timeout_seconds: int = Field(default=120, ge=5, le=600)


class SettingsOut(BaseModel):
    ingestion: IngestionSettings
    processing: ProcessingSettings
    storage: StorageSettings
    geochat: GeoChatSettings = Field(default_factory=GeoChatSettings)
    version: int


class SettingsPatch(BaseModel):
    ingestion: IngestionSettings | None = None
    processing: ProcessingSettings | None = None
    storage: StorageSettingsPatch | None = None
    geochat: GeoChatSettings | None = None
    expected_version: int | None = Field(default=None, ge=1)
