from contextlib import asynccontextmanager
import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.database import SessionLocal
from app.database import engine
from app.models import Base
from app.storage import ensure_storage_root
from app.runtime_settings import migrate_legacy_storage_defaults
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

# Bootstrap settings are environment-level because API routing and middleware are
# configured before database-backed editable settings can be loaded.
class InfrastructureSettings(BaseSettings):
    app_name: str = "GeoHackathon Data API"
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


infrastructure = InfrastructureSettings()
APP_NAME = infrastructure.app_name
API_PREFIX = infrastructure.api_prefix
CORS_ORIGINS = [
    origin.strip() for origin in infrastructure.cors_origins.split(",") if origin.strip()
]


def _migrate_optional_well_associations() -> None:
    """Allow well-log, deviation, and checkshot datasets to exist before assignment."""
    tables = ("well_log_datasets", "deviation_datasets", "checkshot_datasets")
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    present = [table for table in tables if table in existing]
    if not present:
        return

    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            for table in present:
                connection.execute(text(f'ALTER TABLE "{table}" ALTER COLUMN well_id DROP NOT NULL'))
        return

    if engine.dialect.name != "sqlite":
        return

    # SQLite cannot directly drop NOT NULL. Rebuild only the subtype tables that
    # still have the old schema. Foreign keys are disabled for this connection
    # during the table replacement and re-enabled immediately afterwards.
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        for table in present:
            info = cursor.execute(f'PRAGMA table_info("{table}")').fetchall()
            well_column = next((column for column in info if column[1] == "well_id"), None)
            if well_column is None or not well_column[3]:
                continue
            temp = f"{table}_optional_well_tmp"
            cursor.execute(f'DROP TABLE IF EXISTS "{temp}"')
            cursor.execute(
                f'CREATE TABLE "{temp}" ('
                'id VARCHAR(36) NOT NULL PRIMARY KEY, '
                'well_id VARCHAR(36), '
                'FOREIGN KEY(id) REFERENCES datasets(id) ON DELETE CASCADE, '
                'FOREIGN KEY(well_id) REFERENCES wells(id) ON DELETE SET NULL)'
            )
            cursor.execute(f'INSERT INTO "{temp}" (id, well_id) SELECT id, well_id FROM "{table}"')
            cursor.execute(f'DROP TABLE "{table}"')
            cursor.execute(f'ALTER TABLE "{temp}" RENAME TO "{table}"')
            cursor.execute(f'CREATE INDEX IF NOT EXISTS "ix_{table}_well_id" ON "{table}" (well_id)')
        raw.commit()
        cursor.execute("PRAGMA foreign_keys=ON")
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def _migrate_optional_seismic_association() -> None:
    """Allow SEG-Y datasets to exist without a survey assignment."""
    table = "seismic_datasets"
    if table not in set(inspect(engine).get_table_names()):
        return

    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text('ALTER TABLE "seismic_datasets" ALTER COLUMN survey_id DROP NOT NULL'))
            # Keep the database FK aligned with the optional association semantics.
            constraints = inspect(engine).get_foreign_keys(table)
            for constraint in constraints:
                if constraint.get("constrained_columns") == ["survey_id"] and constraint.get("name"):
                    connection.execute(text(
                        f'ALTER TABLE "{table}" DROP CONSTRAINT "{constraint["name"]}"'
                    ))
                    connection.execute(text(
                        f'ALTER TABLE "{table}" ADD CONSTRAINT "{constraint["name"]}" '
                        'FOREIGN KEY (survey_id) REFERENCES surveys(id) ON DELETE SET NULL'
                    ))
        return

    if engine.dialect.name != "sqlite":
        return

    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        info = cursor.execute(f'PRAGMA table_info("{table}")').fetchall()
        survey_column = next((column for column in info if column[1] == "survey_id"), None)
        foreign_keys = cursor.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
        survey_fk = next((fk for fk in foreign_keys if fk[3] == "survey_id"), None)
        needs_nullable = bool(survey_column and survey_column[3])
        needs_set_null = bool(survey_fk and str(survey_fk[6]).upper() != "SET NULL")
        if not needs_nullable and not needs_set_null:
            return

        cursor.execute("PRAGMA foreign_keys=OFF")
        temp = f"{table}_optional_survey_tmp"
        cursor.execute(f'DROP TABLE IF EXISTS "{temp}"')
        cursor.execute(
            f'CREATE TABLE "{temp}" ('
            'id VARCHAR(36) NOT NULL PRIMARY KEY, '
            'survey_id VARCHAR(36), '
            'FOREIGN KEY(id) REFERENCES datasets(id) ON DELETE CASCADE, '
            'FOREIGN KEY(survey_id) REFERENCES surveys(id) ON DELETE SET NULL'
            ')'
        )
        cursor.execute(
            f'INSERT INTO "{temp}" (id, survey_id) '
            f'SELECT id, survey_id FROM "{table}"'
        )
        cursor.execute(f'DROP TABLE "{table}"')
        cursor.execute(f'ALTER TABLE "{temp}" RENAME TO "{table}"')
        cursor.execute(f'CREATE INDEX IF NOT EXISTS "ix_{table}_survey_id" ON "{table}" (survey_id)')
        raw.commit()
        cursor.execute("PRAGMA foreign_keys=ON")
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def _migrate_dataset_storage_mode() -> None:
    """Add linked-file metadata while preserving existing managed datasets."""
    if "datasets" not in set(inspect(engine).get_table_names()):
        return
    columns = {column["name"] for column in inspect(engine).get_columns("datasets")}
    additions = {
        "storage_mode": "VARCHAR(20) NOT NULL DEFAULT 'managed'",
        "source_path": "VARCHAR(2000)",
    }
    missing = {name: ddl for name, ddl in additions.items() if name not in columns}
    if not missing:
        return
    with engine.begin() as connection:
        for name, ddl in missing.items():
            connection.execute(text(f'ALTER TABLE "datasets" ADD COLUMN "{name}" {ddl}'))




def _migrate_well_rt_td() -> None:
    """Add optional rotary-table elevation and total-depth columns to existing wells."""
    if "wells" not in set(inspect(engine).get_table_names()):
        return
    columns = {column["name"] for column in inspect(engine).get_columns("wells")}
    missing = {name: "FLOAT" for name in ("rt", "td") if name not in columns}
    if not missing:
        return
    with engine.begin() as connection:
        for name, ddl in missing.items():
            connection.execute(text(f'ALTER TABLE "wells" ADD COLUMN "{name}" {ddl}'))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _migrate_optional_well_associations()
    _migrate_well_rt_td()
    _migrate_optional_seismic_association()
    Base.metadata.create_all(bind=engine)
    _migrate_dataset_storage_mode()
    with SessionLocal() as db:
        migrate_legacy_storage_defaults(db)
        storage_root = ensure_storage_root(db)
    logger.info("Using database: %s", engine.url.render_as_string(hide_password=True))
    logger.info("Using dataset storage root: %s", storage_root)
    yield

app = FastAPI(
    title=APP_NAME,
    description=(
        "Frontend-agnostic geoscience API organized into INGEST, QC, FEATGEN, "
        "and MLPROC pipeline domains. LAS curve definitions are catalogued as first-class "
        "Curve resources; numerical samples remain in source datasets. INGEST is implemented, "
        "and MLPROC includes a minimal SLURM-backed GPU test job."
    ),
    version="0.5.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=API_PREFIX)


@app.get("/", tags=["system"])
def root():
    return {"service": APP_NAME, "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["system"])
def root_health():
    return {"status": "ok", "service": "geohackathon-data-api"}
