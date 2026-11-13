"""Database bootstrap and SQLAlchemy session management.

The database URL is bootstrap configuration: it must be known before the
editable application settings can be read from the database itself. Override
DATABASE_URL in backend/.env when a non-default database is required.
"""
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = BASE_DIR / "data" / "geohackathon.db"


class DatabaseSettings(BaseSettings):
    database_url: str = f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def resolve_database_path(self):
        """Resolve SQLite paths relative to the backend and create their parent."""
        url = self.database_url
        if url.startswith("sqlite:///"):
            raw_path = url[len("sqlite:///"):]
            if raw_path not in (":memory:", ""):
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = BASE_DIR / path
                path = path.resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                self.database_url = f"sqlite:///{path.as_posix()}"
        return self


database_settings = DatabaseSettings()
connect_args = (
    {"check_same_thread": False}
    if database_settings.database_url.startswith("sqlite")
    else {}
)
engine = create_engine(database_settings.database_url, connect_args=connect_args, future=True)

if database_settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
