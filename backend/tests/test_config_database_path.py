from pathlib import Path

from app.database import DatabaseSettings


def test_absolute_sqlite_database_url_is_resolved_and_parent_created(tmp_path):
    database_path = tmp_path / "site_data" / "geohackathon.db"
    settings = DatabaseSettings(database_url=f"sqlite:///{database_path}")
    assert settings.database_url == f"sqlite:///{database_path.resolve().as_posix()}"
    assert database_path.parent.is_dir()


def test_relative_sqlite_database_url_is_resolved_from_backend_root():
    settings = DatabaseSettings(database_url="sqlite:///data/test.db")
    expected = Path(__file__).resolve().parents[1] / "data" / "test.db"
    assert settings.database_url == f"sqlite:///{expected.resolve().as_posix()}"
