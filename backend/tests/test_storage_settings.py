from pathlib import Path

from app.schemas_settings import (
    DEFAULT_DATA_ROOT,
    DEFAULT_FALLBACK_DATA_ROOT,
    default_data_root,
    default_fallback_data_root,
    StorageSettings,
    StorageSettingsPatch,
)


def test_storage_defaults_use_configured_deployment_paths():
    settings = StorageSettings()
    assert settings.data_root == str(Path(DEFAULT_DATA_ROOT).resolve())
    assert settings.fallback_data_root == str(Path(DEFAULT_FALLBACK_DATA_ROOT).resolve())
    assert settings.data_root == default_data_root()
    assert settings.fallback_data_root == default_fallback_data_root()


def test_storage_paths_are_editable_and_must_be_absolute(tmp_path):
    settings = StorageSettingsPatch(data_root=str(tmp_path / "site_data"))
    assert settings.data_root == str((tmp_path / "site_data").resolve())

    try:
        StorageSettingsPatch(data_root="relative/site_data")
    except Exception as exc:
        assert "absolute paths" in str(exc)
    else:
        raise AssertionError("relative storage paths should be rejected")


def test_legacy_storage_defaults_are_migrated_but_custom_paths_are_preserved(tmp_path):
    import json
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.models import AppSetting, Base
    from app.runtime_settings import migrate_legacy_storage_defaults
    from app.schemas_settings import LEGACY_DATA_ROOT, LEGACY_FALLBACK_DATA_ROOT

    engine = create_engine(f"sqlite:///{tmp_path / 'settings.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(AppSetting(key="storage", value_json=json.dumps({
            "data_root": LEGACY_DATA_ROOT,
            "fallback_data_root": LEGACY_FALLBACK_DATA_ROOT,
        })))
        db.commit()

        assert migrate_legacy_storage_defaults(db) is True
        migrated = json.loads(db.get(AppSetting, "storage").value_json)
        assert migrated["data_root"] == default_data_root()
        assert migrated["fallback_data_root"] == default_fallback_data_root()

        assert migrate_legacy_storage_defaults(db) is False
    engine.dispose()


def test_ensure_storage_root_creates_geochat_directory_in_active_root(tmp_path):
    import json
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.models import AppSetting, Base
    from app.storage import ensure_storage_root

    data_root = tmp_path / "local_site_data"
    engine = create_engine(f"sqlite:///{tmp_path / 'storage.db'}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            db.add(AppSetting(key="storage", value_json=json.dumps({
                "data_root": str(data_root),
                "fallback_data_root": str(tmp_path / "fallback_site_data"),
            })))
            db.commit()

            active_root = ensure_storage_root(db)

            assert active_root == data_root.resolve()
            assert (active_root / "uploads").is_dir()
            assert (active_root / "geochat").is_dir()
    finally:
        engine.dispose()


def test_ensure_storage_root_refreshes_defaults_and_preserves_runtime_scripts(tmp_path):
    import json
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.models import AppSetting, Base
    from app.storage import ensure_storage_root

    data_root = tmp_path / "site_data"
    scripts_dir = data_root / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "my_custom_script.py").write_text("custom", encoding="utf-8")

    engine = create_engine(f"sqlite:///{tmp_path / 'sync.db'}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            db.add(AppSetting(key="storage", value_json=json.dumps({
                "data_root": str(data_root),
                "fallback_data_root": str(tmp_path / "fallback"),
            })))
            db.commit()

            ensure_storage_root(db)

            source_scripts = Path(__file__).resolve().parents[1] / "scripts"
            assert not (scripts_dir / "gpu_test.py").exists()
            assert (scripts_dir / "A_foundations.yaml").read_bytes() == (source_scripts / "A_foundations.yaml").read_bytes()
            assert (scripts_dir / "my_custom_script.py").read_text(encoding="utf-8") == "custom"
            assert (scripts_dir / "geogg_fresh" / "README.md").is_file()
    finally:
        engine.dispose()
