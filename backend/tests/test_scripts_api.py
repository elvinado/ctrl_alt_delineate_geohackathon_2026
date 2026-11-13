from pathlib import Path

import pytest
from fastapi import HTTPException

from app import scripts_api


def test_script_id_rejects_paths_and_non_python_files(tmp_path, monkeypatch):
    monkeypatch.setattr(scripts_api, "_scripts_dir", lambda db: tmp_path)
    for script_id in ("../secret.py", "folder/test.py", "runner.sh", ".py"):
        with pytest.raises(HTTPException) as exc:
            scripts_api._script_path(None, script_id)
        assert exc.value.status_code == 422


def test_script_path_is_under_scripts_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(scripts_api, "_scripts_dir", lambda db: tmp_path)
    assert scripts_api._script_path(None, "analysis.py") == (tmp_path / "analysis.py").resolve()


def test_scripts_routes_are_registered():
    from app.api import api_router

    routes = {(route.path, tuple(sorted(route.methods or []))) for route in api_router.routes}
    assert ("/scripts", ("GET",)) in routes
    assert ("/scripts", ("POST",)) in routes
    assert ("/scripts/{script_id}", ("GET",)) in routes
    assert ("/scripts/{script_id}", ("PUT",)) in routes
    assert ("/scripts/{script_id}", ("DELETE",)) in routes


def test_default_scripts_are_copied_only_when_scripts_directory_is_missing(tmp_path, monkeypatch):
    import shutil

    source = tmp_path / "package" / "scripts"
    source.mkdir(parents=True)
    (source / "gpu_test.py").write_text("print('gpu')\n", encoding="utf-8")
    (source / "pyrunner.local.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    target_root = tmp_path / "site_data"
    target_root.mkdir()

    monkeypatch.setattr("app.storage.Path", Path)
    import app.storage as storage
    monkeypatch.setattr(storage, "__file__", str(tmp_path / "package" / "app" / "storage.py"))
    (tmp_path / "package" / "app").mkdir()
    # Source resolution is based on storage.py's parent directory.
    monkeypatch.setattr(storage, "shutil", shutil)
    storage._ensure_default_scripts(target_root)

    target = target_root / "scripts"
    assert (target / "gpu_test.py").read_text(encoding="utf-8") == "print('gpu')\n"
    assert (target / "pyrunner.local.sh").is_file()

    # Existing directories are user-managed and must never be overwritten/seeded.
    existing_root = tmp_path / "existing"
    (existing_root / "scripts").mkdir(parents=True)
    (existing_root / "scripts" / "custom.py").write_text("# user script\n", encoding="utf-8")
    storage._ensure_default_scripts(existing_root)
    assert [p.name for p in (existing_root / "scripts").iterdir()] == ["custom.py"]
