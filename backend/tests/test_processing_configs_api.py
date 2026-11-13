from pathlib import Path

import pytest
from fastapi import HTTPException

from app import scripts_api


def test_processing_config_read_and_update_preserves_comments(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    pytest.importorskip("ruamel.yaml")
    monkeypatch.setattr(scripts_api, "_scripts_dir", lambda db: tmp_path)
    config_path = tmp_path / "B_inference.yaml"
    config_path.write_text(
        "# Keep this explanation\ninference:\n  model:\n    task: all\n    deep_batch_size: 512\n",
        encoding="utf-8",
    )

    current = scripts_api.get_processing_config("B_inference", db=None)
    assert current["config"]["inference"]["model"]["deep_batch_size"] == 512

    payload = scripts_api.ProcessingConfigUpdate(
        config={
            "inference": {
                "model": {"task": "vsh", "deep_batch_size": 256}
            }
        },
        expected_revision=current["revision"],
    )
    saved = scripts_api.update_processing_config("B_inference", payload, db=None)

    assert saved["config"]["inference"]["model"]["task"] == "vsh"
    assert "Keep this explanation" in config_path.read_text(encoding="utf-8")


def test_processing_config_rejects_stale_revision(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    monkeypatch.setattr(scripts_api, "_scripts_dir", lambda db: tmp_path)
    path = tmp_path / "C_volume_products.yaml"
    path.write_text("volume_products:\n  generate_sections: true\n", encoding="utf-8")
    payload = scripts_api.ProcessingConfigUpdate(
        config={"volume_products": {"generate_sections": False}},
        expected_revision="0" * 64,
    )
    with pytest.raises(HTTPException) as exc:
        scripts_api.update_processing_config("C_volume_products", payload, db=None)
    assert exc.value.status_code == 409


def test_processing_config_allowlist_rejects_arbitrary_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(scripts_api, "_scripts_dir", lambda db: tmp_path)
    with pytest.raises(HTTPException) as exc:
        scripts_api._config_path(None, "../../secrets")
    assert exc.value.status_code == 404


def test_storage_sync_does_not_overwrite_user_edited_stage_yaml(tmp_path, monkeypatch):
    import app.storage as storage

    package = tmp_path / "package"
    app_dir = package / "app"
    scripts = package / "scripts"
    app_dir.mkdir(parents=True)
    scripts.mkdir()
    (scripts / "A_foundations.yaml").write_text("project:\n  data_mode: DEFAULT\n", encoding="utf-8")
    target = tmp_path / "site_data"
    (target / "scripts").mkdir(parents=True)
    edited = target / "scripts" / "A_foundations.yaml"
    edited.write_text("project:\n  data_mode: USER_EDITED\n", encoding="utf-8")

    monkeypatch.setattr(storage, "__file__", str(app_dir / "storage.py"))
    storage._sync_default_scripts(target)

    assert "USER_EDITED" in edited.read_text(encoding="utf-8")



def test_processing_config_routes_are_registered():
    from app.api import api_router

    routes = {(route.path, tuple(sorted(route.methods or []))) for route in api_router.routes}
    assert ("/processing-configs", ("GET",)) in routes
    assert ("/processing-configs/{config_id}", ("GET",)) in routes
    assert ("/processing-configs/{config_id}", ("PUT",)) in routes
