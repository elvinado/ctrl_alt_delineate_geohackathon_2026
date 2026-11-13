import hashlib
import json
import sys
import types
from pathlib import Path

# The release API does not read seismic/LAS files; stub optional header readers
# so these focused tests do not require the full scientific stack.
sys.modules.setdefault("lasio", types.ModuleType("lasio"))
sys.modules.setdefault("segyio", types.ModuleType("segyio"))

from fastapi import HTTPException

from app import model_releases


def _make_release(root: Path, release_id: str = "geogg-test") -> Path:
    release = root / "model_releases" / release_id
    (release / "models").mkdir(parents=True)
    payload = b"model-bytes"
    (release / "models" / "model.pt").write_bytes(payload)
    manifest = {
        "model_release_id": release_id,
        "status": "ready",
        "pipeline": "geogg_fresh",
        "pipeline_version": "v82",
        "model_family": "deep",
        "processing_job_id": "job-1",
        "training_run_id": "run-1",
        "models": [{"model": "tcn", "model_file": "models/model.pt"}],
        "files": {"models/model.pt": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}},
    }
    (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return release


def test_list_and_get_release(tmp_path, monkeypatch):
    release = _make_release(tmp_path)
    monkeypatch.setattr(model_releases, "_release_roots", lambda _db: [tmp_path / "model_releases"])
    listed = model_releases.list_model_releases(db=object())
    assert listed["count"] == 1
    assert listed["items"][0]["model_release_id"] == "geogg-test"
    detail = model_releases.get_model_release("geogg-test", db=object())
    assert detail["manifest"]["models"][0]["model"] == "tcn"


def test_validate_checksums_and_detects_corruption(tmp_path, monkeypatch):
    release = _make_release(tmp_path)
    monkeypatch.setattr(model_releases, "_release_roots", lambda _db: [tmp_path / "model_releases"])
    result = model_releases.validate_model_release("geogg-test", db=object())
    assert result["valid"] is True
    (release / "models" / "model.pt").write_bytes(b"changed")
    result = model_releases.validate_model_release("geogg-test", db=object())
    assert result["valid"] is False
    assert any("Size mismatch" in error for error in result["errors"])


def test_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(model_releases, "_release_roots", lambda _db: [tmp_path / "model_releases"])
    try:
        model_releases.get_model_release("../outside", db=object())
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("Expected invalid release ID to be rejected")


def test_delete_model_releases_for_processing_job_uses_manifest_provenance(tmp_path, monkeypatch):
    release = _make_release(tmp_path)
    unrelated = _make_release(tmp_path, "geogg-other")
    unrelated_manifest = json.loads((unrelated / "manifest.json").read_text())
    unrelated_manifest["processing_job_id"] = "job-2"
    (unrelated / "manifest.json").write_text(json.dumps(unrelated_manifest), encoding="utf-8")
    monkeypatch.setattr(model_releases, "_release_roots", lambda _db: [tmp_path / "model_releases"])

    assert model_releases.delete_model_releases_for_processing_job("job-1", db=object()) == 1
    assert not release.exists()
    assert unrelated.exists()
    # Repeated deletion is idempotent when the paired release was already removed.
    assert model_releases.delete_model_releases_for_processing_job("job-1", db=object()) == 0
