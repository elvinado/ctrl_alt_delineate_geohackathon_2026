import sys
import types

# The catalog test does not parse LAS/SEG-Y; storage imports these optional readers.
sys.modules.setdefault("lasio", types.ModuleType("lasio"))
sys.modules.setdefault("segyio", types.ModuleType("segyio"))

from app.geochat.skills.processing_catalog import handler  # noqa: E402


def test_processing_catalog_scans_workers_and_support_files(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "worker.py").write_text(
        '"""Example processing worker."""\nif __name__ == "__main__":\n    pass\n',
        encoding="utf-8",
    )
    (scripts / "scripts_utils.py").write_text("def helper(): pass\n", encoding="utf-8")
    (scripts / "runner.sbatch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (scripts / "Model_Test" / "checkpoint").mkdir(parents=True)

    monkeypatch.setattr(handler, "ensure_storage_root", lambda db: tmp_path)
    result = handler.execute(db=None, project_id="project-1", parameters={})

    assert result["runnable_count"] == 1
    assert result["workers"][0]["name"] == "worker.py"
    assert result["workers"][0]["runnable_candidate"] is True
    assert result["support_files"] == ["scripts_utils.py"]
    assert result["runners"] == ["runner.sbatch.sh"]
    assert result["models"][0]["name"] == "Model_Test"
    assert result["scan_is_read_only"] is True


def test_processing_catalog_exposes_generic_script_navigation_metadata(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "prepare_inputs.py").write_text(
        '"""Prepare input datasets for processing."""\n', encoding="utf-8"
    )

    monkeypatch.setattr(handler, "ensure_storage_root", lambda db: tmp_path)
    result = handler.execute(db=None, project_id="project-1", parameters={})

    assert result["scripts"] == [{
        "id": "prepare_inputs.py",
        "name": "prepare_inputs.py",
        "path": str(scripts / "prepare_inputs.py"),
        "description": "Prepare input datasets for processing.",
    }]
