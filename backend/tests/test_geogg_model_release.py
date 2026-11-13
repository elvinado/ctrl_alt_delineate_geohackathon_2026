import csv
import json
from pathlib import Path

import pytest

from scripts.geogg_fresh.model_release import ModelReleaseError, create_model_release


def _write(path: Path, value: str = "artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _fixture_job(tmp_path: Path) -> tuple[Path, dict]:
    job = tmp_path / "site_data" / "processing_jobs" / "job-123"
    stages = job / "stages"
    s03, s05, s06, s07 = (stages / name for name in
                          ("03_extraction", "05_training_tables", "06_feature_selection", "07_models"))
    _write(s03 / "tables" / "extraction_metadata.csv", "channel,source\nfull,full.sgy\n")
    _write(s03 / "patches" / "patch_manifest_4ms_v1.json", '{"dtype":"float32"}')
    _write(s05 / "feature_schema.json", '{"schema_version":1}')
    _write(s05 / "training_manifest.json", '{"rows":10}')
    _write(s05 / "tables" / "split_plan_wells.csv", "well,role\nA,development\n")
    _write(s06 / "feature_sets.json", '{"resolutions":{}}')
    _write(s06 / "transforms" / "transforms_4ms.json", '{"folds":{}}')
    _write(s07 / "models" / "final" / "tcn.pt", "fake model weights")
    _write(s07 / "tables" / "outer_test_predictions.csv", "well,prediction\nA,0.5\n")
    _write(s07 / "tables" / "fit_audit.csv", "train_wells,test_wells,overlap\nA,B,\n")
    _write(s07 / "architectures" / "model_architectures.json", '{"deep":{}}')
    registry = s07 / "tables" / "model_registry.csv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "input", "model", "status", "final_model_file",
                                                    "final_params", "final_threshold", "scaling_mode"])
        writer.writeheader()
        writer.writerow({"task": "lithology", "input": "patch_stacks", "model": "tcn", "status": "OK",
                         "final_model_file": "models/final/tcn.pt", "final_params": "epochs=4",
                         "final_threshold": "0.5", "scaling_mode": "common"})
    cfg = {"_runtime": {"config_sha256": "abc123", "config_path": None}, "models": {"deep": ["tcn"]}}
    return job, cfg


def test_create_model_release_packages_models_contracts_and_hashes(tmp_path):
    job, cfg = _fixture_job(tmp_path)
    result = create_model_release(job_dir=job, config=cfg, run_id="job-123")

    assert result["status"] == "ready"
    assert result["model_release_id"] == "geogg-job-123"
    release = Path(result["path"])
    manifest = json.loads((release / "manifest.json").read_text())
    assert manifest["model_family"] == "deep"
    assert manifest["models"][0]["model_file"] == "models/final/tcn.pt"
    assert (release / "preprocessing" / "transforms" / "transforms_4ms.json").is_file()
    assert (release / "contracts" / "feature_schema.json").is_file()
    assert "sha256" in manifest["files"]["models/final/tcn.pt"]

    with pytest.raises(ModelReleaseError, match="already exists"):
        create_model_release(job_dir=job, config=cfg, run_id="job-123")


def test_no_successful_model_does_not_register_empty_release(tmp_path):
    job, cfg = _fixture_job(tmp_path)
    registry = job / "stages/07_models/tables/model_registry.csv"
    registry.write_text("task,input,model,status,final_model_file\nlithology,patch_stacks,tcn,FAILED,\n")
    assert create_model_release(job_dir=job, config=cfg, run_id="job-123") is None
    assert not (job.parent.parent / "model_releases").exists()


def test_missing_required_artifact_fails_release(tmp_path):
    job, cfg = _fixture_job(tmp_path)
    (job / "stages/06_feature_selection/transforms/transforms_4ms.json").unlink()
    with pytest.raises(ModelReleaseError, match="transform"):
        create_model_release(job_dir=job, config=cfg, run_id="job-123")
    assert not (job.parent.parent / "model_releases/geogg-job-123").exists()
