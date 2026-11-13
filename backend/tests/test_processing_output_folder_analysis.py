import csv
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base
from app.models import AppSetting
from app.schemas_settings import StorageSettings
from app.geochat.skills.processing_output_folder_analysis.handler import execute


def test_output_folder_skill_summarizes_all_artifacts(tmp_path):
    data_root = tmp_path / "site_data"
    job_dir = data_root / "processing_jobs" / "job-123"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({
        "python_script": "example_worker.py", "target_well": "Well-A", "status": "succeeded"
    }), encoding="utf-8")
    (job_dir / "result.json").write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
    with (job_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["depth", "score"])
        writer.writeheader()
        writer.writerows([{"depth": "100", "score": "0.2"}, {"depth": "110", "score": "0.8"}])
    (job_dir / "notes.txt").write_text("Processing completed\nTwo sample rows", encoding="utf-8")

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    settings = StorageSettings().model_dump()
    settings["data_root"] = str(data_root)
    settings["fallback_data_root"] = str(tmp_path / "fallback")
    with Session(engine) as db:
        db.add(AppSetting(key="storage", value_json=json.dumps(settings)))
        db.commit()
        report = execute(db=db, project_id=None, parameters={"job_id": "job-123"})

    assert report["skill"] == "processing_output_folder_analysis"
    assert report["python_script"] == "example_worker.py"
    assert {item["name"] for item in report["artifacts"]} == {"results.csv", "notes.txt"}
    csv_report = next(item for item in report["artifacts"] if item["format"] == "CSV")
    assert csv_report["rows_scanned"] == 2
    assert csv_report["numeric_fields"]["score"]["mean"] == 0.5
