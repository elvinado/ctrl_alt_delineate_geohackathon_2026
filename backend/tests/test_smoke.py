"""Smoke tests for pipeline capabilities and resource-oriented route design."""

from app.api import api_router
from app.format_info import capabilities


def test_capabilities_include_pipeline_stages_and_readiness() -> None:
    """Capabilities should describe stages without defining URL namespaces."""
    response = capabilities()
    domains = {domain["id"]: domain for domain in response["pipeline_domains"]}

    assert set(domains) == {"INGEST", "QC", "FEATGEN", "MLPROC"}
    assert domains["INGEST"]["status"] == "available"
    assert domains["QC"]["status"] == "placeholder"
    assert domains["FEATGEN"]["status"] == "placeholder"
    assert domains["MLPROC"]["status"] == "available"
    assert all("base_path" not in domain for domain in response["pipeline_domains"])
    assert "las" in response["parsers"]
    assert "segy" in response["parsers"]
    assert "seismic" in response["dataset_kinds"]


def test_routes_are_resource_oriented_not_domain_prefixed() -> None:
    """Pipeline stages must not register their own URL namespaces."""
    paths = {route.path for route in api_router.routes}

    assert "/projects" in paths
    assert "/surveys" in paths
    assert "/wells" in paths
    assert "/datasets" in paths
    assert not any(
        path.startswith(("/ingest", "/qc", "/featgen", "/mlproc"))
        for path in paths
    )


def test_removed_planned_operations_are_not_registered() -> None:
    """Quality-check and feature placeholder endpoints are removed."""
    paths = {route.path for route in api_router.routes}
    assert "/datasets/{dataset_id}/quality-checks" not in paths
    assert "/datasets/{dataset_id}/features" not in paths
    assert "/processing-jobs" in paths


def test_processing_job_inspector_routes_are_registered() -> None:
    """The frontend can list jobs, list their files, and read permitted files."""
    paths = {route.path for route in api_router.routes}
    assert "/processing-jobs" in paths
    assert "/processing-jobs/{job_id}" in paths
    assert "/processing-jobs/{job_id}/files" in paths
    assert "/processing-jobs/{job_id}/files/{file_name}" in paths


def test_curve_resource_routes_are_registered() -> None:
    """Curve definitions are first-class resources beneath their source dataset."""
    paths = {route.path for route in api_router.routes}

    assert "/datasets/{dataset_id}/curves" in paths
    assert "/curves/{curve_id}" in paths
    assert "/curves/{curve_id}/data" in paths


def test_curve_model_has_dataset_provenance_and_definition_fields() -> None:
    """Curve records identify their source dataset and curve-header properties."""
    from app.models import Curve

    assert Curve.__tablename__ == "curves"
    assert {
        "id",
        "dataset_id",
        "mnemonic",
        "unit",
        "description",
        "position",
        "is_index",
        "metadata_json",
    }.issubset(Curve.__table__.columns.keys())
    assert Curve.__table__.c.dataset_id.foreign_keys


def test_curve_sample_service_paginates_and_converts_missing_values(monkeypatch, tmp_path) -> None:
    """Curve retrieval returns JSON-safe samples from the indexed LAS column."""
    from types import SimpleNamespace

    import numpy as np

    from app.models import Curve
    from app import curve_data

    fake_las = SimpleNamespace(
        data=np.array([[100.0, 1.5], [101.0, np.nan], [102.0, 2.5]]),
        # Match lasio's real API: ``las.index`` is an ndarray without ``name``.
        index=np.array([100.0, 101.0, 102.0]),
        curves=[SimpleNamespace(mnemonic="DEPT", unit="m")],
    )
    monkeypatch.setattr(curve_data.lasio, "read", lambda *args, **kwargs: fake_las)

    curve = Curve(
        id="curve-1",
        dataset_id="dataset-1",
        mnemonic="GR",
        unit="API",
        position=1,
        is_index=False,
        metadata_json="{}",
    )
    result = curve_data.read_curve_samples(
        tmp_path / "source.las", curve, offset=1, limit=2, index_min=101.0
    )

    assert result["total_samples"] == 2
    assert result["returned_samples"] == 1
    assert result["samples"] == [{"index": 102, "value": 2.5}]
    assert result["index"] == {"mnemonic": "DEPT", "unit": "m"}


def test_deviation_text_parser_reads_comment_headers_and_rows(tmp_path) -> None:
    from app.well_tables import read_well_table

    path = tmp_path / "deviation.txt"
    path.write_text(
        "#m-1\n#Deviation\n#MDKB (m)\n#AZIM\n#INCL\n"
        "0 0 0\n181.37 347.29 1.37\n207.26 349.7 1.01\n",
        encoding="utf-8",
    )
    result = read_well_table(path, "deviation", offset=1, limit=1)
    assert result["row_count"] == 3
    assert result["columns"][0]["name"] == "MDKB"
    assert result["columns"][0]["unit"] == "m"
    assert result["rows"] == [{"MDKB": 181.37, "AZIM": 347.29, "INCL": 1.37}]


def test_checkshot_text_parser_reads_rows_and_paginates(tmp_path) -> None:
    from app.well_tables import read_well_table

    path = tmp_path / "checkshot.txt"
    path.write_text(
        '#"m-1" checkshots\n#3\n#MDKB\n#TWT\n#TVDSS\n'
        "29.400000 0.000000 0.000000\n300.010000 336.200000 270.570000\n",
        encoding="utf-8",
    )
    result = read_well_table(path, "checkshot", offset=0, limit=1)
    assert result["row_count"] == 2
    assert result["columns"] == [
        {"name": "MDKB", "source_name": "MDKB", "unit": None},
        {"name": "TWT", "source_name": "TWT", "unit": None},
        {"name": "TVDSS", "source_name": "TVDSS", "unit": None},
    ]
    assert result["rows"] == [{"MDKB": 29.4, "TWT": 0.0, "TVDSS": 0.0}]


def test_processing_jobs_route_and_model_are_registered() -> None:
    from app.models import ProcessingJob

    paths = {route.path for route in api_router.routes}
    assert "/processing-jobs" in paths
    assert "/processing-jobs/{job_id}" in paths
    assert ProcessingJob.__tablename__ == "processing_jobs"


def test_processing_job_requires_registered_script_and_runner() -> None:
    import pytest

    from app.processing_jobs import ProcessingJobCreate, _validate_submission

    request = ProcessingJobCreate(
        python_script="gpu_test.py",
        shell_script="pyrunner.local.sh",
    )
    mode, worker, runner = _validate_submission(request)
    assert mode == "local"
    assert worker.name == "gpu_test.py"
    assert runner.name == "pyrunner.local.sh"

    from fastapi import HTTPException
    invalid = ProcessingJobCreate(python_script="../../etc/passwd", shell_script="pyrunner.local.sh")
    with pytest.raises(HTTPException):
        _validate_submission(invalid)



def test_processing_job_request_accepts_multiple_dataset_ids() -> None:
    from app.processing_jobs import ProcessingJobCreate

    request = ProcessingJobCreate(
        python_script="gpu_test.py",
        shell_script="pyrunner.local.sh",
        dataset_ids=["dataset-1", "dataset-2"],
    )
    assert request.dataset_ids == ["dataset-1", "dataset-2"]
    assert ProcessingJobCreate(
        python_script="gpu_test.py", shell_script="pyrunner.local.sh"
    ).dataset_ids == []

def test_dataset_inheritance_and_curve_parent_contract() -> None:
    from app.models import (
        Dataset, SeismicDataset, WellLogDataset, CheckshotDataset,
        DeviationDataset, Curve,
    )

    assert Dataset.__mapper_args__["polymorphic_identity"] == "dataset"
    assert SeismicDataset.__mapper_args__["polymorphic_identity"] == "seismic"
    assert WellLogDataset.__mapper_args__["polymorphic_identity"] == "well_log"
    assert CheckshotDataset.__mapper_args__["polymorphic_identity"] == "checkshot"
    assert DeviationDataset.__mapper_args__["polymorphic_identity"] == "deviation"
    assert next(iter(Curve.__table__.c.dataset_id.foreign_keys)).target_fullname == "well_log_datasets.id"


def test_dataset_data_route_is_registered() -> None:
    paths = {route.path for route in api_router.routes}
    assert "/datasets/{dataset_id}/data" in paths
    assert "/datasets/{dataset_id}/seismic/geometry" in paths
    assert "/datasets/{dataset_id}/seismic/section" in paths


def test_table_data_matches_frontend_shape(tmp_path) -> None:
    from app.dataset_data import read_table_data

    path = tmp_path / "deviation.txt"
    path.write_text(
        "#m-1\n#Deviation\n#MDKB (m) AZIM INCL\n100 25 2\n110 30 3\n",
        encoding="utf-8",
    )
    result = read_table_data(path, "deviation")
    assert result["type"] == "table"
    assert result["headers"] == ["MDKB", "AZIM", "INCL"]
    assert result["rows"] == [["100", "25", "2"], ["110", "30", "3"]]
    assert result["rowCount"] == 2
    assert result["delimiter"] == " "


def test_well_data_matches_frontend_shape(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    import numpy as np
    from app import dataset_data

    fake_las = SimpleNamespace(
        data=np.array([[100.0, 1.5], [101.0, np.nan], [102.0, 2.5]]),
        index=np.array([100.0, 101.0, 102.0]),
        curves=[SimpleNamespace(mnemonic="DEPT", unit="M"), SimpleNamespace(mnemonic="GR", unit="API")],
        well=SimpleNamespace(NULL=SimpleNamespace(value=-999.25)),
    )
    monkeypatch.setattr(dataset_data.lasio, "read", lambda *args, **kwargs: fake_las)
    result = dataset_data.read_well_data(tmp_path / "well.las")
    assert result["type"] == "well"
    assert result["curves"] == ["DEPT", "GR"]
    assert result["units"] == ["M", "API"]
    assert result["rows"] == [[100.0, 1.5], [101.0, -999.25], [102.0, 2.5]]
    assert result["depthStart"] == 100.0
    assert result["depthStop"] == 102.0
    assert result["nullValue"] == -999.25


def test_seismic_data_matches_frontend_shape(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    from app import dataset_data

    class FakeSegy:
        def __init__(self):
            self.trace = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            self.header = [
                {189: 10, 193: 20},
                {189: 11, 193: 20},
                {189: 12, 193: 21},
            ]
            self.samples = [0, 1]
            self.bin = {1: 2000, 2: 5}
            self.text = ["C 1 TEST TEXT HEADER"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(dataset_data.segyio, "open", lambda *args, **kwargs: FakeSegy())
    monkeypatch.setattr(dataset_data.segyio.TraceField, "INLINE_3D", 189, raising=False)
    monkeypatch.setattr(dataset_data.segyio.TraceField, "CROSSLINE_3D", 193, raising=False)
    monkeypatch.setattr(dataset_data.segyio.BinField, "Interval", 1, raising=False)
    monkeypatch.setattr(dataset_data.segyio.BinField, "Format", 2, raising=False)
    monkeypatch.setattr(dataset_data, "_decode_segy_text", lambda value: str(value))

    result = dataset_data.read_seismic_data(tmp_path / "seismic.sgy", max_traces=2)
    assert result["type"] == "seismic"
    assert result["traces"] == [[1.0, 2.0], [3.0, 4.0]]
    assert result["traceHeaders"] == [
        {"inline": 10, "crossline": 20},
        {"inline": 11, "crossline": 20},
    ]
    assert result["traceCount"] == 3
    assert result["scannedTraceCount"] == 2
    assert result["samplesPerTrace"] == 2
    assert result["sampleIntervalUs"] == 2000
    assert result["formatCode"] == 5
    assert result["inlineRange"] == [10, 12]
    assert result["crosslineRange"] == [20, 21]


def test_dataset_associations_live_only_on_concrete_subtype_tables() -> None:
    """Common Dataset stores project ownership; domain FKs belong to child tables."""
    from app.models import (
        Dataset,
        SeismicDataset,
        WellLogDataset,
        CheckshotDataset,
        DeviationDataset,
    )

    assert "project_id" in Dataset.__table__.columns
    assert "survey_id" not in Dataset.__table__.columns
    assert "well_id" not in Dataset.__table__.columns
    assert "survey_id" in SeismicDataset.__table__.columns
    assert "well_id" not in SeismicDataset.__table__.columns
    for subtype in (WellLogDataset, CheckshotDataset, DeviationDataset):
        assert "well_id" in subtype.__table__.columns
        assert "survey_id" not in subtype.__table__.columns


def test_dataset_subtypes_keep_joined_table_inheritance() -> None:
    from sqlalchemy import create_engine, event, select
    from sqlalchemy.orm import Session, configure_mappers
    from app.models import Base, Project, Survey, Well, Dataset, SeismicDataset, WellLogDataset

    configure_mappers()
    engine = create_engine("sqlite:///:memory:")
    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        project = Project(id="project-1", name="Test")
        survey = Survey(id="survey-1", project=project, name="Survey")
        well = Well(id="well-1", project=project, name="Well")
        seismic = SeismicDataset(
            id="seismic-1", project=project, survey=survey, kind="seismic", name="Seismic",
            original_filename="a.sgy", format="sgy", storage_root="/tmp/site_data", storage_key="uploads/a.sgy",
            size_bytes=1, checksum_sha256="abc",
        )
        log = WellLogDataset(
            id="log-1", project=project, well=well, kind="well_log", name="Log",
            original_filename="a.las", format="las", storage_root="/tmp/site_data", storage_key="uploads/a.las",
            size_bytes=1, checksum_sha256="def",
        )
        db.add(project)
        db.commit()
        assert isinstance(db.get(Dataset, "seismic-1"), SeismicDataset)
        assert isinstance(db.get(Dataset, "log-1"), WellLogDataset)
        assert db.get(SeismicDataset, "seismic-1").survey_id == "survey-1"
        assert db.get(WellLogDataset, "log-1").well_id == "well-1"
        db.delete(project)
        db.commit()
        assert db.scalar(select(Project.id)) is None
