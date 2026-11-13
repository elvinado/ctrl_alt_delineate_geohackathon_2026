
import pytest
from fastapi import HTTPException

from app.storage import validate_linked_path


def test_linked_path_accepts_file_inside_configured_input_root(tmp_path, monkeypatch):
    root = tmp_path / "input_data"
    nested = root / "challenge1_challenge2"
    nested.mkdir(parents=True)
    source = nested / "survey.sgy"
    source.write_bytes(b"seismic")
    monkeypatch.setenv("DATASET_ALLOWED_ROOTS", str(root))

    assert validate_linked_path(str(source)) == source.resolve()


def test_linked_path_rejects_file_outside_configured_root(tmp_path, monkeypatch):
    root = tmp_path / "input_data"
    root.mkdir()
    outside = tmp_path / "outside.sgy"
    outside.write_bytes(b"seismic")
    monkeypatch.setenv("DATASET_ALLOWED_ROOTS", str(root))

    with pytest.raises(HTTPException) as error:
        validate_linked_path(str(outside))
    assert error.value.status_code == 403


def test_linked_path_rejects_symlink_escape(tmp_path, monkeypatch):
    root = tmp_path / "input_data"
    root.mkdir()
    outside = tmp_path / "outside.sgy"
    outside.write_bytes(b"seismic")
    link = root / "escape.sgy"
    link.symlink_to(outside)
    monkeypatch.setenv("DATASET_ALLOWED_ROOTS", str(root))

    with pytest.raises(HTTPException) as error:
        validate_linked_path(str(link))
    assert error.value.status_code == 403


def test_linked_path_requires_existing_readable_file(tmp_path, monkeypatch):
    root = tmp_path / "input_data"
    root.mkdir()
    monkeypatch.setenv("DATASET_ALLOWED_ROOTS", str(root))

    with pytest.raises(HTTPException) as error:
        validate_linked_path(str(root / "missing.sgy"))
    assert error.value.status_code == 404


def test_register_linked_seismic_records_reference_without_copy(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.models import Base, Dataset, Project
    from app.storage import register_linked_file

    root = tmp_path / "input_data"
    root.mkdir()
    source = root / "survey.sgy"
    source.write_bytes(b"synthetic-segy-content")
    monkeypatch.setenv("DATASET_ALLOWED_ROOTS", str(root))
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        db.add(Project(id="project-1", name="Test project"))
        db.commit()
        record = register_linked_file(
            db,
            source_path=str(source),
            kind="seismic",
            project_id="project-1",
        )
        stored = db.get(Dataset, record.id)
        assert stored.storage_mode == "linked"
        assert stored.source_path == str(source.resolve())
        assert stored.size_bytes == len(b"synthetic-segy-content")
        assert source.read_bytes() == b"synthetic-segy-content"


def test_deleting_linked_dataset_keeps_source_file(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.models import Base, Project
    from app.resources_api import delete_dataset
    from app.storage import register_linked_file

    root = tmp_path / "input_data"
    root.mkdir()
    source = root / "survey.sgy"
    source.write_bytes(b"keep-this-file")
    monkeypatch.setenv("DATASET_ALLOWED_ROOTS", str(root))
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        db.add(Project(id="project-1", name="Test project"))
        db.commit()
        record = register_linked_file(db, source_path=str(source), kind="seismic", project_id="project-1")
        delete_dataset(record.id, db)
        assert source.read_bytes() == b"keep-this-file"


def test_link_browser_lists_configured_roots_and_supported_files(tmp_path, monkeypatch):
    from app.resources_api import browse_link_local_directory

    root = tmp_path / "input_data"
    nested = root / "challenge"
    nested.mkdir(parents=True)
    (nested / "survey.sgy").write_bytes(b"seismic")
    (nested / "notes.md").write_text("ignore")
    monkeypatch.setenv("DATASET_ALLOWED_ROOTS", str(root))

    roots = browse_link_local_directory(None)
    assert roots["roots"] == [{"name": "input_data", "path": str(root.resolve())}]
    result = browse_link_local_directory(str(nested))
    assert result["current_path"] == str(nested.resolve())
    assert result["parent_path"] == str(root.resolve())
    assert [(item["name"], item["kind"]) for item in result["entries"]] == [("survey.sgy", "file")]


def test_link_browser_rejects_directory_outside_configured_roots(tmp_path, monkeypatch):
    from app.resources_api import browse_link_local_directory

    root = tmp_path / "input_data"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("DATASET_ALLOWED_ROOTS", str(root))

    with pytest.raises(HTTPException) as error:
        browse_link_local_directory(str(outside))
    assert error.value.status_code == 403


def test_linked_roots_require_explicit_configuration(monkeypatch):
    from app.storage import allowed_link_roots

    monkeypatch.delenv("DATASET_ALLOWED_ROOTS", raising=False)
    assert allowed_link_roots() == []
