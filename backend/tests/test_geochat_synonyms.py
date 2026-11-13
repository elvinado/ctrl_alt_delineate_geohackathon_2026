from app.geochat.skills.synonyms import (
    canonical_dataset_kind,
    normalize_skill_parameters,
)


def test_well_log_aliases_normalize_to_canonical_kind():
    for alias in ("well_log", "well logs", "well LOG", "well-log", "LAS data", "well log data"):
        assert canonical_dataset_kind(alias) == "well_log"


def test_all_dataset_type_aliases_normalize():
    cases = {
        "SEG-Y": "seismic",
        "seismic cube": "seismic",
        "check shot": "checkshot",
        "checkshot data": "checkshot",
        "well deviation": "deviation",
        "directional survey": "deviation",
    }
    for alias, expected in cases.items():
        assert canonical_dataset_kind(alias) == expected


def test_inventory_parameters_normalize_lists_and_deduplicate():
    result = normalize_skill_parameters(
        "data_inventory",
        {"data_type": ["well logs", "well_log", "LAS"]},
    )
    assert result["data_type"] == ["well_log"]


def test_non_inventory_parameters_are_unchanged():
    parameters = {"data_type": "well logs"}
    assert normalize_skill_parameters("dataset_analysis", parameters) == parameters


def test_data_inventory_renderer_uses_stable_table_and_keeps_filename_inline() -> None:
    from app.geochat.skills.data_inventory.formatting import render_markdown

    response = render_markdown({
        "count": 2,
        "datasets": [
            {
                "filename": "Example_Seismic.sgy",
                "kind": "seismic",
                "survey": "Survey 1",
                "status": "ready",
                "registered": True,
                "file_accessible": True,
            },
            {
                "filename": "m-1.las",
                "kind": "well_log",
                "well": "M-1",
                "status": "ready",
                "registered": True,
                "file_accessible": False,
            },
        ],
    })

    assert "| Dataset | Type | Well / Survey | Description |" in response
    assert "Status" not in response
    assert "Registration" not in response
    assert "Source file" not in response
    assert "`Example_Seismic.sgy` | Seismic | Survey 1 | Seismic volume data" in response
    assert "`m-1.las` | Well Log | M-1 | Well log curves and measurements" in response
    assert "**Total: 2 datasets**" in response


def test_data_inventory_renderer_reports_empty_results() -> None:
    from app.geochat.skills.data_inventory.formatting import render_markdown

    response = render_markdown({"count": 0, "datasets": []})
    assert "No matching datasets are registered" in response
    assert "**Total: 0 datasets**" in response


def test_data_inventory_renderer_uses_explicit_description_when_available() -> None:
    from app.geochat.skills.data_inventory.formatting import render_markdown

    response = render_markdown({
        "count": 1,
        "datasets": [{
            "filename": "custom.sgy",
            "kind": "seismic",
            "survey": "Survey 1",
            "description": "Pre-stack depth migrated volume",
        }],
    })

    assert "Pre-stack depth migrated volume" in response
    assert "Seismic volume data" not in response


def test_inventory_description_uses_parsed_las_header(monkeypatch) -> None:
    from pathlib import Path
    from app.geochat.skills.data_inventory import descriptions

    monkeypatch.setattr(descriptions, "_read_file_header", lambda *_: {
        "readable": True,
        "well": [{"mnemonic": "WELL", "value": "M-1"}],
        "index_mnemonic": "DEPT",
        "index_unit": "m",
        "curves": [
            {"mnemonic": "GR", "unit": "API"},
            {"mnemonic": "RHOB", "unit": "g/cc"},
        ],
        "data_row_count": 1250,
    })

    result = descriptions.dataset_description("well_log", "las", Path("m-1.las"))
    assert result == "LAS well log; well M-1; index DEPT (m); curves: GR (API), RHOB (g/cc); 1,250 samples"


def test_inventory_description_uses_parsed_segy_header(monkeypatch) -> None:
    from pathlib import Path
    from app.geochat.skills.data_inventory import descriptions

    monkeypatch.setattr(descriptions, "_read_file_header", lambda *_: {
        "readable": True,
        "trace_count": 1200,
        "sample_count_per_trace": 1500,
        "binary_header": {"sample_interval_us": 2000},
        "textual_headers": ["C01 CLIENT: APGCE\nC02 AREA: TEST SURVEY"],
    })

    result = descriptions.dataset_description("seismic", "sgy", Path("volume.sgy"))
    assert "1,200 traces" in result
    assert "1,500 samples per trace" in result
    assert "2 ms sample interval" in result
    assert "CLIENT: APGCE" in result


def test_inventory_description_uses_parsed_txt_columns_and_row_count(monkeypatch) -> None:
    from pathlib import Path
    from app.geochat.skills.data_inventory import descriptions

    monkeypatch.setattr(descriptions, "_read_well_table", lambda *_args, **_kwargs: {
        "kind": "checkshot",
        "columns": [
            {"name": "MDKB", "unit": "m"},
            {"name": "TWT", "unit": "ms"},
            {"name": "TVDSS", "unit": "m"},
        ],
        "row_count": 42,
    })

    result = descriptions.dataset_description("checkshot", "txt", Path("m-1.txt"))
    assert result == "Checkshot table; columns: MDKB (m), TWT (ms), TVDSS (m); 42 rows"


def test_inventory_description_falls_back_when_header_parsing_fails(monkeypatch) -> None:
    from pathlib import Path
    from app.geochat.skills.data_inventory import descriptions

    def fail(*_args, **_kwargs):
        raise ValueError("invalid header")

    monkeypatch.setattr(descriptions, "_read_file_header", fail)
    assert descriptions.dataset_description("well_log", "las", Path("broken.las")) == (
        "Well log curves and measurements"
    )
    assert descriptions.dataset_description("seismic", "sgy", None) == "Seismic volume data"


def test_inventory_intent_recognizes_broad_well_data_as_all_well_dataset_types():
    from app.geochat.skills.data_inventory.intent import parse_inventory_request

    assert parse_inventory_request("what well data do I have?") == {
        "data_type": ["well_log", "deviation", "checkshot"]
    }
    assert parse_inventory_request("show me the well datasets") == {
        "data_type": ["well_log", "deviation", "checkshot"]
    }
    assert parse_inventory_request("what data do I have for well M-1?") == {
        "well_name": "m-1"
    }


def test_inventory_intent_prefers_specific_well_dataset_type():
    from app.geochat.skills.data_inventory.intent import parse_inventory_request

    assert parse_inventory_request("what well logs do I have?") == {
        "data_type": "well_log"
    }
    assert parse_inventory_request("list checkshot data for M-1") == {
        "data_type": "checkshot",
        "well_name": "m-1",
    }
    assert parse_inventory_request("show deviation data") == {
        "data_type": "deviation"
    }


def test_inventory_intent_does_not_capture_conceptual_questions():
    from app.geochat.skills.data_inventory.intent import parse_inventory_request

    assert parse_inventory_request("what is well log data?") is None
    assert parse_inventory_request("explain checkshot surveys") is None


def test_inventory_intent_supports_short_and_survey_scoped_queries():
    from app.geochat.skills.data_inventory.intent import parse_inventory_request

    assert parse_inventory_request("any well data?") == {
        "data_type": ["well_log", "deviation", "checkshot"]
    }
    assert parse_inventory_request("well data for M-1") == {
        "data_type": ["well_log", "deviation", "checkshot"],
        "well_name": "m-1",
    }
    assert parse_inventory_request("what seismic data do I have for survey 1?") == {
        "data_type": "seismic",
        "survey_name": "1",
    }


def test_well_data_phrase_does_not_become_well_name_filter():
    from app.geochat.skills.data_inventory.intent import parse_inventory_request, parse_well_name_filter

    assert parse_inventory_request("What well data do I have?") == {
        "data_type": ["well_log", "deviation", "checkshot"]
    }
    assert parse_well_name_filter("What well data do I have?") is None
    assert parse_well_name_filter("What data do I have for well M-1?") == "m-1"
    assert parse_well_name_filter("M-1 deviation") == "m-1"


def test_well_survey_inventory_intent_is_separate_from_dataset_inventory():
    from app.geochat.skills.well_survey_inventory.intent import parse_entity_inventory_request

    assert parse_entity_inventory_request("What well do I have?") == {"entity_type": "wells"}
    assert parse_entity_inventory_request("What wells are available?") == {"entity_type": "wells"}
    assert parse_entity_inventory_request("What survey do I have?") == {"entity_type": "surveys"}
    assert parse_entity_inventory_request("List all surveys") == {"entity_type": "surveys"}
    assert parse_entity_inventory_request("What well data do I have?") is None
    assert parse_entity_inventory_request("What survey data do I have?") is None


def test_well_survey_inventory_handler_is_project_scoped(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.models import Base, Project, Well, Survey
    from app.geochat.skills.well_survey_inventory.handler import execute

    engine = create_engine(f"sqlite:///{tmp_path / 'entities.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all([
            Project(id="p1", name="Project 1"),
            Project(id="p2", name="Project 2"),
        ])
        db.flush()
        db.add_all([
            Well(id="w1", project_id="p1", name="M-1", uwi="UWI-001"),
            Well(id="w2", project_id="p2", name="N-2"),
            Survey(id="s1", project_id="p1", name="Survey 1", description="Main survey"),
            Survey(id="s2", project_id="p2", name="Survey 2"),
        ])
        db.commit()

        wells = execute(db, "p1", {"entity_type": "wells"})
        surveys = execute(db, "p1", {"entity_type": "surveys"})
        assert wells["count"] == 1
        assert wells["items"] == [{"name": "M-1", "uwi": "UWI-001"}]
        assert surveys["count"] == 1
        assert surveys["items"][0]["name"] == "Survey 1"



def test_well_survey_inventory_formatter_renders_both_entity_types():
    from app.geochat.skills.well_survey_inventory.formatting import render_markdown

    wells = render_markdown({
        "entity_type": "wells",
        "count": 1,
        "items": [{"name": "M-1", "uwi": "UWI-001"}],
    })
    assert "**Project wells**" in wells
    assert "| M-1 | UWI-001 |" in wells
    assert "**Total: 1 well**" in wells

    surveys = render_markdown({
        "entity_type": "surveys",
        "count": 1,
        "items": [{"name": "Survey 1", "description": "Main survey", "crs": "EPSG:4326"}],
    })
    assert "**Project surveys**" in surveys
    assert "| Survey 1 | Main survey | EPSG:4326 |" in surveys
    assert "**Total: 1 survey**" in surveys
