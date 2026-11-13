from __future__ import annotations

from pathlib import Path

from geogg_fresh.config import cfg_get, resolve_path
from geogg_fresh.stage_compat import SpecialistStageContext
from geogg_fresh.stages import s02_labels, s03_extraction
from geogg_fresh.runner import STAGE_MODULES


def test_stage_c_modules_import_and_expose_run():
    assert callable(s02_labels.run)
    assert callable(s03_extraction.run)
    assert list(STAGE_MODULES) == [
        "00_inventory", "01_well_ties", "02_labels", "03_extraction",
        "04_features", "05_training_tables", "06_feature_selection", "07_models",
    ]


def test_config_helpers_resolve_project_relative_paths(tmp_path):
    cfg = {"paths": {"project_root": str(tmp_path)}}
    assert cfg_get({"labels": {"policy": {"x": 3}}}, "labels.policy.x") == 3
    assert resolve_path(cfg, "logs/well.las") == tmp_path / "logs" / "well.las"
    assert resolve_path(cfg, None) is None


def test_specialist_context_records_blocker_and_outputs(tmp_path):
    class Base:
        run_id = "job-1"
        layout = type("Layout", (), {"stages": tmp_path / "stages"})()
        stage_dir = tmp_path / "stages" / "03_extraction"

        def add_input(self, path, role, hash_it=True):
            return None

    ctx = SpecialistStageContext(Base(), {"figures": {"dpi": 100}}, "03_extraction")
    ctx.block("full stack unavailable")
    assert ctx.blockers == ["full stack unavailable"]
    assert ctx.table_path("qc.csv").parent == ctx.tables_dir


def test_expected_wells_fall_back_to_stage00_inventory():
    import pandas as pd
    from geogg_fresh.config import resolve_expected_wells

    inventory = pd.DataFrame({"well": ["EO-1", "m-1", "eo-1", None]})
    assert resolve_expected_wells({"wells": {"expected": []}}, inventory=inventory) == ["eo-1", "m-1"]


def test_expected_wells_prefers_explicit_configuration():
    import pandas as pd
    from geogg_fresh.config import resolve_expected_wells

    inventory = pd.DataFrame({"well": ["inventory-well"]})
    assert resolve_expected_wells(
        {"wells": {"expected": ["Configured-Well"]}}, inventory=inventory
    ) == ["configured-well"]


def test_expected_wells_can_fall_back_to_stage02_labels():
    import pandas as pd
    from geogg_fresh.config import resolve_expected_wells

    labels = pd.DataFrame({"well": ["p-1", "no-1", "p-1"]})
    assert resolve_expected_wells({"wells": {"expected": []}}, labels=labels) == ["p-1", "no-1"]
