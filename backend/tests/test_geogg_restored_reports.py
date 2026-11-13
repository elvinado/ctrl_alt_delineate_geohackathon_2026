import numpy as np
import pandas as pd

from scripts.geogg_fresh import evaluation_reports as ev
from scripts.geogg_fresh.stages.s08_evaluation import _build_reports, _evaluate_group
from scripts.geogg_fresh.inference_reporting import _prediction_columns, _safe_name


def _predictions():
    rows = []
    for well_i, well in enumerate(("W1", "W2", "W3")):
        for i in range(20):
            truth = int(i % 2)
            depth = 0.25 + 0.5 * truth
            rows.append({
                "sample_id": f"{well}:{i}", "well": well,
                "desired_twt_ms": 1000 + i * 4,
                "task": "lithology", "input": "stacks", "model": "depth_trend",
                "y_true": truth, "y_pred": depth, "y_pred_class": int(depth >= 0.5),
                "threshold": 0.5, "clipped": False,
            })
            rows.append({
                "sample_id": f"{well}:{i}", "well": well,
                "desired_twt_ms": 1000 + i * 4,
                "task": "lithology", "input": "stacks", "model": "rf",
                "y_true": truth, "y_pred": 0.2 + 0.6 * truth, "y_pred_class": truth,
                "threshold": 0.5, "clipped": False,
            })
    return pd.DataFrame(rows)


def test_specialist_evaluation_helpers_return_well_bootstrap_and_conformal():
    frame = _predictions()
    group = frame[frame.model == "rf"]
    result = ev.well_bootstrap("binary", "roc_auc", group, n_boot=30, seed=4)
    assert result["n_wells"] == 3
    assert result["point"] == 1.0
    assert ev.class_metrics(group.y_true.to_numpy(int), group.y_pred_class.to_numpy(int))["f1_macro"] == 1.0


def test_stage08_builds_specialist_report_tables_from_sealed_predictions():
    reports = _build_reports(
        _predictions(),
        {"n_boot": 30, "seed": 4, "interval_bin_ms": 20.0,
         "conformal_levels": [0.8, 0.9], "_champions": {("lithology", "stacks"): "rf"}},
    )
    assert len(reports["model_performance.csv"]) == 2
    assert len(reports["per_well_metrics.csv"]) == 6
    assert len(reports["per_class_metrics.csv"]) == 4
    assert not reports["paired_comparisons.csv"].empty
    assert not reports["calibration_bins.csv"].empty
    assert "error" in reports["outer_oof_predictions.csv"]


def test_inference_report_helpers_use_safe_names_and_detect_prediction_columns():
    frame = pd.DataFrame({
        "lithology_probability": [0.2], "vsh_prediction": [0.3],
        "phit_prediction": [0.15],
    })
    assert _prediction_columns(frame) == {
        "lithology_probability": "lithology_probability",
        "vsh": "vsh_prediction", "phit": "phit_prediction",
    }
    assert _safe_name("../Well 01") == "Well_01"


def test_group_metrics_include_well_count():
    group = _predictions().query("model == 'rf'")
    result = _evaluate_group(("lithology", "stacks", "rf"), group)
    assert result["n_wells"] == 3
    assert result["outer_roc_auc"] == 1.0


def test_inference_reporting_writes_per_well_tables_and_role_summary(tmp_path):
    from scripts.geogg_fresh.inference_reporting import write_well_prediction_reports

    class Context:
        def __init__(self):
            self.tables = {}
            self.figures = {}
        def save_table(self, frame, name, description):
            self.tables[name] = frame.copy()
        def save_figure(self, fig, name, description, **kwargs):
            import matplotlib.pyplot as plt
            self.figures[name] = True
            plt.close(fig)

    frame = pd.DataFrame({
        "well": ["W1", "W1", "W2"],
        "desired_twt_ms": [1000, 1004, 1000],
        "role": ["development", "development", "reserved"],
        "prediction_provenance": ["final_in_sample", "final_in_sample", "final_blind"],
        "lithology_probability": [0.8, 0.2, 0.7],
    })
    ctx = Context()
    summary = write_well_prediction_reports(
        ctx, frame, source_mode="source_training_survey_patch_replay", model="cnn3d_multitask",
    )
    assert summary["wells"] == 2
    assert summary["well_roles"] == {"development": 1, "reserved": 1}
    assert "per_well/W1_predictions.csv" in ctx.tables
    assert "well_prediction_summary.csv" in ctx.tables


def test_attach_well_roles_uses_source_split_plan(tmp_path):
    from scripts.geogg_fresh.stages.s09_inference import _attach_well_roles

    source = tmp_path / "processing_jobs" / "source-1"
    split_dir = source / "stages" / "05_training_tables" / "tables"
    split_dir.mkdir(parents=True)
    pd.DataFrame({"well": ["W1", "W2"], "role": ["development", "reserved"]}).to_csv(
        split_dir / "split_plan_wells.csv", index=False,
    )
    frame = pd.DataFrame({"well": ["w1", "W2", "W3"]})
    result = _attach_well_roles(frame, source)
    assert result["role"].tolist() == ["development", "reserved", "not_in_split_plan"]
    assert result["prediction_provenance"].tolist() == [
        "final_in_sample", "final_blind", "not_in_split_plan",
    ]
