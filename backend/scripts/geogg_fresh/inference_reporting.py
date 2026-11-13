"""Well-level reporting helpers for released-model inference outputs.

Reports describe predictions only. They do not imply blind validation unless
independent target labels and an explicit evaluation design are available.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_name(value: object) -> str:
    """Create a stable filesystem-safe well identifier."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return name or "unknown_well"


def _prediction_columns(frame: pd.DataFrame) -> dict[str, str]:
    candidates = {
        "lithology_probability": ("lithology_probability", "prediction"),
        "vsh": ("vsh_prediction", "vsh"),
        "phit": ("phit_prediction", "phit"),
    }
    found = {}
    for task, names in candidates.items():
        column = next((name for name in names if name in frame.columns), None)
        if column is not None:
            found[task] = column
    return found


def write_well_prediction_reports(ctx, frame: pd.DataFrame, *,
                                  prediction_columns: dict[str, str] | None = None,
                                  source_mode: str,
                                  model: str) -> dict:
    """Write one row per well, per-well CSVs and compact prediction-track figures."""
    if "well" not in frame:
        return {"wells": 0, "rows": int(len(frame)), "per_well_files": []}
    prediction_columns = prediction_columns or _prediction_columns(frame)
    summary_rows, files, figures = [], [], []
    for well, group in frame.groupby("well", dropna=False, sort=True):
        well_name = str(well)
        safe = _safe_name(well_name)
        ordered = group.copy()
        sort_column = "desired_twt_ms" if "desired_twt_ms" in ordered else (
            "sampled_twt_ms" if "sampled_twt_ms" in ordered else None
        )
        if sort_column:
            ordered = ordered.sort_values(sort_column)
        relative = f"per_well/{safe}_predictions.csv"
        ctx.save_table(ordered, relative,
                       f"Released-model predictions for well {well_name}; source mode: {source_mode}")
        files.append(relative)

        row = {"well": well_name, "n_rows": int(len(group)), "model": model,
               "source_mode": source_mode}
        if "role" in group:
            row["role"] = str(group["role"].iloc[0])
        if "prediction_provenance" in group:
            row["prediction_provenance"] = str(group["prediction_provenance"].iloc[0])
        if sort_column:
            row["twt_min_ms"] = float(pd.to_numeric(group[sort_column], errors="coerce").min())
            row["twt_max_ms"] = float(pd.to_numeric(group[sort_column], errors="coerce").max())
        for task, column in prediction_columns.items():
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            row[f"{task}_n_valid"] = int(finite.size)
            if finite.size:
                row[f"{task}_mean"] = float(finite.mean())
                row[f"{task}_std"] = float(finite.std())
                row[f"{task}_min"] = float(finite.min())
                row[f"{task}_max"] = float(finite.max())
            if task == "lithology_probability":
                row["lithology_positive_fraction"] = float((finite >= 0.5).mean()) if finite.size else np.nan
        summary_rows.append(row)

        if prediction_columns and sort_column:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                tasks = list(prediction_columns.items())
                fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 6), squeeze=False)
                twt = pd.to_numeric(ordered[sort_column], errors="coerce").to_numpy(float)
                for ax, (task, column) in zip(axes[0], tasks):
                    values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(float)
                    valid = np.isfinite(twt) & np.isfinite(values)
                    ax.plot(values[valid], twt[valid], linewidth=0.8)
                    ax.set_title(task.replace("_", " "))
                    ax.set_ylabel("TWT (ms)")
                    ax.invert_yaxis()
                    ax.grid(alpha=0.2)
                fig.suptitle(f"{well_name} — {model} predictions")
                fig.tight_layout()
                figure_name = f"per_well/{safe}_predictions.png"
                ctx.save_figure(fig, figure_name, f"Prediction tracks for well {well_name}", multipanel=True)
                figures.append(figure_name)
            except ImportError:
                # Tables remain complete when matplotlib is not installed.
                pass

    summary = pd.DataFrame(summary_rows)
    ctx.save_table(summary, "well_prediction_summary.csv",
                   "Per-well prediction counts, ranges and summary statistics")
    return {
        "wells": int(len(summary)), "rows": int(len(frame)),
        "per_well_files": files, "figures": figures,
        "prediction_columns": prediction_columns,
        "well_roles": (
            summary["role"].value_counts().to_dict() if "role" in summary else {}
        ),
        "interpretation": (
            "Predictions on source training-survey rows; not independent blind validation."
            if source_mode.startswith("source_training") else
            "Predictions on the selected inference input; no accuracy claim without independent targets."
        ),
    }
