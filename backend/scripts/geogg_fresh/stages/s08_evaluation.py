"""Stage 08: specialist-style held-out-well evaluation from an immutable release.

All reported metrics use sealed outer-test predictions only. Model selection is not
performed here; champions remain those declared by Stage 07 inner-CV results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import evaluation_reports as ev
from ..metrics import task_metrics
from ..stage_compat import StageBlocked


REQUIRED_COLUMNS = {
    "task", "input", "model", "well", "y_true", "y_pred", "threshold",
}
BINARY_TASKS = {"lithology", "sand_fluid"}
DEFAULTS = {"n_boot": 300, "seed": 0, "interval_bin_ms": 200.0,
            "conformal_levels": [0.8, 0.9]}


def _select_predictions(predictions: pd.DataFrame, selectors: dict) -> pd.DataFrame:
    """Apply optional selectors without changing the sealed source rows."""
    selected = predictions
    for column in ("task", "input", "model"):
        value = selectors.get(column)
        if value and value != "*":
            selected = selected[selected[column].astype(str) == str(value)]
    return selected.copy()


def _kind(task: str) -> str | None:
    if task in BINARY_TASKS:
        return "binary"
    if task in {"vsh", "phit", "swt"}:
        return "regression"
    return None


def _evaluate_group(key: tuple, group: pd.DataFrame) -> dict:
    task, input_name, model_name = key
    kind = _kind(str(task))
    if kind is None:
        return {}
    thresholds = pd.to_numeric(group["threshold"], errors="coerce").dropna()
    threshold = float(thresholds.median()) if not thresholds.empty else 0.5
    metrics = task_metrics(kind, group["y_true"].to_numpy(float),
                           group["y_pred"].to_numpy(float), threshold)
    return {
        "task": task, "input": input_name, "model": model_name,
        "kind": kind, "n_rows": int(metrics.get("n", len(group))),
        "n_wells": int(group["well"].nunique()),
        **{f"outer_{name}": value for name, value in metrics.items() if name != "n"},
    }


def _build_reports(selected: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Build specialist evaluation tables without refitting or changing predictions."""
    if "y_pred_class" not in selected:
        thresholds = pd.to_numeric(selected["threshold"], errors="coerce").fillna(0.5)
        selected["y_pred_class"] = np.where(
            selected["task"].isin(BINARY_TASKS),
            (selected["y_pred"].astype(float) >= thresholds).astype(int), -1,
        )
    sort_columns = ["well"] + (["sample_id"] if "sample_id" in selected else [])
    groups = {
        key: frame.sort_values(sort_columns).reset_index(drop=True)
        for key, frame in selected.groupby(["task", "input", "model"], dropna=False)
    }
    n_boot, seed = int(cfg["n_boot"]), int(cfg["seed"])
    performance, per_well, per_class = [], [], []
    calibration, intervals, conformal, paired, beyond_depth = [], [], [], [], []

    registry_path = None
    # Inner-CV champion flags are carried by the release registry, if available.
    champions: dict[tuple[str, str], str] = {}
    # The caller adds the registry-derived map to cfg; it is not inferred from outer results.
    champions.update(cfg.get("_champions", {}))

    for (task, input_name, model_name), frame in groups.items():
        kind = _kind(str(task))
        if kind is None:
            continue
        row = _evaluate_group((task, input_name, model_name), frame)
        primary = "roc_auc" if kind == "binary" else "rmse"
        boot = ev.well_bootstrap(kind, primary, frame, n_boot=n_boot, seed=seed)
        row.update({f"outer_{primary}_ci_lo": boot["ci_lo"],
                    f"outer_{primary}_ci_hi": boot["ci_hi"],
                    "bootstrap_wells": boot["n_wells"],
                    "bootstrap_replicates": boot["n_boot"],
                    "declared_champion": champions.get((str(task), str(input_name))) == str(model_name)})
        registry_row = cfg.get("_registry", {}).get((str(task), str(input_name), str(model_name)), {})
        inner_value = pd.to_numeric(
            pd.Series([registry_row.get(f"inner_{primary}_mean", np.nan)]), errors="coerce"
        ).iloc[0]
        row[f"inner_{primary}"] = float(inner_value) if pd.notna(inner_value) else np.nan
        row["optimism_inner_minus_outer"] = (
            float(inner_value - row.get(f"outer_{primary}", np.nan))
            if pd.notna(inner_value) and pd.notna(row.get(f"outer_{primary}", np.nan)) else np.nan
        )
        well_metrics = ev.per_well(kind, frame)
        if not well_metrics.empty:
            per_well.append(well_metrics.assign(task=task, input=input_name, model=model_name))
            for metric in (primary,):
                vals = pd.to_numeric(well_metrics[metric], errors="coerce").dropna()
                row[f"within_well_{metric}_median"] = float(vals.median()) if len(vals) else np.nan
                row[f"within_well_{metric}_min"] = float(vals.min()) if len(vals) else np.nan
                row[f"within_well_{metric}_max"] = float(vals.max()) if len(vals) else np.nan
        performance.append(row)

        if kind == "binary":
            valid = frame[["y_true", "y_pred_class"]].dropna()
            if not valid.empty:
                cm = ev.class_metrics(valid["y_true"].to_numpy(int), valid["y_pred_class"].to_numpy(int))
                for class_id in (0, 1):
                    per_class.append({
                        "task": task, "input": input_name, "model": model_name,
                        "class_id": class_id,
                        "precision": cm[f"precision_{class_id}"],
                        "recall": cm[f"recall_{class_id}"],
                        "f1": cm[f"f1_{class_id}"],
                        "support": cm[f"support_{class_id}"],
                    })
            if model_name == "depth_trend" or champions.get((str(task), str(input_name))) == str(model_name):
                bins = ev.calibration_bins(frame)
                if not bins.empty:
                    calibration.append(bins.assign(task=task, input=input_name, model=model_name))

        if kind == "regression":
            conformal.append({
                "task": task, "input": input_name, "model": model_name,
                "n_rows": int(len(frame)), **ev.conformal(frame, tuple(cfg["conformal_levels"])),
            })

        if "desired_twt_ms" in frame:
            by_interval = ev.interval_errors(kind, frame, float(cfg["interval_bin_ms"]))
            if not by_interval.empty:
                intervals.append(by_interval.assign(task=task, input=input_name, model=model_name))

    # Paired comparisons are limited to same task/model rows and use sample_id joins.
    for (task, input_name, model_name), frame in groups.items():
        kind = _kind(str(task))
        if kind is None or model_name == "depth_trend" or "sample_id" not in frame:
            continue
        reference_input = input_name if (task, input_name, "depth_trend") in groups else "stacks"
        reference = groups.get((task, reference_input, "depth_trend"))
        if reference is not None:
            primary = "roc_auc" if kind == "binary" else "rmse"
            diff = ev.paired_difference(kind, primary, frame, reference, n_boot=n_boot, seed=seed)
            paired.append({
                "task": task, "comparison": "model_vs_depth_trend",
                "model": model_name, "input": input_name,
                "reference_model": "depth_trend", "reference_input": reference_input,
                "metric": primary, **diff, "verdict": ev.verdict(primary, diff),
            })
        if input_name == "fused@matched" and (task, "stacks@matched", model_name) in groups:
            primary = "roc_auc" if kind == "binary" else "rmse"
            diff = ev.paired_difference(
                kind, primary, frame, groups[(task, "stacks@matched", model_name)],
                n_boot=n_boot, seed=seed,
            )
            paired.append({
                "task": task, "comparison": "fused_vs_stacks_same_model",
                "model": model_name, "input": input_name,
                "reference_model": model_name, "reference_input": "stacks@matched",
                "metric": primary, **diff, "verdict": ev.verdict(primary, diff),
            })

    for task in BINARY_TASKS | {"vsh", "phit", "swt"}:
        patch_model = champions.get((task, "patch_stacks"))
        tabular_model = champions.get((task, "stacks"))
        patch = groups.get((task, "patch_stacks", patch_model)) if patch_model else None
        tabular = groups.get((task, "stacks", tabular_model)) if tabular_model else None
        kind = _kind(task)
        if patch is not None and tabular is not None and "sample_id" in patch and "sample_id" in tabular:
            primary = "roc_auc" if kind == "binary" else "rmse"
            diff = ev.paired_difference(kind, primary, patch, tabular, n_boot=n_boot, seed=seed)
            paired.append({
                "task": task, "comparison": "patch_champion_vs_tabular_stacks_champion",
                "model": patch_model, "input": "patch_stacks",
                "reference_model": tabular_model, "reference_input": "stacks",
                "metric": primary, **diff, "verdict": ev.verdict(primary, diff),
            })

    for (task, input_name, model_name), frame in groups.items():
        kind = _kind(str(task))
        if kind is None or model_name == "depth_trend":
            continue
        reference_input = input_name if (task, input_name, "depth_trend") in groups else "stacks"
        depth = groups.get((task, reference_input, "depth_trend"))
        if depth is not None:
            result = ev.beyond_depth(frame, depth, kind, n_boot=n_boot, seed=seed)
            lo = result.get("ci_lo", np.nan)
            result["verdict"] = (
                "adds information beyond depth" if np.isfinite(lo) and lo > 0 else
                "no detectable information beyond depth" if np.isfinite(lo) else "not testable"
            )
            beyond_depth.append({
                "task": task, "input": input_name, "model": model_name, **result,
            })

    return {
        "model_performance.csv": pd.DataFrame(performance),
        "per_well_metrics.csv": pd.concat(per_well, ignore_index=True) if per_well else pd.DataFrame(),
        "outer_oof_predictions.csv": _prediction_errors(selected),
        "per_class_metrics.csv": pd.DataFrame(per_class),
        "paired_comparisons.csv": pd.DataFrame(paired),
        "beyond_depth.csv": pd.DataFrame(beyond_depth),
        "conformal_intervals.csv": pd.DataFrame(conformal),
        "calibration_bins.csv": pd.concat(calibration, ignore_index=True) if calibration else pd.DataFrame(),
        "errors_by_interval.csv": pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame(),
    }


def _prediction_errors(selected: pd.DataFrame) -> pd.DataFrame:
    """Retain sealed predictions and append transparent error columns."""
    result = selected.copy()
    result["error"] = result["y_pred"].astype(float) - result["y_true"].astype(float)
    result["abs_error"] = result["error"].abs()
    if "y_pred_class" in result:
        result["correct"] = np.where(
            result["task"].isin(BINARY_TASKS),
            (result["y_pred_class"] == result["y_true"]).astype(float), np.nan,
        )
    return result


def _save_figures(ctx, reports: dict[str, pd.DataFrame], groups: dict) -> list[str]:
    """Create compact, task-agnostic report figures; figures never drive model selection."""
    if not callable(getattr(ctx, "save_figure", None)):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    performance = reports["model_performance.csv"]
    if not performance.empty:
        tasks = list(performance["task"].dropna().astype(str).unique())
        fig, axes = plt.subplots(max(1, len(tasks)), 1, figsize=(10, max(4, 3.2 * len(tasks))), squeeze=False)
        for ax, task in zip(axes[:, 0], tasks):
            data = performance[performance["task"].astype(str) == task].copy()
            metric = "roc_auc" if task in BINARY_TASKS else "rmse"
            data = data.sort_values(metric if metric in data else f"outer_{metric}")
            y = np.arange(len(data))
            values = pd.to_numeric(data.get(f"outer_{metric}"), errors="coerce").to_numpy(float)
            lo = pd.to_numeric(data.get(f"outer_{metric}_ci_lo"), errors="coerce").to_numpy(float)
            hi = pd.to_numeric(data.get(f"outer_{metric}_ci_hi"), errors="coerce").to_numpy(float)
            ax.scatter(values, y, label=metric)
            valid = np.isfinite(lo) & np.isfinite(hi)
            if valid.any():
                ax.hlines(y[valid], lo[valid], hi[valid])
            ax.set_yticks(y)
            ax.set_yticklabels([f"{r.model} ({r.input})" for r in data.itertuples()], fontsize=8)
            ax.set_xlabel(f"Held-out {metric} (95% well-bootstrap interval)")
            ax.set_title(str(task))
            ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        ctx.save_figure(fig, "model_comparison_outer.png", "Held-out model performance with well-bootstrap intervals",
                        multipanel=True)
        written.append("model_comparison_outer.png")

    calib = reports["calibration_bins.csv"]
    if not calib.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        for key, frame in calib.groupby(["task", "input", "model"]):
            ax.plot(frame["mean_predicted"], frame["observed_frequency"], marker="o",
                    label=" / ".join(map(str, key)))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
               ylabel="Observed positive frequency", title="Held-out probability calibration")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        ctx.save_figure(fig, "calibration.png", "Reliability curves from sealed held-out predictions")
        written.append("calibration.png")

    intervals = reports["errors_by_interval.csv"]
    if not intervals.empty:
        for (task, input_name, model_name), frame in intervals.groupby(["task", "input", "model"]):
            metric = "error_rate" if task in BINARY_TASKS else "rmse"
            if metric not in frame:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(frame["twt_from_ms"], frame[metric], marker="o")
            ax.set(xlabel="TWT interval start (ms)", ylabel=metric,
                   title=f"{task}: {model_name} ({input_name})")
            ax.grid(alpha=0.2)
            safe = "_".join(str(v).replace("/", "_") for v in (task, input_name, model_name))
            name = f"errors_by_interval/{safe}.png"
            ctx.save_figure(fig, name, "Held-out error by TWT interval")
            written.append(name)
    # Held-out ROC / precision-recall curves for binary tasks.
    binary_groups = [(key, frame) for key, frame in groups.items() if key[0] in BINARY_TASKS]
    if binary_groups:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for (task, input_name, model_name), frame in binary_groups:
            y = frame["y_true"].to_numpy(int)
            p = frame["y_pred"].to_numpy(float)
            fpr, tpr = ev.roc_curve(y, p)
            recall, precision = ev.pr_curve(y, p)
            label = f"{task}: {model_name} ({input_name})"
            axes[0].plot(fpr, tpr, label=label)
            axes[1].plot(recall, precision, label=label)
        axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Held-out ROC")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=6)
        fig.tight_layout()
        ctx.save_figure(fig, "roc_pr_curves.png", "ROC and precision-recall curves from sealed predictions",
                        multipanel=True)
        written.append("roc_pr_curves.png")

        # Confusion matrices use the fold-specific stored threshold/class prediction.
        fig, axes = plt.subplots(len(binary_groups), 1, figsize=(6, max(3, 2.8 * len(binary_groups))),
                                 squeeze=False)
        for ax, ((task, input_name, model_name), frame) in zip(axes[:, 0], binary_groups):
            y = frame["y_true"].to_numpy(int)
            yh = frame["y_pred_class"].to_numpy(int)
            matrix = np.array([[((y == a) & (yh == b)).sum() for b in (0, 1)] for a in (0, 1)], dtype=float)
            norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            for i in (0, 1):
                for j in (0, 1):
                    ax.text(j, i, f"{norm[i, j]:.2f}\\n(n={int(matrix[i, j])})",
                            ha="center", va="center")
            ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted class", ylabel="True class",
                   title=f"{task}: {model_name} ({input_name})")
        fig.tight_layout()
        ctx.save_figure(fig, "confusion_matrices.png", "Held-out confusion matrices", multipanel=True)
        written.append("confusion_matrices.png")

    regression_groups = [(key, frame) for key, frame in groups.items() if key[0] not in BINARY_TASKS]
    if regression_groups:
        fig, axes = plt.subplots(len(regression_groups), 1,
                                 figsize=(6, max(3, 3 * len(regression_groups))), squeeze=False)
        for ax, ((task, input_name, model_name), frame) in zip(axes[:, 0], regression_groups):
            truth = frame["y_true"].to_numpy(float)
            prediction = frame["y_pred"].to_numpy(float)
            ax.scatter(truth, prediction, s=7, alpha=0.35)
            lo = float(np.nanmin([np.nanmin(truth), np.nanmin(prediction)]))
            hi = float(np.nanmax([np.nanmax(truth), np.nanmax(prediction)]))
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            ax.set(xlabel=f"True {task}", ylabel=f"Predicted {task}",
                   title=f"{task}: {model_name} ({input_name})")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        ctx.save_figure(fig, "true_vs_predicted.png", "Held-out regression truth versus prediction",
                        multipanel=True)
        written.append("true_vs_predicted.png")

    per_well = reports["per_well_metrics.csv"]
    if not per_well.empty:
        fig, axes = plt.subplots(per_well["task"].nunique(), 1,
                                 figsize=(10, max(3, 3 * per_well["task"].nunique())), squeeze=False)
        for ax, (task, frame) in zip(axes[:, 0], per_well.groupby("task")):
            metric = "roc_auc" if task in BINARY_TASKS else "rmse"
            frame = frame.sort_values("well")
            labels = frame["well"].astype(str)
            ax.scatter(frame[metric], np.arange(len(frame)))
            ax.set(yticks=np.arange(len(frame)), yticklabels=labels,
                   xlabel=f"Within-well {metric}", title=f"{task}: per-well performance")
            ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        ctx.save_figure(fig, "per_well_metrics.png", "Within-well held-out performance", multipanel=True)
        written.append("per_well_metrics.png")

    beyond = reports["beyond_depth.csv"]
    if not beyond.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = [f"{r.task}: {r.model} ({r.input})" for r in beyond.itertuples()]
        values = pd.to_numeric(beyond["mean_within_well_partial_rho"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(beyond["ci_lo"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(beyond["ci_hi"], errors="coerce").to_numpy(float)
        y = np.arange(len(labels))
        ax.scatter(values, y)
        valid = np.isfinite(lo) & np.isfinite(hi)
        if valid.any():
            ax.hlines(y[valid], lo[valid], hi[valid])
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set(yticks=y, yticklabels=labels, xlabel="Within-well partial Spearman rho",
               title="Prediction information beyond depth")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        ctx.save_figure(fig, "beyond_depth.png", "Within-well partial correlation beyond depth")
        written.append("beyond_depth.png")
    return written


def run(ctx) -> None:
    """Evaluate sealed outer-test predictions and produce specialist-style reports."""
    prediction_path = ctx.release / "training" / "outer_test_predictions.csv"
    if not prediction_path.is_file():
        raise StageBlocked("Selected release has no sealed outer-test predictions")

    predictions = pd.read_csv(prediction_path)
    missing = sorted(REQUIRED_COLUMNS - set(predictions.columns))
    if missing:
        raise StageBlocked(f"Outer prediction table is missing columns: {missing}")

    selectors = ((ctx.cfg.get("inference") or {}).get("evaluation") or {})
    selected = _select_predictions(predictions, selectors)
    if selected.empty:
        raise StageBlocked("No sealed predictions match inference.evaluation selectors")

    eval_cfg = {**DEFAULTS, **(selectors if isinstance(selectors, dict) else {})}
    registry_path = ctx.release / "training" / "model_registry.csv"
    if registry_path.is_file():
        registry = pd.read_csv(registry_path)
        if {"task", "input", "model", "status"}.issubset(registry.columns):
            successful = registry[registry["status"].astype(str).eq("OK")]
            eval_cfg["_registry"] = {
                (str(row.task), str(row.input), str(row.model)): row._asdict()
                for row in successful.itertuples(index=False)
            }
            for row in successful.itertuples():
                if bool(getattr(row, "champion_inner_cv", False)):
                    eval_cfg.setdefault("_champions", {})[(str(row.task), str(row.input))] = str(row.model)

    reports = _build_reports(selected, eval_cfg)
    summary_rows = reports["model_performance.csv"]
    if summary_rows.empty:
        raise StageBlocked("No supported evaluation tasks were found in the selected sealed predictions")

    for filename, frame in reports.items():
        if filename == "model_performance.csv":
            continue
        description = {
            "model_performance.csv": "Held-out metrics and 95% well-bootstrap intervals for every candidate",
            "per_well_metrics.csv": "Metrics calculated separately within each held-out well",
            "outer_oof_predictions.csv": "Sealed outer-test predictions with signed and absolute errors",
            "per_class_metrics.csv": "Per-class precision, recall, F1 and support at stored fold thresholds",
            "paired_comparisons.csv": "Paired same-row comparisons with whole-well bootstrap intervals",
            "beyond_depth.csv": "Within-well partial Spearman analysis controlling for depth trend and TWT",
            "conformal_intervals.csv": "Regression residual intervals and leave-one-well-out coverage",
            "calibration_bins.csv": "Quantile reliability bins for selected binary candidates",
            "errors_by_interval.csv": "Held-out prediction errors grouped by TWT interval",
        }[filename]
        ctx.save_table(frame, filename, description)

    ctx.save_table(
        reports["model_performance.csv"], "model_performance.csv",
        "Held-out metrics and 95% well-bootstrap intervals for every candidate",
    )

    groups = {
        key: frame for key, frame in selected.groupby(["task", "input", "model"], dropna=False)
    }
    figures = _save_figures(ctx, reports, groups)
    summary = {
        "release_id": ctx.release_id,
        "source": "training/outer_test_predictions.csv",
        "prediction_rows": int(len(selected)),
        "groups": int(len(summary_rows)),
        "well_ids": sorted(selected["well"].astype(str).unique().tolist()),
        "evaluation_policy": "sealed outer-test predictions only; no refitting or model reselection",
        "figures": figures,
        "tables": {name: int(len(frame)) for name, frame in reports.items()},
        "bootstrap": {"replicates": int(eval_cfg["n_boot"]), "seed": int(eval_cfg["seed"]),
                      "unit": "well"},
    }
    ctx.save_json(summary, "evaluation_summary.json", "Stage 08 evaluation provenance and report index")
    ctx.metric("prediction_rows", int(len(selected)))
    ctx.metric("evaluation_groups", int(len(summary_rows)))
    ctx.metric("evaluation_figures", len(figures))
