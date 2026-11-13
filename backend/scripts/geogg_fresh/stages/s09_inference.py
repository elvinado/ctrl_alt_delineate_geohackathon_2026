"""09_inference - apply a released tabular or patch-based deep model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..preprocess import apply_transform
from ..metrics import apply_bounds
from ..stage_compat import StageBlocked
from ..provenance import sha256_file


def _input_path(ctx, required_columns):
    """Find exactly one manifest-registered feature table matching the released schema."""
    candidates = []
    for item in getattr(ctx, "input_records", []):
        get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
        path = Path(get("path", "") or "").expanduser().resolve()
        if not path.is_file() or path.suffix.casefold() != ".csv":
            continue
        try:
            header = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if required_columns.issubset(header):
            candidates.append(({
                "dataset_id": get("dataset_id"),
                "name": get("name") or path.name,
                "path": str(path),
            }, path))
    if len(candidates) != 1:
        names = [str(item.get("name") or path.name) for item, path in candidates]
        raise StageBlocked(
            "Expected exactly one registered CSV feature table matching the released model contract; "
            f"found {len(candidates)}: {names}. Select one compatible registered feature table."
        )
    return candidates[0]


def _required_source_columns(features):
    return {"sample_id", "well", "desired_twt_ms", *features}


def _source_training_job(ctx) -> Path:
    """Resolve the release's source job from backend storage, never from a hard-coded path."""
    storage_root = ctx.base.job_dir.parent.parent.resolve()
    release_manifest = json.loads((ctx.release / "manifest.json").read_text(encoding="utf-8"))
    provenance = release_manifest.get("provenance") or {}
    source_id = str(provenance.get("source_processing_job_id") or
                    release_manifest.get("processing_job_id") or
                    release_manifest.get("training_run_id") or "").strip()
    if not source_id or Path(source_id).name != source_id or source_id in {".", ".."}:
        raise StageBlocked("The release does not contain a safe source training job ID")
    source = (storage_root / "processing_jobs" / source_id).resolve()
    expected_root = (storage_root / "processing_jobs").resolve()
    if source.parent != expected_root or source.is_symlink() or not source.is_dir():
        raise StageBlocked(
            "The source training job directory referenced by this release is unavailable. "
            "The release stores model weights/contracts, but not the large Stage 03 patch tensor."
        )
    return source


def _find_stage_file(source_job: Path, stage: str, relative: str) -> Path:
    candidates = [
        source_job / stage / relative,
        source_job / "stages" / stage / relative,
    ]
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve()
    raise StageBlocked(f"Source training artifact is missing: {stage}/{relative}")




def _attach_well_roles(frame: pd.DataFrame, source_job: Path | None) -> pd.DataFrame:
    """Annotate source-survey rows with the frozen well split, when available."""
    result = frame.copy()
    if source_job is None:
        result["role"] = "external_input"
        result["prediction_provenance"] = "external_inference"
        return result
    split_path = _find_stage_file(source_job, "05_training_tables", "tables/split_plan_wells.csv")
    split = pd.read_csv(split_path)
    if not {"well", "role"}.issubset(split.columns):
        raise StageBlocked("Source split_plan_wells.csv must contain well and role columns")
    role_map = {str(well).casefold(): str(role)
                for well, role in zip(split["well"], split["role"])}
    result["role"] = result["well"].astype(str).str.casefold().map(role_map).fillna("not_in_split_plan")
    result["prediction_provenance"] = np.where(
        result["role"].eq("reserved"), "final_blind",
        np.where(result["role"].eq("development"), "final_in_sample", "not_in_split_plan"),
    )
    return result


def _deep_inference(ctx, selection: dict) -> None:
    """Replay the release's source-survey patch locations using the released deep weights.

    The immutable release intentionally excludes the large patch tensor. For a default
    source-survey replay, retrieve the exact Stage 03 tensor/index from the release's
    source processing job and verify it against the copied patch contract.
    """
    import torch
    from .. import models_deep as mdeep

    model_name = str(selection.get("model") or "").casefold()
    if str(selection.get("input") or "patch_stacks") != "patch_stacks":
        raise StageBlocked("Deep model input must be 'patch_stacks' as declared by the released model registry")
    if model_name not in {"tcn", "cnn3d_multitask"}:
        raise StageBlocked(f"Unsupported deep model selector: {model_name!r}")

    reg = pd.read_csv(ctx.release / "training" / "model_registry.csv")
    selected = reg[(reg["status"] == "OK") &
                   (reg["input"].astype(str) == "patch_stacks") &
                   (reg["model"].astype(str) == model_name)]
    if selected.empty:
        raise StageBlocked(f"Release contains no successful {model_name} patch_stacks model")
    requested_task = str(selection.get("task") or "all").casefold()
    supported_tasks = {"lithology", "vsh", "phit"}
    if requested_task != "all" and requested_task not in supported_tasks:
        raise StageBlocked(f"{model_name} supports only {sorted(supported_tasks)}; got {requested_task!r}")
    if requested_task != "all" and requested_task not in set(selected["task"].astype(str)):
        raise StageBlocked(f"Release has no {model_name} artifact for task {requested_task!r}")

    # The deep network is multi-task: one checkpoint produces all three heads.
    checkpoint_rel = Path(str(selected.iloc[0]["final_model_file"]))
    if checkpoint_rel.is_absolute() or ".." in checkpoint_rel.parts or checkpoint_rel.suffix != ".pt":
        raise StageBlocked("Unsafe deep model artifact reference in release registry")
    checkpoint_path = (ctx.release / checkpoint_rel).resolve()
    if ctx.release.resolve() not in checkpoint_path.parents or not checkpoint_path.is_file():
        raise StageBlocked("Deep model checkpoint is missing from the immutable release")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != model_name:
        raise StageBlocked("Deep checkpoint architecture does not match the selected model")
    channels = list(checkpoint.get("channels") or [])
    if not channels or len(channels) != 5:
        raise StageBlocked("Deep checkpoint does not declare the five training stack channels")

    source_job = _source_training_job(ctx)
    contract_path = ctx.release / "contracts" / "patches" / "patch_manifest_v1.json"
    if not contract_path.is_file():
        raise StageBlocked("Release is missing contracts/patches/patch_manifest_v1.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    patch_path = _find_stage_file(source_job, "03_extraction",
                                  f"patches/{contract.get('file', '')}")
    if sha256_file(patch_path) != contract.get("sha256"):
        raise StageBlocked("Source Stage 03 patch tensor checksum does not match the immutable release contract")
    index_path = _find_stage_file(source_job, "03_extraction", "tables/sample_index.csv")
    ctx.add_input(patch_path, "release source job: Stage 03 seismic patches")
    ctx.add_input(index_path, "release source job: Stage 03 sample index")
    index = pd.read_csv(index_path)
    required_index = {"patch_row", "well", "desired_twt_ms", "patch_status"}
    if not required_index.issubset(index.columns):
        raise StageBlocked(f"Source sample_index.csv is missing {sorted(required_index - set(index.columns))}")

    array = np.load(patch_path, mmap_mode="r")
    if list(array.shape[1:]) != [len(contract["channels"]), 5, 5, 21]:
        raise StageBlocked(f"Source patch tensor shape {array.shape} disagrees with the released patch contract")
    channel_index = [contract["channels"].index(ch) for ch in channels]
    valid = (index["patch_status"].astype(str).eq("extracted") &
             (pd.to_numeric(index["patch_row"], errors="coerce") >= 0))
    rows = index.loc[valid].copy().reset_index(drop=True)
    if rows.empty:
        raise StageBlocked("Source sample index contains no extracted patch rows")
    patch_rows = rows["patch_row"].to_numpy(int)
    if np.any(patch_rows >= array.shape[0]):
        raise StageBlocked("Source sample_index.csv contains patch_row values outside the patch tensor")
    patches = np.asarray(array[patch_rows][:, channel_index], dtype=np.float32)
    max_nan = float((ctx.cfg.get("inference") or {}).get("deep_max_nan_fraction", 0.2))
    nan_fraction = 1.0 - np.isfinite(patches).reshape(len(patches), -1).mean(axis=1)
    keep = nan_fraction <= max_nan
    rows, patches = rows.loc[keep].reset_index(drop=True), patches[keep]
    if not len(rows):
        raise StageBlocked(f"No patches satisfy deep_max_nan_fraction={max_nan}")

    scaling = checkpoint.get("scaling")
    if not isinstance(scaling, dict) or not {"centre", "scale", "mode"}.issubset(scaling):
        raise StageBlocked("Deep checkpoint is missing its frozen patch scaling parameters")
    scaled = mdeep.apply_patch_scaling(patches, scaling)
    device_pref = str((ctx.cfg.get("inference") or {}).get("deep_device", "auto"))
    device = mdeep.device_name(device_pref)
    model = mdeep.build(model_name, c_in=len(channels), dropout=0.2).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    predictions = mdeep.predict(model, scaled, device=device,
                                 batch_size=int((ctx.cfg.get("inference") or {}).get("deep_batch_size", 512)))

    output = rows[[c for c in ("patch_row", "well", "twt_bin_index", "desired_twt_ms",
                               "sampled_twt_ms", "il", "xl", "x", "y") if c in rows]].copy()
    output = _attach_well_roles(output, source_job)
    output.insert(0, "sample_id", [f"{w}:{int(k)}" if pd.notna(k) else f"patch:{int(p)}"
                                  for w, k, p in zip(rows["well"], rows.get("twt_bin_index", pd.Series(np.nan, index=rows.index)),
                                                     rows["patch_row"])])
    output["model"] = model_name
    threshold = float(checkpoint.get("lithology_threshold", 0.5))
    for head in ("lithology", "vsh", "phit"):
        if requested_task != "all" and head != requested_task:
            continue
        raw = np.asarray(predictions[head], dtype=float)
        if head == "lithology":
            output["lithology_probability"] = raw
            output["lithology_threshold"] = threshold
            output["lithology_prediction"] = (raw >= threshold).astype(np.int8)
        else:
            bounded, _ = apply_bounds(head, raw)
            output[f"{head}_prediction_raw"] = raw
            output[f"{head}_prediction"] = bounded
            output[f"{head}_clipped"] = raw != bounded

    ctx.save_table(output, "deep_predictions.csv",
                   "Multi-task deep predictions from immutable release weights and frozen patch scaling")
    from ..inference_reporting import write_well_prediction_reports
    well_report = write_well_prediction_reports(
        ctx, output,
        prediction_columns={
            name: column for name, column in {
                "lithology_probability": "lithology_probability",
                "vsh": "vsh_prediction",
                "phit": "phit_prediction",
            }.items() if column in output.columns
        },
        source_mode="source_training_survey_patch_replay",
        model=model_name,
    )
    ctx.save_json({
        "release_id": ctx.release_id, "model": model_name, "tasks": sorted(supported_tasks if requested_task == "all" else {requested_task}),
        "input_mode": "source_training_survey_patch_replay",
        "source_training_job_id": source_job.name, "source_patch_tensor": patch_path.name,
        "source_patch_sha256": contract["sha256"], "source_index": index_path.name,
        "source_rows": int(len(index)), "predicted_rows": int(len(output)),
        "excluded_nan_fraction_rows": int((~keep).sum()), "channels": channels,
        "patch_shape": list(array.shape[1:]), "scaling": scaling, "device": device,
        "well_reporting": well_report,
        "note": "This replays the original training-survey well-path patches. New-survey raw SEG-Y patch extraction is a separate inference-data preparation operation."
    }, "deep_inference_summary.json", "Deep inference provenance")
    ctx.metric("deep_inference_rows", int(len(output)))
    ctx.metric("deep_model", model_name)
    ctx.metric("deep_inference_device", device)


def _source_feature_table(ctx, features: list[str], artifact: dict) -> tuple[None, Path]:
    """Resolve the source survey's feature table using the released resolution contract.

    Stage 05 names its base-resolution table ``training_samples_all.csv``;
    resolution-specific names are reserved for the coarser ablation tables.
    """
    source_job = _source_training_job(ctx)
    transform_name = Path(str(artifact.get("transform_file") or "")).name
    resolution = "4ms"
    if transform_name.startswith("transforms_") and transform_name.endswith(".json"):
        resolution = transform_name[len("transforms_"):-len(".json")]

    candidates = [f"tables/training_samples_{resolution}.csv"]
    manifest_path = ctx.release / "training" / "training_manifest.json"
    if manifest_path.is_file():
        try:
            row_counts = json.loads(manifest_path.read_text(encoding="utf-8")).get("rows", {})
            base_resolution = min((int(str(key).replace("ms", "")) for key in row_counts), default=None)
        except (ValueError, TypeError, json.JSONDecodeError):
            base_resolution = None
        if base_resolution is not None and f"{base_resolution}ms" == resolution:
            candidates.append("tables/training_samples_all.csv")

    candidate = None
    last_error = None
    for relative in candidates:
        try:
            candidate = _find_stage_file(source_job, "05_training_tables", relative)
            break
        except StageBlocked as exc:
            last_error = exc
    if candidate is None:
        raise StageBlocked(
            f"Could not find the Stage 05 feature table for released resolution {resolution}; "
            f"checked {candidates}. {last_error or ''}".strip()
        )
    header = set(pd.read_csv(candidate, nrows=0).columns)
    required = _required_source_columns(features)
    if not required.issubset(header):
        raise StageBlocked(
            f"Source training feature table {candidate.name} does not satisfy the released model schema; "
            f"missing {sorted(required - header)}"
        )
    return None, candidate


def _tabular_inference(ctx, selection: dict) -> None:
    required = ("task", "input", "model")
    if any(not selection.get(k) for k in required):
        raise StageBlocked("inference.model must specify exact task, input and model values")
    reg = pd.read_csv(ctx.release / "training" / "model_registry.csv")
    chosen = reg[(reg["status"] == "OK") & (reg["task"].astype(str) == str(selection["task"])) &
                 (reg["input"].astype(str) == str(selection["input"])) &
                 (reg["model"].astype(str) == str(selection["model"]))]
    if len(chosen) != 1:
        raise StageBlocked(f"Model selector matched {len(chosen)} successful registry rows; expected exactly one")
    row = chosen.iloc[0]
    rel = Path(str(row["final_model_file"]))
    if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".joblib":
        raise StageBlocked("Selected tabular model must reference a released .joblib artifact")
    model_path = (ctx.release / rel).resolve()
    if ctx.release.resolve() not in model_path.parents or not model_path.is_file():
        raise StageBlocked("Selected model artifact is missing or outside the release")
    try:
        import joblib
    except ImportError as exc:
        raise StageBlocked("joblib is required for tabular model inference") from exc
    artifact = joblib.load(model_path)
    features = list(artifact.get("features") or [])
    if not features and artifact.get("base_model") not in {"prior", "depth_trend"}:
        raise StageBlocked("Released model artifact does not declare its predictor feature columns")
    if getattr(ctx, "input_records", []):
        source_record, source = _input_path(ctx, _required_source_columns(features))
        dataset_id = source_record.get("dataset_id")
        input_mode = "registered_feature_table"
    else:
        source_record, source = _source_feature_table(ctx, features, artifact)
        dataset_id = None
        input_mode = "source_training_feature_table_replay"
    ctx.add_input(source, f"inference source feature table ({input_mode})")
    frame = pd.read_csv(source)
    source_job = _source_training_job(ctx) if input_mode == "source_training_feature_table_replay" else None
    frame = _attach_well_roles(frame, source_job)
    if artifact.get("base_model") in {"prior", "depth_trend"}:
        from ..models_tabular import depth_features
        ref = artifact.get("depth_trend_centre_scale")
        if not ref or len(ref) != 2:
            raise StageBlocked("Released baseline model is missing its frozen TWT reference")
        X = depth_features(frame["desired_twt_ms"].to_numpy(float), float(ref[0]), float(ref[1]))
    else:
        transform_rel = Path(str(artifact.get("transform_file", "")))
        if transform_rel.is_absolute() or ".." in transform_rel.parts:
            raise StageBlocked("Unsafe transform reference in released model artifact")
        transform_path = (ctx.release / transform_rel).resolve()
        if not transform_path.is_file():
            transform_path = (ctx.release / "preprocessing" / "transforms" / transform_rel.name).resolve()
        if ctx.release.resolve() not in transform_path.parents or not transform_path.is_file():
            raise StageBlocked("Frozen transform file referenced by model is missing from release")
        transforms = json.loads(transform_path.read_text(encoding="utf-8"))
        try:
            params = transforms["folds"][artifact.get("transform_key")]["params"]
        except (KeyError, TypeError) as exc:
            raise StageBlocked("Frozen transform key is absent from release") from exc
        X = apply_transform(frame, params, features, mode=artifact.get("scaling_mode", "independent"),
                            add_indicators=False).to_numpy(float)
    if X.shape[1] != getattr(artifact["estimator"], "n_features_in_", X.shape[1]):
        raise StageBlocked("Feature matrix width does not match the released estimator")
    from ..models_tabular import predict
    kind = str(artifact["kind"])
    raw = predict(artifact["estimator"], kind, X)
    if kind == "regression":
        pred, n_clipped = apply_bounds(str(selection["task"]), raw)
    else:
        pred, n_clipped = np.asarray(raw, float), 0
    threshold = float(artifact.get("threshold", 0.5) or 0.5)
    out = frame[["sample_id", "well", "desired_twt_ms"]].copy()
    out["task"], out["input"], out["model"] = selection["task"], selection["input"], selection["model"]
    out["prediction_raw"], out["prediction"] = raw, pred
    out["threshold"] = threshold if kind == "binary" else np.nan
    out["prediction_class"] = (pred >= threshold).astype(int) if kind == "binary" else -1
    out["clipped"] = (np.asarray(raw) != np.asarray(pred)) if kind == "regression" else False
    ctx.save_table(out, "predictions.csv", "Predictions from the selected immutable tabular model release")
    from ..inference_reporting import write_well_prediction_reports
    task_columns = {
        "lithology": {"lithology_probability": "prediction"},
        "sand_fluid": {"sand_fluid_probability": "prediction"},
        "vsh": {"vsh": "prediction"},
        "phit": {"phit": "prediction"},
        "swt": {"swt": "prediction"},
    }
    well_report = write_well_prediction_reports(
        ctx, out, prediction_columns=task_columns.get(str(selection["task"]), {}),
        source_mode=input_mode, model=str(selection["model"]),
    )
    ctx.save_json({"release_id": ctx.release_id, "model": selection, "dataset_id": dataset_id,
                   "input_mode": input_mode, "source_file_name": source.name,
                   "source_training_job_id": source_job.name if source_job is not None else None,
                   "rows": int(len(out)), "clipped_predictions": int(n_clipped),
                   "feature_columns": features, "preprocessing": "release-frozen",
                   "well_reporting": well_report},
                  "inference_summary.json", "Stage 09 tabular inference provenance")
    ctx.metric("inference_rows", int(len(out)))
    ctx.metric("clipped_predictions", int(n_clipped))


def run(ctx):
    cfg = ctx.cfg.get("inference", {})
    selection = cfg.get("model", {})
    model_name = str(selection.get("model") or "").casefold()
    input_mode = str(cfg.get("input_mode", "survey_volume")).casefold()
    if input_mode not in {"auto", "source_replay", "survey_volume"}:
        raise StageBlocked("inference.input_mode must be auto, source_replay or survey_volume")
    has_segy = any(
        Path(str(item.get("path", "") if isinstance(item, dict) else getattr(item, "path", "")))
        .suffix.casefold() in {".sgy", ".segy"}
        for item in getattr(ctx, "input_records", [])
    )
    if input_mode == "survey_volume" or (input_mode == "auto" and has_segy):
        # Survey-volume mode resolves the originating foundation job's registered
        # datasets from the model release. The inference job itself needs no inputs.
        from ..volume_inference import run_volume_inference
        run_volume_inference(ctx, selection, cfg)
    elif model_name in {"tcn", "cnn3d_multitask"}:
        _deep_inference(ctx, selection)
    else:
        _tabular_inference(ctx, selection)
