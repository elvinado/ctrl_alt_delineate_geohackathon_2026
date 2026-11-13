from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest

from scripts.geogg_fresh.inference_runner import _resolve_release


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resolve_release_checks_manifest_and_payload(tmp_path):
    job = tmp_path / "processing_jobs" / "job-1"
    job.mkdir(parents=True)
    release = tmp_path / "model_releases" / "release-1"
    release.mkdir(parents=True)
    artifact = release / "training" / "model_registry.csv"
    artifact.parent.mkdir()
    artifact.write_text("task,input,model\nvsh,stacks,extratrees\n")
    (release / "manifest.json").write_text(json.dumps({
        "model_release_id": "release-1", "status": "ready", "foundation_job_id": "job-1",
        "files": {"training/model_registry.csv": {"bytes": artifact.stat().st_size, "sha256": _sha(artifact)}}
    }))
    assert _resolve_release(job, "release-1") == release.resolve()


def test_resolve_release_rejects_tampered_payload(tmp_path):
    job = tmp_path / "processing_jobs" / "job-1"
    job.mkdir(parents=True)
    release = tmp_path / "model_releases" / "release-1"
    release.mkdir(parents=True)
    artifact = release / "weights.bin"
    artifact.write_bytes(b"original")
    (release / "manifest.json").write_text(json.dumps({
        "model_release_id": "release-1", "status": "ready", "foundation_job_id": "job-1",
        "files": {"weights.bin": {"bytes": 8, "sha256": _sha(artifact)}}
    }))
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        _resolve_release(job, "release-1")


def test_resolve_release_rejects_path_traversal(tmp_path):
    job = tmp_path / "processing_jobs" / "job-1"
    job.mkdir(parents=True)
    with pytest.raises(ValueError, match="simple release ID"):
        _resolve_release(job, "../outside")


def test_stage09_required_columns_are_derived_from_released_features():
    from scripts.geogg_fresh.stages.s09_inference import _required_source_columns

    assert _required_source_columns(["stack_full", "stack_near"]) == {
        "sample_id", "well", "desired_twt_ms", "stack_full", "stack_near"
    }


def test_deep_model_architectures_accept_released_patch_contract():
    torch = pytest.importorskip("torch")
    from scripts.geogg_fresh import models_deep

    x = torch.zeros((2, 5, 5, 5, 21), dtype=torch.float32)
    for name in ("tcn", "cnn3d_multitask"):
        model = models_deep.build(name, c_in=5, dropout=0.2)
        lith, reg = model(x)
        assert tuple(lith.shape) == (2,)
        assert tuple(reg.shape) == (2, 2)


def test_source_training_job_is_resolved_from_release_provenance(tmp_path):
    from types import SimpleNamespace
    from scripts.geogg_fresh.stages.s09_inference import _source_training_job

    storage = tmp_path
    job = storage / "processing_jobs" / "inference-1"
    source = storage / "processing_jobs" / "training-1"
    job.mkdir(parents=True)
    source.mkdir(parents=True)
    release = storage / "model_releases" / "geogg-training-1"
    release.mkdir(parents=True)
    (release / "manifest.json").write_text(json.dumps({
        "provenance": {"source_processing_job_id": "training-1"}
    }))
    ctx = SimpleNamespace(
        base=SimpleNamespace(job_dir=job),
        release=release,
        release_id="geogg-training-1",
    )
    assert _source_training_job(ctx) == source.resolve()


def test_source_feature_table_resolution_uses_release_transform_resolution(tmp_path):
    from types import SimpleNamespace
    from scripts.geogg_fresh.stages.s09_inference import _source_feature_table

    storage = tmp_path
    inference_job = storage / "processing_jobs" / "inference-1"
    source_job = storage / "processing_jobs" / "training-1"
    (inference_job).mkdir(parents=True)
    table_dir = source_job / "05_training_tables" / "tables"
    table_dir.mkdir(parents=True)
    table = table_dir / "training_samples_4ms.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame(columns=["sample_id", "well", "desired_twt_ms", "feature_a"]).to_csv(table, index=False)
    release = storage / "model_releases" / "release-1"
    release.mkdir(parents=True)
    (release / "manifest.json").write_text(json.dumps({
        "provenance": {"source_processing_job_id": "training-1"}
    }))
    ctx = SimpleNamespace(
        base=SimpleNamespace(job_dir=inference_job),
        release=release,
        release_id="release-1",
    )
    record, resolved = _source_feature_table(ctx, ["feature_a"],
                                              {"transform_file": "06_feature_selection/transforms/transforms_4ms.json"})
    assert record is None
    assert resolved == table.resolve()


def test_source_feature_table_uses_stage05_all_name_for_base_resolution(tmp_path):
    from types import SimpleNamespace
    from scripts.geogg_fresh.stages.s09_inference import _source_feature_table

    storage = tmp_path
    inference_job = storage / "processing_jobs" / "inference-1"
    source_job = storage / "processing_jobs" / "training-1"
    inference_job.mkdir(parents=True)
    table_dir = source_job / "stages" / "05_training_tables" / "tables"
    table_dir.mkdir(parents=True)
    table = table_dir / "training_samples_all.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame(columns=["sample_id", "well", "desired_twt_ms", "feature_a"]).to_csv(table, index=False)
    release = storage / "model_releases" / "release-1"
    release.mkdir(parents=True)
    (release / "manifest.json").write_text(json.dumps({
        "provenance": {"source_processing_job_id": "training-1"}
    }))
    (release / "training").mkdir()
    (release / "training" / "training_manifest.json").write_text(json.dumps({
        "rows": {"4ms": 4004, "12ms": 1345, "24ms": 678}
    }))
    ctx = SimpleNamespace(
        base=SimpleNamespace(job_dir=inference_job),
        release=release,
        release_id="release-1",
    )
    record, resolved = _source_feature_table(
        ctx, ["feature_a"],
        {"transform_file": "06_feature_selection/transforms/transforms_4ms.json"},
    )
    assert record is None
    assert resolved == table.resolve()


def test_specialist_context_keeps_input_records_separate_from_added_inputs(tmp_path):
    from types import SimpleNamespace
    from geogg_fresh.stage_compat import SpecialistStageContext
    from geogg_fresh.contracts import InputRecord

    source = tmp_path / "patches.npy"
    source.write_bytes(b"patches")
    base = SimpleNamespace(
        run_id="job-1",
        layout=SimpleNamespace(stages=tmp_path / "stages"),
        stage_dir=tmp_path / "stages" / "09_inference",
        inputs=[InputRecord(dataset_id="dataset-1", kind="seismic", format="npy",
                            name="registered.npy", path=str(source))],
        add_input=lambda path, role, hash_it=True: "a" * 64,
    )
    ctx = SpecialistStageContext(base, {}, "09_inference")
    assert len(ctx.input_records) == 1
    assert ctx.inputs == {}
    ctx.add_input(source, "source training patch tensor")
    assert str(source.resolve()) in ctx.inputs
    assert ctx.inputs[str(source.resolve())]["sha256"] == "a" * 64


def test_volume_stack_resolution_uses_registered_dataset_names(tmp_path):
    from types import SimpleNamespace
    from scripts.geogg_fresh.volume_inference import resolve_stack_records

    records = []
    for role in ("full", "near", "mid", "far", "ultrafar"):
        path = tmp_path / f"Challenge_Survey_{role}_Stack.sgy"
        path.write_bytes(b"segy")
        records.append(SimpleNamespace(path=str(path), name=path.name, dataset_id=f"ds-{role}", metadata={}))
    resolved = resolve_stack_records(records)
    assert set(resolved) == {"full", "near", "mid", "far", "ultrafar"}
    assert resolved["ultrafar"]["dataset_id"] == "ds-ultrafar"


def test_volume_stack_resolution_rejects_missing_required_channel(tmp_path):
    from types import SimpleNamespace
    from scripts.geogg_fresh.volume_inference import resolve_stack_records
    from scripts.geogg_fresh.stage_compat import StageBlocked

    path = tmp_path / "survey_full_stack.sgy"
    path.write_bytes(b"segy")
    records = [SimpleNamespace(path=str(path), name=path.name, dataset_id="ds-full", metadata={})]
    with pytest.raises(StageBlocked, match="missing"):
        resolve_stack_records(records, required=["full", "near"])


def test_volume_feature_builder_reproduces_released_vsh_predictors():
    np = pytest.importorskip("numpy")
    from scripts.geogg_fresh.volume_inference import _features_from_patches

    rng = np.random.default_rng(7)
    patches = rng.normal(size=(3, 5, 5, 5, 21)).astype(np.float32)
    features = [
        "ultrafar__spatial_std_c", "far__spatial_std_c", "mid_far__rms_nd",
        "ultrafar__grad_xy_c", "far__rms_short", "mid_far__rms_nd",
    ]
    frame = _features_from_patches(patches, ["full", "near", "mid", "far", "ultrafar"],
                                   list(dict.fromkeys(features)), 4.0)
    assert list(frame.columns) == list(dict.fromkeys(features))
    assert frame.shape == (3, 5)
    assert np.isfinite(frame["mid_far__rms_nd"]).all()


def test_volume_centres_cover_live_grid_cells_and_time_samples():
    import numpy as np
    from types import SimpleNamespace
    from scripts.geogg_fresh.volume_inference import _iter_centres

    geom = SimpleNamespace(grid=np.array([[0, -1], [1, 2]]), ns=3)
    batches = list(_iter_centres(geom, 4))
    ci = np.concatenate([x[0] for x in batches])
    cj = np.concatenate([x[1] for x in batches])
    ck = np.concatenate([x[2] for x in batches])
    assert len(ci) == 9
    assert list(zip(ci, cj, ck)) == [
        (0, 0, 0), (0, 0, 1), (0, 0, 2),
        (1, 0, 0), (1, 0, 1), (1, 0, 2),
        (1, 1, 0), (1, 1, 1), (1, 1, 2),
    ]


def test_volume_patch_batch_preserves_channel_and_centre_sample_order():
    import numpy as np
    from types import SimpleNamespace
    from scripts.geogg_fresh.volume_inference import _patch_batch
    from scripts.geogg_fresh.extraction import patch_window

    class Geometry:
        il_min = 100
        il_step = 1
        xl_min = 200
        xl_step = 1
        ns = 25

        def trace_index(self, il, xl):
            il = np.asarray(il)
            xl = np.asarray(xl)
            return np.where((il >= 100) & (il < 103) & (xl >= 200) & (xl < 203),
                            (il - 100) * 3 + (xl - 200), -1)

    class Reader:
        def __init__(self, offset):
            self.offset = offset

        def read_traces(self, trace_ids):
            trace_ids = np.asarray(trace_ids)
            values = np.full((len(trace_ids), 25), np.nan, dtype=np.float32)
            valid = trace_ids >= 0
            values[valid] = trace_ids[valid, None] + np.arange(25, dtype=np.float32)[None, :] + self.offset
            return values

    geom = Geometry()
    channels = ["full", "near"]
    readers = {"full": Reader(0), "near": Reader(1000)}
    geometries = {name: geom for name in channels}
    ci = np.array([1])
    cj = np.array([1])
    ck = np.array([12])
    patches = _patch_batch(readers, geometries, geom, ci, cj, ck, patch_window(5, 5, 21), channels)
    assert patches.shape == (1, 2, 5, 5, 21)
    assert patches[0, 0, 2, 2, 10] == 4 + 12
    assert patches[0, 1, 2, 2, 10] == 1000 + 4 + 12
    assert np.isnan(patches[0, 0, 0, 0]).all()


def test_volume_geometry_uses_full_stack_reference_and_allows_channel_grid_differences():
    import numpy as np
    from types import SimpleNamespace
    from scripts.geogg_fresh.volume_inference import _same_sampling

    full = SimpleNamespace(dt_ms=4.0, t0_ms=0.0, ns=21, il_min=100, xl_min=200)
    partial = SimpleNamespace(dt_ms=4.0, t0_ms=0.0, ns=21, il_min=102, xl_min=198)
    _same_sampling(full, partial, "near")
    partial.dt_ms = 2.0
    from scripts.geogg_fresh.stage_compat import StageBlocked
    with pytest.raises(StageBlocked, match="time sampling"):
        _same_sampling(full, partial, "near")


def test_tabular_volume_does_not_apply_deep_global_nan_filter():
    import numpy as np
    from scripts.geogg_fresh.volume_inference import _patch_keep_mask

    patches = np.full((2, 5, 5, 5, 21), np.nan, dtype=np.float32)
    patches[0, 0, 0, 0, 0] = 1.0
    assert _patch_keep_mask(patches, deep=False, max_nan_fraction=0.2).tolist() == [True, True]
    assert _patch_keep_mask(patches, deep=True, max_nan_fraction=0.2).tolist() == [False, False]


def test_volume_inference_has_no_apgce_app_import_discovery():
    import inspect
    import scripts.geogg_fresh.volume_inference as module

    source = inspect.getsource(module)
    assert "_ensure_backend_import_path" not in source
    assert "from app.database" not in source
    assert "from app.models" not in source
    assert "from app.storage" not in source


def test_volume_inference_resolves_release_originating_foundation_job(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from scripts.geogg_fresh.volume_inference import _resolve_foundation_job

    storage = tmp_path
    jobs = storage / "processing_jobs"
    inference = jobs / "inference-1"
    foundation = jobs / "foundation-1"
    inference.mkdir(parents=True)
    foundation.mkdir(parents=True)
    seismic = tmp_path / "survey_full_stack.sgy"
    seismic.write_bytes(b"exact SEG-Y bytes")
    digest = hashlib.sha256(seismic.read_bytes()).hexdigest()
    stages = ["00_inventory", "01_well_ties", "02_labels", "03_extraction",
              "04_features", "05_training_tables", "06_feature_selection", "07_models"]
    (foundation / "geogg_foundations_summary.json").write_text(json.dumps({
        "status": "succeeded",
        "stages": [{"stage_id": stage, "status": "COMPLETED"} for stage in stages],
    }))
    (foundation / "provenance").mkdir()
    (foundation / "provenance" / "00_inventory.json").write_text(json.dumps({
        "inputs": [{"dataset_id": "dataset-full", "kind": "seismic", "name": seismic.name,
                    "path": str(seismic), "sha256": digest}],
    }))
    ctx = SimpleNamespace(base=SimpleNamespace(job_dir=inference))
    records = {"full": {"path": seismic}}
    assert _resolve_foundation_job(ctx, records, "foundation-1") == foundation.resolve()


def test_volume_inference_does_not_compare_against_nonexistent_role_keys_in_inventory_provenance(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from scripts.geogg_fresh.volume_inference import _resolve_foundation_job
    from scripts.geogg_fresh.stage_compat import StageBlocked

    jobs = tmp_path / "processing_jobs"
    inference = jobs / "inference-1"
    foundation = jobs / "foundation-1"
    inference.mkdir(parents=True)
    foundation.mkdir(parents=True)
    seismic = tmp_path / "survey_full_stack.sgy"
    seismic.write_bytes(b"new SEG-Y bytes")
    stages = ["00_inventory", "01_well_ties", "02_labels", "03_extraction",
              "04_features", "05_training_tables", "06_feature_selection", "07_models"]
    (foundation / "geogg_foundations_summary.json").write_text(json.dumps({
        "status": "succeeded",
        "stages": [{"stage_id": stage, "status": "COMPLETED"} for stage in stages],
    }))
    (foundation / "provenance").mkdir()
    (foundation / "provenance" / "00_inventory.json").write_text(json.dumps({
        "inputs": [{"role": "seismic:full", "sha256": hashlib.sha256(b"old SEG-Y bytes").hexdigest()}],
    }))
    ctx = SimpleNamespace(base=SimpleNamespace(job_dir=inference))
    # The foundation provenance schema stores dataset records, not role-keyed
    # ``seismic:full`` entries. The data_manifest resolver owns file hash checks.
    assert _resolve_foundation_job(ctx, {"full": {"path": seismic}}, "foundation-1") == foundation.resolve()


def test_volume_inference_rejects_missing_release_foundation_id(tmp_path):
    from types import SimpleNamespace
    from scripts.geogg_fresh.volume_inference import _resolve_foundation_job
    from scripts.geogg_fresh.stage_compat import StageBlocked

    jobs = tmp_path / "processing_jobs"
    inference = jobs / "inference-1"
    inference.mkdir(parents=True)
    ctx = SimpleNamespace(base=SimpleNamespace(job_dir=inference))
    with pytest.raises(StageBlocked, match="no valid foundation_job_id"):
        _resolve_foundation_job(ctx, {}, "")


def test_volume_patch_parity_checks_foundation_stage03_artifacts(tmp_path):
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from scripts.geogg_fresh.volume_inference import _verify_stage03_patch_parity
    from scripts.geogg_fresh.extraction import patch_window

    class Geometry:
        il_min = 0
        il_step = 1
        xl_min = 0
        xl_step = 1
        ns = 21
        t0_ms = 0.0
        dt_ms = 4.0

        def trace_index(self, il, xl):
            il, xl = np.asarray(il), np.asarray(xl)
            valid = (il >= 0) & (il < 5) & (xl >= 0) & (xl < 5)
            return np.where(valid, il * 5 + xl, -1)

    class Reader:
        def read_traces(self, trace_ids):
            trace_ids = np.asarray(trace_ids)
            values = np.full((len(trace_ids), 21), np.nan, dtype=np.float32)
            valid = trace_ids >= 0
            values[valid] = trace_ids[valid, None] + np.arange(21, dtype=np.float32)[None, :]
            return values

    foundation = tmp_path / "foundation-1"
    stage = foundation / "stages" / "03_extraction"
    (stage / "tables").mkdir(parents=True)
    (stage / "patches").mkdir()
    geom = Geometry()
    channels = ["full", "near"]
    geometries = {name: geom for name in channels}
    readers = {name: Reader() for name in channels}
    offsets = patch_window(5, 5, 21)
    expected = __import__("scripts.geogg_fresh.volume_inference", fromlist=["_patch_batch"])._patch_batch(
        readers, geometries, geom, np.array([2]), np.array([2]), np.array([10]), offsets, channels
    )
    patch_path = stage / "patches" / "patches_4ms_v1.npy"
    np.save(patch_path, expected, allow_pickle=False)
    import hashlib
    digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    (stage / "patches" / "patch_manifest_v1.json").write_text(json.dumps({
        "file": patch_path.name, "sha256": digest, "channels": channels,
    }))
    pd.DataFrame([{
        "patch_row": 0, "patch_status": "extracted", "grid_i": 2, "grid_j": 2,
        "desired_twt_ms": 40.0,
    }]).to_csv(stage / "tables" / "sample_index.csv", index=False)
    class Ctx:
        def add_input(self, *args, **kwargs):
            return None
    result = _verify_stage03_patch_parity(
        Ctx(), foundation, geom, geometries, readers, channels, offsets,
    )
    assert result["rows_compared"] == 1
    assert result["nan_pattern_equal"] is True
    assert result["max_abs_difference"] == 0.0



def test_stage09_volume_mode_resolves_originating_datasets_without_job_inputs(monkeypatch):
    from types import SimpleNamespace
    import scripts.geogg_fresh.volume_inference as volume
    from scripts.geogg_fresh.stages.s09_inference import run

    called = {}
    monkeypatch.setattr(volume, "run_volume_inference",
                        lambda ctx, selection, cfg: called.update(selection=selection, cfg=cfg))
    ctx = SimpleNamespace(cfg={"inference": {"model": {
        "task": "all", "input": "patch_stacks", "model": "cnn3d_multitask"
    }}}, input_records=[])
    run(ctx)
    assert called["selection"]["model"] == "cnn3d_multitask"


def test_stage09_source_replay_must_be_explicit():
    from types import SimpleNamespace
    from scripts.geogg_fresh.stages.s09_inference import run

    ctx = SimpleNamespace(cfg={"inference": {
        "input_mode": "source_replay",
        "model": {"task": "all", "input": "patch_stacks", "model": "cnn3d_multitask"}
    }}, input_records=[])
    # Dispatch is intentionally routed to the replay implementation. Patch it to
    # prove source replay is opt-in without requiring model artifacts in this unit test.
    import scripts.geogg_fresh.stages.s09_inference as stage
    called = []
    original = stage._deep_inference
    stage._deep_inference = lambda context, selection: called.append(selection)
    try:
        run(ctx)
    finally:
        stage._deep_inference = original
    assert called == [ctx.cfg["inference"]["model"]]


def test_manifest_stack_resolution_uses_inventory_dataset_ids_without_legacy_paths(tmp_path):
    import json
    from scripts.geogg_fresh.volume_inference import resolve_stack_records

    inputs = []
    for role in ("full", "near", "mid", "far", "ultrafar"):
        inputs.append({
            "path": f"/obsolete/windows-or-home/path/Survey_{role}_Stack.sgy",
            "name": f"Survey_{role}_Stack.sgy",
            "dataset_id": f"dataset-{role}",
            "kind": "seismic",
            "sha256": f"hash-{role}",
        })
    resolved = resolve_stack_records(inputs, require_existing=False)
    assert set(resolved) == {"full", "near", "mid", "far", "ultrafar"}
    assert resolved["full"]["dataset_id"] == "dataset-full"
    assert resolved["ultrafar"]["dataset_id"] == "dataset-ultrafar"


def test_volume_spatial_tiles_cover_each_live_cell_once():
    import numpy as np
    from types import SimpleNamespace
    from scripts.geogg_fresh.volume_inference import _iter_spatial_tiles

    grid = np.array([[0, -1, 1, 2, 3], [4, 5, 6, -1, 7], [-1, 8, 9, 10, 11]])
    geom = SimpleNamespace(grid=grid)
    covered = np.zeros(grid.shape, dtype=np.uint8)
    tile_count = 0
    for _, (i0, i1, j0, j1), (ii, jj) in _iter_spatial_tiles(geom, 2, 2):
        tile_count += 1
        covered[ii, jj] += 1
        assert np.all(ii >= i0) and np.all(ii < i1)
        assert np.all(jj >= j0) and np.all(jj < j1)
    assert tile_count == 6
    assert np.array_equal(covered, (grid >= 0).astype(np.uint8))


def test_volume_cell_batches_preserve_all_samples_without_duplicates():
    import numpy as np
    from scripts.geogg_fresh.volume_inference import _iter_cell_batches

    live_i = np.array([0, 0, 2, 4])
    live_j = np.array([1, 3, 0, 2])
    batches = list(_iter_cell_batches(live_i, live_j, ns=3, batch_size=6))
    rows = [(int(i), int(j), int(k)) for ci, cj, ck in batches for i, j, k in zip(ci, cj, ck)]
    assert len(rows) == 12
    assert len(set(rows)) == 12
    assert rows[:6] == [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 3, 0), (0, 3, 1), (0, 3, 2)]


def test_volume_checkpoint_is_atomic_and_round_trips(tmp_path):
    import json
    from scripts.geogg_fresh.volume_inference import _checkpoint_write

    path = tmp_path / "checkpoints" / "state.json"
    _checkpoint_write(path, {"fingerprint": "abc", "completed_tiles": ["0", "1"]})
    assert json.loads(path.read_text()) == {"fingerprint": "abc", "completed_tiles": ["0", "1"]}
    assert not path.with_suffix(".json.tmp").exists()


def test_volume_patch_extraction_is_independent_of_batch_partition():
    import numpy as np
    from scripts.geogg_fresh.volume_inference import _patch_batch
    from scripts.geogg_fresh.extraction import patch_window

    class Geometry:
        il_min, il_step, xl_min, xl_step, ns = 10, 1, 20, 1, 25
        def trace_index(self, il, xl):
            il, xl = np.asarray(il), np.asarray(xl)
            return np.where((il >= 10) & (il < 14) & (xl >= 20) & (xl < 24),
                            (il - 10) * 4 + (xl - 20), -1)

    class Reader:
        def read_traces(self, ids):
            ids = np.asarray(ids)
            out = np.full((len(ids), 25), np.nan, dtype=np.float32)
            valid = ids >= 0
            out[valid] = ids[valid, None] * 100 + np.arange(25, dtype=np.float32)
            return out

    geom = Geometry()
    centres = [(1, 1, 8), (1, 2, 9), (2, 1, 10), (2, 2, 11)]
    ci, cj, ck = (np.array([row[i] for row in centres]) for i in range(3))
    offsets = patch_window(5, 5, 21)
    readers = {"full": Reader()}
    geometries = {"full": geom}
    grouped = _patch_batch(readers, geometries, geom, ci, cj, ck, offsets, ["full"])
    partitioned = np.concatenate([
        _patch_batch(readers, geometries, geom, ci[i:i+1], cj[i:i+1], ck[i:i+1], offsets, ["full"])
        for i in range(len(ci))
    ])
    assert np.array_equal(np.isfinite(grouped), np.isfinite(partitioned))
    assert np.allclose(grouped, partitioned, equal_nan=True)


def test_stage08_evaluates_only_selected_sealed_prediction_groups(tmp_path):
    pd = pytest.importorskip("pandas")
    from geogg_fresh.stages.s08_evaluation import run

    release = tmp_path / "release"
    training = release / "training"
    training.mkdir(parents=True)
    pd.DataFrame([
        {"task": "lithology", "input": "patch_stacks", "model": "cnn3d_multitask",
         "well": "W1", "y_true": 1, "y_pred": 0.9, "threshold": 0.5},
        {"task": "lithology", "input": "patch_stacks", "model": "cnn3d_multitask",
         "well": "W2", "y_true": 0, "y_pred": 0.1, "threshold": 0.5},
        {"task": "vsh", "input": "patch_stacks", "model": "cnn3d_multitask",
         "well": "W1", "y_true": 0.2, "y_pred": 0.25, "threshold": float("nan")},
    ]).to_csv(training / "outer_test_predictions.csv", index=False)

    class Context:
        def __init__(self):
            self.release = release
            self.release_id = "release-1"
            self.cfg = {"inference": {"evaluation": {"task": "lithology"}}}

        def save_table(self, frame, name, description):
            self.metrics = frame.copy()

        def save_json(self, payload, name, description):
            self.summary = payload

        def metric(self, key, value):
            setattr(self, key, value)

    ctx = Context()
    run(ctx)
    assert len(ctx.metrics) == 1
    assert ctx.metrics.iloc[0]["task"] == "lithology"
    assert ctx.summary["prediction_rows"] == 2
    assert ctx.summary["well_ids"] == ["W1", "W2"]


def test_specialist_context_rejects_output_path_traversal(tmp_path):
    from geogg_fresh.context import StageContext
    from geogg_fresh.stage_compat import SpecialistStageContext

    base = StageContext(stage_id="09_inference", job_dir=tmp_path, config={})
    ctx = SpecialistStageContext(base, {}, "09_inference")

    with pytest.raises(ValueError, match="inside"):
        ctx.save_text("outside", "../outside.txt", "test")
    with pytest.raises(ValueError, match="inside"):
        ctx.save_table(__import__("pandas").DataFrame({"x": [1]}), "../outside.csv", "test")
