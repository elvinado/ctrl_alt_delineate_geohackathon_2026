import hashlib
import json
from pathlib import Path

import pytest

from scripts.geogg_fresh.stage_compat import StageBlocked
from scripts.geogg_fresh.volume_inference import (
    _resolve_originating_stack_records, _summarize_tile_valid_counts,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_originating_stack_resolution_uses_inventory_without_app_imports(tmp_path, monkeypatch):
    foundation = tmp_path / "foundation"
    manifest_dir = foundation / "stages" / "00_inventory"
    manifest_dir.mkdir(parents=True)
    inputs = []
    names = {
        "full": "survey_full.sgy", "near": "survey_near.sgy",
        "mid": "survey_mid.sgy", "far": "survey_far.sgy",
        "ultrafar": "survey_ultrafar.sgy",
    }
    for role, name in names.items():
        source = tmp_path / name
        source.write_bytes((role + "-segy").encode())
        inputs.append({"dataset_id": f"id-{role}", "kind": "seismic", "name": name,
                       "path": str(source), "sha256": _sha(source), "metadata": {}})
    (manifest_dir / "data_manifest.json").write_text(json.dumps({"inputs": inputs}))

    # If any future implementation tries to discover/import the APGCE app,
    # this test will fail: manifest resolution must be self-contained.
    import scripts.geogg_fresh.volume_inference as module
    monkeypatch.setattr(module, "_ensure_backend_import_path",
                        lambda: (_ for _ in ()).throw(AssertionError("app import attempted")), raising=False)

    records = _resolve_originating_stack_records(foundation, "foundation-id")
    assert set(records) == set(names)
    assert records["full"]["dataset_id"] == "id-full"
    assert records["ultrafar"]["path"].name == "survey_ultrafar.sgy"


def test_originating_stack_resolution_rejects_checksum_mismatch(tmp_path):
    foundation = tmp_path / "foundation"
    manifest_dir = foundation / "stages" / "00_inventory"
    manifest_dir.mkdir(parents=True)
    inputs = []
    for role in ("full", "near", "mid", "far", "ultrafar"):
        source = tmp_path / f"survey_{role}.sgy"
        source.write_bytes(role.encode())
        inputs.append({"dataset_id": f"id-{role}", "kind": "seismic",
                       "name": source.name, "path": str(source),
                       "sha256": "0" * 64, "metadata": {}})
    (manifest_dir / "data_manifest.json").write_text(json.dumps({"inputs": inputs}))
    with pytest.raises(StageBlocked, match="SHA-256"):
        _resolve_originating_stack_records(foundation, "foundation-id")


def test_originating_stack_resolution_rejects_missing_manifest_file(tmp_path):
    foundation = tmp_path / "foundation"
    manifest_dir = foundation / "stages" / "00_inventory"
    manifest_dir.mkdir(parents=True)
    inputs = []
    for role in ("full", "near", "mid", "far", "ultrafar"):
        inputs.append({"dataset_id": f"id-{role}", "kind": "seismic",
                       "name": f"survey_{role}.sgy", "path": str(tmp_path / f"survey_{role}.sgy"),
                       "sha256": "0" * 64, "metadata": {}})
    (manifest_dir / "data_manifest.json").write_text(json.dumps({"inputs": inputs}))
    with pytest.raises(StageBlocked, match="unavailable file"):
        _resolve_originating_stack_records(foundation, "foundation-id")


def test_tile_valid_count_summary_avoids_volume_mask_scan():
    counts = {"0": 120, "1": 80, "2": 0}
    completed = {"0", "1", "2"}
    assert _summarize_tile_valid_counts(counts, completed, 3, 500) == 200


def test_tile_valid_count_summary_rejects_incomplete_checkpoint():
    with pytest.raises(StageBlocked, match="incomplete"):
        _summarize_tile_valid_counts({"0": 10}, {"0"}, 2, 100)


def test_tile_valid_count_summary_rejects_counts_above_live_voxels():
    with pytest.raises(StageBlocked, match="exceeds live survey voxels"):
        _summarize_tile_valid_counts({"0": 101}, {"0"}, 1, 100)


def _region_geometry():
    import numpy as np
    from types import SimpleNamespace
    grid = np.arange(30, dtype=np.int32).reshape(5, 6)
    grid[1, 2] = -1
    return SimpleNamespace(
        grid=grid, il_min=100, il_step=10, xl_min=200, xl_step=5, ns=21,
        affine={"a0": 1000.0, "a1": 2.0, "a2": 0.0,
                "b0": 2000.0, "b1": 0.0, "b2": 3.0},
        affine_residual_max=0.0,
        bin_spacing=lambda: (3.0, 20.0),
        ilxl_to_xy=lambda il, xl: (1000.0 + 2.0 * il, 2000.0 + 3.0 * xl),
    )


def test_volume_region_inline_crossline_returns_cropped_grid():
    from scripts.geogg_fresh.volume_inference import _resolve_volume_region
    geom = _region_geometry()
    i0, i1, j0, j1, grid, meta = _resolve_volume_region(
        geom, {"volume_region": {"inline": [110, 130], "crossline": [205, 220]}}
    )
    assert (i0, i1, j0, j1) == (1, 4, 1, 5)
    assert grid.shape == (3, 4)
    assert grid[0, 1] == -1  # preserve dead bins inside the ROI
    assert meta["inline_range"] == [110, 130]
    assert meta["crossline_range"] == [205, 220]


def test_volume_region_xy_bbox_masks_live_bins_and_crops():
    from scripts.geogg_fresh.volume_inference import _resolve_volume_region
    geom = _region_geometry()
    # affine XY: x=1000+2*IL, y=2000+3*XL
    i0, i1, j0, j1, grid, meta = _resolve_volume_region(
        geom, {"volume_region": {"xy": {"x_min": 1220, "x_max": 1260,
                                         "y_min": 2600, "y_max": 2615}}}
    )
    assert (i0, i1, j0, j1) == (1, 4, 0, 2)
    assert grid.shape == (3, 2)
    assert meta["xy"]["selection_method"] == "stage00_affine_transform"
    assert (grid >= 0).sum() == 6


def test_volume_region_rejects_non_intersecting_range():
    from scripts.geogg_fresh.volume_inference import _resolve_volume_region
    with pytest.raises(StageBlocked, match="does not intersect"):
        _resolve_volume_region(_region_geometry(), {"volume_region": {"inline": [999, 999]}})


def test_volume_region_rejects_xy_on_materially_non_affine_geometry():
    from scripts.geogg_fresh.volume_inference import _resolve_volume_region
    geom = _region_geometry()
    geom.affine_residual_max = 1.0
    with pytest.raises(StageBlocked, match="materially non-affine"):
        _resolve_volume_region(geom, {"volume_region": {"xy": {
            "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}}})


def test_vectorized_tile_patch_gather_matches_stage03_patch_contract():
    import numpy as np
    from types import SimpleNamespace
    from scripts.geogg_fresh.extraction import patch_window
    from scripts.geogg_fresh.volume_inference import (
        _patch_batch, _read_tile_trace_blocks, _extract_patches_from_tile,
    )

    class FakeGeometry:
        def __init__(self, grid, il_min, xl_min, il_step, xl_step, ns):
            self.grid = grid
            self.il_min, self.xl_min = il_min, xl_min
            self.il_step, self.xl_step, self.ns = il_step, xl_step, ns

        def trace_index(self, il, xl):
            il = np.asarray(il, dtype=np.int64)
            xl = np.asarray(xl, dtype=np.int64)
            ii = (il - self.il_min) // self.il_step
            jj = (xl - self.xl_min) // self.xl_step
            ok = ((il - self.il_min) % self.il_step == 0) & ((xl - self.xl_min) % self.xl_step == 0)
            ok &= (ii >= 0) & (ii < self.grid.shape[0]) & (jj >= 0) & (jj < self.grid.shape[1])
            out = np.full(il.shape, -1, dtype=np.int64)
            out[ok] = self.grid[ii[ok], jj[ok]]
            return out

    class FakeSegy:
        def __init__(self, samples):
            self.samples = samples

        def read_traces(self, indices):
            indices = np.asarray(indices, dtype=np.int64)
            out = np.full((indices.size, self.samples.shape[1]), np.nan, dtype=np.float32)
            valid = indices >= 0
            out[valid] = self.samples[indices[valid]]
            return out

    rng = np.random.default_rng(42)
    ni, nj, ns = 9, 10, 31
    grid = np.arange(ni * nj, dtype=np.int64).reshape(ni, nj)
    grid[3, 4] = -1
    ref = SimpleNamespace(il_min=100, il_step=2, xl_min=200, xl_step=3)
    channels = ["full", "near"]
    geometries, open_files = {}, {}
    for role in channels:
        channel_grid = grid.copy()
        channel_grid[5, 5] = -1
        geom = FakeGeometry(channel_grid, 100, 200, 2, 3, ns)
        geometries[role] = geom
        samples = rng.normal(size=(ni * nj, ns)).astype(np.float32)
        open_files[role] = FakeSegy(samples)
    offsets = patch_window(5, 5, 21)
    ci = np.array([0, 2, 4, 8, 7, 3], dtype=np.int64)
    cj = np.array([0, 3, 5, 9, 1, 4], dtype=np.int64)
    ck = np.array([0, 4, 15, 30, 20, 10], dtype=np.int64)

    expected = _patch_batch(open_files, geometries, ref, ci, cj, ck, offsets, channels)
    blocks, cache_i0, cache_j0 = _read_tile_trace_blocks(
        open_files, geometries, ref, channels, 0, ni, 0, nj, halo=2)
    actual = _extract_patches_from_tile(
        blocks, channels, cache_i0, cache_j0, ci, cj, ck, offsets)

    assert np.array_equal(np.isfinite(actual), np.isfinite(expected))
    finite = np.isfinite(expected)
    assert np.max(np.abs(actual[finite] - expected[finite])) == 0.0


def test_vectorized_voxel_batches_cover_each_live_sample_once():
    import numpy as np
    from scripts.geogg_fresh.volume_inference import _iter_voxel_batches

    live_i = np.array([2, 4, 7], dtype=np.int64)
    live_j = np.array([1, 3, 5], dtype=np.int64)
    batches = list(_iter_voxel_batches(live_i, live_j, 7, 5))
    got_i = np.concatenate([batch[0] for batch in batches])
    got_j = np.concatenate([batch[1] for batch in batches])
    got_k = np.concatenate([batch[2] for batch in batches])
    assert len(got_i) == 3 * 7
    assert all(len(batch[0]) <= 5 for batch in batches)
    assert len(set(zip(got_i.tolist(), got_j.tolist(), got_k.tolist()))) == 21
    assert np.array_equal(got_k, np.tile(np.arange(7), 3))


def test_deep_predict_cpu_fp32_returns_all_heads():
    import numpy as np
    import torch
    from scripts.geogg_fresh.models_deep import predict

    class TinyModel(torch.nn.Module):
        def forward(self, x):
            z = x.mean(dim=(1, 2, 3, 4))
            return z, torch.stack([torch.sigmoid(z), torch.sigmoid(-z)], dim=1)

    model = TinyModel()
    values = np.ones((5, 2, 5, 5, 21), dtype=np.float32)
    output = predict(model, values, device="cpu", batch_size=2, pin_memory=True, precision="fp32")
    assert set(output) == {"lithology", "vsh", "phit"}
    assert all(value.shape == (5,) for value in output.values())
    assert np.allclose(output["lithology"], 1.0 / (1.0 + np.exp(-1.0)), atol=1e-6)
