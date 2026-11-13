"""GeoGG Stage 12: exploratory 3-D geobody extraction from canonical Stage 11 NPY."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

# The worker is launched as a file by the backend runner. In that mode Python
# adds ``scripts/`` (not the backend root) to sys.path, so package imports such
# as ``scripts.geogg_fresh`` otherwise fail when SEG-Y export begins.
_STAGE_DIR = Path(__file__).resolve().parent
_SCRIPT_DIR = _STAGE_DIR.parents[2]
_BACKEND_ROOT = _STAGE_DIR.parents[3]
for _import_root in (str(_BACKEND_ROOT), str(_SCRIPT_DIR)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

import numpy as np

OUTPUT = "geogg_geobody_preflight.json"
SUMMARY = "geogg_geobody_summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Stage 12 geobody requires PyYAML") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise RuntimeError("Stage 12 geobody YAML root must be a mapping")
    return value


def _check(name: str, ok: bool, detail: str, *, required: bool = True) -> dict:
    return {"check": name, "status": "PASS" if ok else ("BLOCKED" if required else "CONDITIONAL"), "detail": detail}


def _safe_product_path(jobs_root: Path, row: dict) -> Path:
    rel = str(row.get("storage_root_relative_path") or row.get("path") or "").strip()
    if not rel:
        raise RuntimeError(f"Product {row.get('product')} has no storage-root-relative path")
    path = (jobs_root.parent / rel).resolve()
    try:
        path.relative_to(jobs_root.parent.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Product path escapes backend storage root: {rel}") from exc
    return path


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _extract(score: np.ndarray, valid: np.ndarray, tau_low: float, tau_high: float,
             min_voxels: int, min_columns: int, min_extent_cells: int,
             connectivity: int = 6) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Seeded hysteresis and deterministic component filtering on [IL, XL, time]."""
    from scipy import ndimage

    if score.ndim != 3 or valid.shape != score.shape:
        raise ValueError(f"Expected matching 3-D score/valid arrays; got {score.shape} and {valid.shape}")
    if not (0.0 <= tau_low <= tau_high <= 1.0):
        raise ValueError("Thresholds must satisfy 0 <= tau_low <= tau_high <= 1")
    if min_voxels < 1 or min_columns < 1 or min_extent_cells < 1:
        raise ValueError("Minimum component filters must be positive integers")
    if connectivity not in (6, 18, 26):
        raise ValueError("Connectivity must be 6, 18, or 26")
    structure = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else (2 if connectivity == 18 else 3))
    valid = np.asarray(valid, dtype=bool) & np.isfinite(score)
    low = valid & (score >= tau_low)
    seeds = valid & (score >= tau_high)
    raw_labels, n_components = ndimage.label(low, structure=structure)
    if n_components == 0:
        labels = np.zeros(score.shape, dtype=np.int32)
        labels[~valid] = -1
        return labels, np.zeros(score.shape, dtype=np.uint8), []

    flat_labels = raw_labels.ravel()
    sizes = np.bincount(flat_labels, minlength=n_components + 1)
    seed_counts = np.bincount(flat_labels[seeds.ravel()], minlength=n_components + 1)

    # Number of unique IL/XL columns touched by each component.
    lateral_count = score.shape[0] * score.shape[1]
    flat_valid_idx = np.flatnonzero(flat_labels)
    comp_ids = flat_labels[flat_valid_idx].astype(np.int64, copy=False)
    lateral_ids = flat_valid_idx // score.shape[2]
    unique_pairs = np.unique(comp_ids * lateral_count + lateral_ids)
    columns = np.bincount(unique_pairs // lateral_count, minlength=n_components + 1)

    # Bounding extents in IL/XL grid cells.
    ii = (flat_valid_idx // (score.shape[1] * score.shape[2])).astype(np.int64)
    jj = ((flat_valid_idx // score.shape[2]) % score.shape[1]).astype(np.int64)
    i_min = np.full(n_components + 1, score.shape[0], dtype=np.int64)
    j_min = np.full(n_components + 1, score.shape[1], dtype=np.int64)
    i_max = np.full(n_components + 1, -1, dtype=np.int64)
    j_max = np.full(n_components + 1, -1, dtype=np.int64)
    np.minimum.at(i_min, comp_ids, ii)
    np.minimum.at(j_min, comp_ids, jj)
    np.maximum.at(i_max, comp_ids, ii)
    np.maximum.at(j_max, comp_ids, jj)

    component_rows = []
    keep_component = np.zeros(n_components + 1, dtype=bool)
    for cid in range(1, n_components + 1):
        if seed_counts[cid] == 0:
            decision, reason = "REJECTED", "ISOLATED_NO_SEED"
        elif sizes[cid] < min_voxels:
            decision, reason = "REJECTED", "SMALL_BELOW_MIN_VOXELS"
        elif columns[cid] < min_columns:
            decision, reason = "REJECTED", "FEW_DISTINCT_COLUMNS"
        elif max(i_max[cid] - i_min[cid] + 1, j_max[cid] - j_min[cid] + 1) < min_extent_cells:
            decision, reason = "REJECTED", "LATERAL_EXTENT_BELOW_MIN"
        else:
            decision, reason = "RETAINED", "RETAINED"
            keep_component[cid] = True
        component_rows.append({
            "raw_component": cid, "voxel_count": int(sizes[cid]),
            "seed_voxel_count": int(seed_counts[cid]), "distinct_columns": int(columns[cid]),
            "il_min_index": int(i_min[cid]), "il_max_index": int(i_max[cid]),
            "xl_min_index": int(j_min[cid]), "xl_max_index": int(j_max[cid]),
            "decision": decision, "reason": reason,
        })

    # Relabel retained components in deterministic scan-order (raw labels are scan ordered).
    id_map = np.zeros(n_components + 1, dtype=np.int32)
    kept_ids = np.flatnonzero(keep_component)
    id_map[kept_ids] = np.arange(1, len(kept_ids) + 1, dtype=np.int32)
    labels = id_map[raw_labels]
    labels[~valid] = -1
    candidate = (labels > 0).astype(np.uint8)
    for row in component_rows:
        row["geobody_id"] = int(id_map[row["raw_component"]]) if row["decision"] == "RETAINED" else None
    return labels, candidate, component_rows


def _catalogue(labels: np.ndarray, score: np.ndarray, valid: np.ndarray,
               components: list[dict], geometry: dict) -> list[dict]:
    """Vectorized per-body statistics; coordinates remain in declared grid/time units."""
    shape = labels.shape
    flat_labels = labels.ravel()
    voxel_idx = np.flatnonzero(flat_labels > 0)
    if voxel_idx.size == 0:
        return []
    ids = flat_labels[voxel_idx].astype(np.int64, copy=False)
    n_bodies = int(ids.max())
    il = voxel_idx // (shape[1] * shape[2])
    xl = (voxel_idx // shape[2]) % shape[1]
    kk = voxel_idx % shape[2]
    vals = np.asarray(score).ravel()[voxel_idx].astype(np.float64, copy=False)
    counts = np.bincount(ids, minlength=n_bodies + 1)
    sums = np.bincount(ids, weights=vals, minlength=n_bodies + 1)
    sum_il = np.bincount(ids, weights=il, minlength=n_bodies + 1)
    sum_xl = np.bincount(ids, weights=xl, minlength=n_bodies + 1)
    sum_k = np.bincount(ids, weights=kk, minlength=n_bodies + 1)
    mins = {}
    maxs = {}
    for name, coord, upper in (("il", il, shape[0]), ("xl", xl, shape[1]), ("k", kk, shape[2])):
        mn = np.full(n_bodies + 1, upper, dtype=np.int64)
        mx = np.full(n_bodies + 1, -1, dtype=np.int64)
        np.minimum.at(mn, ids, coord)
        np.maximum.at(mx, ids, coord)
        mins[name], maxs[name] = mn, mx
    t0, dt = float(geometry.get("t0_ms", 0.0)), float(geometry.get("dt_ms", 1.0))
    il_min, il_step = int(geometry.get("il_min", 0)), int(geometry.get("il_step", 1))
    xl_min, xl_step = int(geometry.get("xl_min", 0)), int(geometry.get("xl_step", 1))
    seed_counts = {int(r["geobody_id"]): int(r["seed_voxel_count"])
                   for r in components if r.get("geobody_id") is not None}
    column_counts = {int(r["geobody_id"]): int(r["distinct_columns"])
                     for r in components if r.get("geobody_id") is not None}
    rows = []
    for gid in range(1, n_bodies + 1):
        n = int(counts[gid])
        if not n:
            continue
        ci, cj, ck = sum_il[gid] / n, sum_xl[gid] / n, sum_k[gid] / n
        imin, imax = int(mins["il"][gid]), int(maxs["il"][gid])
        jmin, jmax = int(mins["xl"][gid]), int(maxs["xl"][gid])
        kmin, kmax = int(mins["k"][gid]), int(maxs["k"][gid])
        touches = (imin == 0 or imax == shape[0] - 1 or jmin == 0 or jmax == shape[1] - 1
                   or kmin == 0 or kmax == shape[2] - 1)
        rows.append({
            "geobody_id": gid, "voxel_count": n,
            "distinct_columns": column_counts.get(gid, 0),
            "mean_score": float(sums[gid] / n),
            "seed_voxel_count": seed_counts.get(gid, 0),
            "centroid_il_index": float(ci), "centroid_xl_index": float(cj), "centroid_time_sample": float(ck),
            "centroid_inline": float(il_min + ci * il_step),
            "centroid_crossline": float(xl_min + cj * xl_step), "centroid_twt_ms": float(t0 + ck * dt),
            "il_min": il_min + imin * il_step, "il_max": il_min + imax * il_step,
            "xl_min": xl_min + jmin * xl_step, "xl_max": xl_min + jmax * xl_step,
            "twt_min_ms": float(t0 + kmin * dt), "twt_max_ms": float(t0 + kmax * dt),
            "touches_il_min": imin == 0, "touches_il_max": imax == shape[0] - 1,
            "touches_xl_min": jmin == 0, "touches_xl_max": jmax == shape[1] - 1,
            "touches_time_top": kmin == 0, "touches_time_base": kmax == shape[2] - 1,
            "closure_status": "TRUNCATED_BY_EXPORT_BOUNDARY" if touches else "WITHIN_EXPORTED_WINDOW",
            "interpretation_status": "EXPLORATORY_UNCALIBRATED_CANDIDATE",
        })
    return rows



def _write_mesh(labels: np.ndarray, body_id: int, path: Path, geometry: dict, max_faces: int) -> int:
    """Write an exact exposed-voxel-face PLY; coordinates are IL, XL and -TWT."""
    rows = np.argwhere(labels == body_id)
    if not len(rows):
        return 0
    lo, hi = rows.min(axis=0), rows.max(axis=0) + 1
    sub = np.pad(labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == body_id, 1)
    n_i, n_j, n_k = sub.shape
    faces = []
    face_total = 0
    for axis in range(3):
        delta = np.diff(sub.astype(np.int8), axis=axis)
        for sign in (1, -1):
            idx = np.argwhere(delta == sign)
            if not len(idx):
                continue
            idx[:, axis] += 1
            other = [x for x in range(3) if x != axis]
            quads = []
            for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                corner = idx.copy()
                corner[:, other[0]] += da
                corner[:, other[1]] += db
                quads.append(corner)
            quad = np.stack(quads, axis=1)
            if sign < 0:
                quad = quad[:, ::-1]
            face_total += len(quad)
            if face_total > max_faces:
                return -face_total
            faces.append(quad)
    if not faces:
        return 0
    all_faces = np.concatenate(faces, axis=0)
    if len(all_faces) > max_faces:
        return -len(all_faces)
    flat = all_faces.reshape(-1, 3)
    key = (flat[:, 0] * (n_j + 1) + flat[:, 1]) * (n_k + 1) + flat[:, 2]
    unique, inverse = np.unique(key, return_inverse=True)
    verts = np.column_stack((unique // ((n_j + 1) * (n_k + 1)),
                             (unique // (n_k + 1)) % (n_j + 1), unique % (n_k + 1))).astype(float)
    verts += lo - 1 - 0.5
    il = float(geometry.get('il_min', 0)) + verts[:, 0] * float(geometry.get('il_step', 1))
    xl = float(geometry.get('xl_min', 0)) + verts[:, 1] * float(geometry.get('xl_step', 1))
    twt = float(geometry.get('t0_ms', 0)) + verts[:, 2] * float(geometry.get('dt_ms', 1))
    vertices = np.column_stack((il, xl, -twt)).astype('<f8')
    face_idx = inverse.reshape(-1, 4).astype('<i4')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as fh:
        header = ("ply\nformat binary_little_endian 1.0\n"
                  "comment GeoGG geobody voxel-boundary surface\n"
                  "comment x=inline_number y=crossline_number z=negative_TWT_ms; not map coordinates\n"
                  f"comment geobody_id={body_id}\nelement vertex {len(vertices)}\n"
                  "property double x\nproperty double y\nproperty double z\n"
                  f"element face {len(face_idx)}\nproperty list uchar int vertex_indices\nend_header\n")
        fh.write(header.encode('ascii'))
        fh.write(vertices.tobytes())
        records = np.empty(len(face_idx), dtype=[('n', 'u1'), ('v', '<i4', (4,))])
        records['n'], records['v'] = 4, face_idx
        fh.write(records.tobytes())
    return int(len(face_idx))


def _export_segy_products(stage_dir: Path, source_stage: Path, inference_stage: Path,
                          manifest: dict, labels: np.ndarray, valid: np.ndarray,
                          config: dict) -> list[dict]:
    """Write mask and ID SEG-Y on Stage-11 SEG-Y geometry.

    The trace-index grid is a canonical Stage-09 inference artifact. Stage 11
    references it in its geometry manifest but does not copy it into Stage 11.
    """
    from scripts.geogg_fresh.segy_write import NULL_VALUE, ReferenceSurvey, write_product, reopen_qc
    from scripts.geogg_fresh.segy import SegyFile

    segy_products = manifest.get('derived_segy_export', {}).get('products', [])
    source_row = next((x for x in segy_products if isinstance(x, dict)
                       and x.get('product') == str(config.get('source_product', 'prediction_lithology'))
                       and str(x.get('format', '')).lower() == 'segy'), None)
    if not source_row or str(source_row.get('status', 'PASS')).upper() != 'PASS':
        raise RuntimeError('Stage 11 has no QC-passed SEG-Y reference for the selected source product')
    ref_path = (source_stage / str(source_row.get('path', ''))).resolve()
    try:
        ref_path.relative_to(source_stage.resolve())
    except ValueError as exc:
        raise RuntimeError('Stage 11 SEG-Y reference path escapes its stage directory') from exc
    if not ref_path.is_file():
        raise RuntimeError(f'Stage 11 SEG-Y reference is missing: {ref_path}')
    expected_hash = str(source_row.get('sha256') or '').lower()
    if expected_hash and _sha256(ref_path) != expected_hash:
        raise RuntimeError('Stage 11 SEG-Y reference checksum mismatch')
    geometry = manifest.get('geometry') if isinstance(manifest.get('geometry'), dict) else {}
    declared_grid = str(geometry.get('trace_index_grid') or 'volumes/trace_index_grid.npy').strip()
    # Stage-11 geometry carries the relative path from the Stage-09 inference
    # directory (e.g. volumes/trace_index_grid.npy), not from Stage 11.
    trace_grid_path = (inference_stage / declared_grid).resolve()
    try:
        trace_grid_path.relative_to(inference_stage.resolve())
    except ValueError as exc:
        raise RuntimeError('Stage 11 trace_index_grid path escapes the inference stage') from exc
    if not trace_grid_path.is_file():
        raise RuntimeError(
            f'Canonical Stage 09 trace_index_grid.npy is missing: {trace_grid_path} '
            f'(Stage 11 declares {declared_grid!r})'
        )
    trace_grid = np.load(trace_grid_path, mmap_mode='r', allow_pickle=False)
    if tuple(trace_grid.shape) != tuple(labels.shape[:2]):
        raise RuntimeError(f'trace_index_grid geometry {trace_grid.shape} does not match labels {labels.shape[:2]}')
    live_flat = np.flatnonzero(np.asarray(trace_grid).reshape(-1) >= 0)
    with SegyFile(ref_path) as source_segy:
        if source_segy.n_traces != len(live_flat) or source_segy.ns != labels.shape[2]:
            raise RuntimeError(f'Stage 11 SEG-Y geometry mismatch: traces={source_segy.n_traces}, ns={source_segy.ns}, expected={len(live_flat)}x{labels.shape[2]}')
        expected_dt_us = int(round(float(manifest.get('geometry', {}).get('dt_ms', source_segy.dt_us / 1000.0)) * 1000.0))
        if source_segy.dt_us != expected_dt_us:
            raise RuntimeError(f'Stage 11 SEG-Y sample interval {source_segy.dt_us} us does not match registered geometry {expected_dt_us} us')
    # Stage-11 SEG-Y rows are already ordered by flattened live trace-grid cells.
    trace_idx_source = np.asarray(trace_grid, dtype=np.int64).reshape(-1)[live_flat]
    if np.unique(trace_idx_source).size != trace_idx_source.size:
        raise RuntimeError('Stage 11 trace_index_grid contains duplicate source trace mappings')
    trace_idx = np.arange(len(live_flat), dtype=np.int64)
    flat_labels = labels.reshape(-1, labels.shape[2])
    flat_valid = valid.reshape(-1, valid.shape[2])
    max_id = int(labels.max(initial=0))
    if max_id > 2**24:
        raise RuntimeError(f'{max_id} geobody IDs cannot be represented exactly in float32 SEG-Y')
    export_dir = stage_dir / 'segy'
    export_dir.mkdir(parents=True, exist_ok=True)
    products = []
    with _reference_context_for_geobody(ref_path) as ref:
        for name, kind in (('geobody_mask', 'mask'), ('geobody_id', 'id')):
            data_path = export_dir / f'.{name}.staging.npy'
            data = np.lib.format.open_memmap(data_path, mode='w+', dtype=np.float32,
                                             shape=(len(live_flat), labels.shape[2]))
            for start in range(0, len(live_flat), 512):
                stop = min(len(live_flat), start + 512)
                bins = live_flat[start:stop]
                lb = np.asarray(flat_labels[bins], dtype=np.int32)
                vb = np.asarray(flat_valid[bins], dtype=bool)
                if kind == 'mask':
                    block = (lb > 0).astype(np.float32)
                else:
                    block = lb.astype(np.float32)
                block[~vb] = np.float32(NULL_VALUE)
                data[start:stop] = block
            data.flush()
            out_path = export_dir / f'{name}.sgy'
            write_meta = write_product(ref, out_path, trace_idx, 0, data,
                provenance=[f'APGCE Stage 12 {name}', f'Source geobody job: {stage_dir.parent.parent.name}',
                            'Canonical labels: geobody_labels.npy', f'NULL={NULL_VALUE} invalid'])
            qc = reopen_qc(name, 'class_id' if kind == 'id' else 'bitmask', out_path, ref,
                           trace_idx, 0, data, [0, 1] if kind == 'mask' else list(range(max_id + 1)))
            with SegyFile(out_path) as check_file, SegyFile(ref_path) as ref_file:
                same = check_file.n_traces == ref_file.n_traces and check_file.ns == ref_file.ns
                if same:
                    for start in range(0, check_file.n_traces, 100000):
                        same = same and np.array_equal(np.asarray(check_file._mm[start:start+100000, :240]),
                                                        np.asarray(ref_file._mm[start:start+100000, :240]))
                        if not same:
                            break
            qc.append({'product': name, 'check': 'trace_headers_identical_to_stage11_reference',
                       'result': 'PASS' if same else 'FAIL', 'tolerance': 'byte-identical'})
            if not same or any(x.get('result') == 'FAIL' for x in qc):
                raise RuntimeError(f'{name} SEG-Y read-back QC failed: {qc}')
            sidecar = {'schema_version': 1, 'product': name, 'format': 'SEG-Y IEEE float32',
                       'canonical_source': 'geobody_labels.npy', 'reference_segy': str(ref_path),
                       'reference_sha256': expected_hash, 'trace_mapping': 'Stage-11 trace_index_grid flattened live bins; reference rows retained in same order',
                       'null_value': NULL_VALUE, 'label_codes': {'-1': 'invalid', '0': 'background', '>0': 'geobody ID'},
                       'write': write_meta, 'reopen_qc': qc, 'status': 'PASS'}
            sidecar_path = out_path.with_suffix('.manifest.json')
            _atomic_write_json(sidecar_path, sidecar)
            products.append({'product': name, 'format': 'segy', 'path': str(out_path.relative_to(stage_dir.parent.parent)),
                             'sha256': _sha256(out_path), 'bytes': out_path.stat().st_size,
                             'sidecar': str(sidecar_path.relative_to(stage_dir.parent.parent)), 'status': 'PASS',
                             'reopen_qc': qc})
            del data
            data_path.unlink(missing_ok=True)
    return products


def _reference_context_for_geobody(path: Path):
    """ReferenceSurvey is a lightweight reader wrapper; no external SEG-Y dependency."""
    from scripts.geogg_fresh.segy_write import ReferenceSurvey
    class _Context:
        def __enter__(self):
            self.value = ReferenceSurvey(path)
            return self.value
        def __exit__(self, exc_type, exc, tb):
            return False
    return _Context()


def _write_visualisations(stage_out: Path, labels: np.ndarray, catalog_rows: list[dict]) -> list[dict]:
    """Create compact overview and per-body orthogonal section PNGs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = stage_out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    assets = []
    # Overview uses deterministic voxel subsampling to bound render time and file size.
    body_voxels = np.flatnonzero(labels.ravel() > 0)
    if body_voxels.size > 250_000:
        body_voxels = body_voxels[np.linspace(0, body_voxels.size - 1, 250_000, dtype=np.int64)]
    if body_voxels.size:
        shape = labels.shape
        ii = body_voxels // (shape[1] * shape[2])
        jj = (body_voxels // shape[2]) % shape[1]
        kk = body_voxels % shape[2]
        gids = labels.ravel()[body_voxels]
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('tab20')
        for gid in sorted(set(gids.tolist())):
            sel = gids == gid
            ax.scatter(jj[sel], ii[sel], kk[sel], s=1, alpha=0.55, color=cmap((int(gid)-1) % 20), label=f'GB {gid}')
        ax.set_xlabel('Crossline bin'); ax.set_ylabel('Inline bin'); ax.set_zlabel('Time sample')
        ax.set_title('GeoGG geobody overview (deterministic point subsample)')
        if len(set(gids.tolist())) <= 20: ax.legend(loc='upper left', fontsize=7)
        fig.tight_layout(); path = figures / 'geobody_overview_3d.png'; fig.savefig(path, dpi=150); plt.close(fig)
        assets.append({'product': 'geobody_visualisation', 'name': path.name, 'path': str(path.relative_to(stage_out.parent.parent)), 'stage_relative_path': str(path.relative_to(stage_out)), 'kind': 'overview'})
    # Catalogue ranking plot.
    if catalog_rows:
        ordered = sorted(catalog_rows, key=lambda x: int(x['voxel_count']), reverse=True)[:30]
        fig, ax = plt.subplots(figsize=(10, max(4, len(ordered) * 0.28)))
        ax.barh([f"GB {r['geobody_id']}" for r in ordered[::-1]], [int(r['voxel_count']) for r in ordered[::-1]])
        ax.set_xlabel('Voxel count'); ax.set_title('Largest retained geobodies')
        fig.tight_layout(); path = figures / 'geobody_catalog.png'; fig.savefig(path, dpi=150); plt.close(fig)
        assets.append({'product': 'geobody_visualisation', 'name': path.name, 'path': str(path.relative_to(stage_out.parent.parent)), 'stage_relative_path': str(path.relative_to(stage_out)), 'kind': 'catalogue'})
    # Three orthogonal labelled slices through each of the largest 12 bodies.
    for row in sorted(catalog_rows, key=lambda x: int(x['voxel_count']), reverse=True)[:12]:
        gid = int(row['geobody_id'])
        ci = int(round(float(row['centroid_il_index'])))
        cj = int(round(float(row['centroid_xl_index'])))
        ck = int(round(float(row['centroid_time_sample'])))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(labels[ci, :, :].T == gid, origin='lower', aspect='auto', cmap='gray_r')
        axes[0].set_title(f'Inline {ci}')
        axes[1].imshow(labels[:, cj, :].T == gid, origin='lower', aspect='auto', cmap='gray_r')
        axes[1].set_title(f'Crossline {cj}')
        axes[2].imshow(labels[:, :, ck] == gid, origin='lower', aspect='auto', cmap='gray_r')
        axes[2].set_title(f'Time sample {ck}')
        for ax in axes: ax.set_xlabel('Grid index'); ax.set_ylabel('Grid index')
        fig.suptitle(f'Geobody {gid} - orthogonal footprint slices'); fig.tight_layout()
        path = figures / f'geobody_{gid:04d}_slices.png'; fig.savefig(path, dpi=130); plt.close(fig)
        assets.append({'product': 'geobody_visualisation', 'name': path.name, 'path': str(path.relative_to(stage_out.parent.parent)), 'stage_relative_path': str(path.relative_to(stage_out)), 'kind': 'body_slices', 'geobody_id': gid})
    return assets


def _load_trace_xy(inference_job: Path, volume_stage: Path, volume_manifest: dict,
                   labels_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, str]:
    """Map IL/XL bins to CDP X/Y using Stage 09 trace grid and Stage 11 SEG-Y."""
    nx, ny = labels_shape[:2]
    x_grid = np.full((nx, ny), np.nan, dtype=np.float64)
    y_grid = np.full((nx, ny), np.nan, dtype=np.float64)
    geometry = volume_manifest.get("geometry") if isinstance(volume_manifest.get("geometry"), dict) else {}
    rel_grid = str(geometry.get("trace_index_grid") or "volumes/trace_index_grid.npy")
    grid_path = (inference_job / "stages/09_inference" / rel_grid).resolve()
    try:
        grid_path.relative_to((inference_job / "stages/09_inference").resolve())
    except ValueError:
        return x_grid, y_grid, "UNAVAILABLE_UNSAFE_TRACE_GRID_PATH"
    if not grid_path.is_file():
        return x_grid, y_grid, "UNAVAILABLE_TRACE_INDEX_GRID"
    trace_grid = np.load(grid_path, mmap_mode="r", allow_pickle=False)
    if tuple(trace_grid.shape) != (nx, ny):
        return x_grid, y_grid, "UNAVAILABLE_TRACE_GRID_SHAPE_MISMATCH"
    segy_cfg = volume_manifest.get("derived_segy_export")
    segy_products = segy_cfg.get("products", []) if isinstance(segy_cfg, dict) else []
    segy_row = next((r for r in segy_products if isinstance(r, dict) and
                     str(r.get("format", "")).lower() == "segy" and
                     str(r.get("status", "PASS")).upper() == "PASS"), None)
    if not segy_row:
        return x_grid, y_grid, "UNAVAILABLE_NO_QC_PASSED_REFERENCE_SEGY"
    rel = str(segy_row.get("path") or "").strip()
    segy_path = (volume_stage / rel).resolve()
    try:
        segy_path.relative_to(volume_stage.resolve())
    except ValueError:
        return x_grid, y_grid, "UNAVAILABLE_UNSAFE_REFERENCE_SEGY_PATH"
    if not segy_path.is_file():
        return x_grid, y_grid, "UNAVAILABLE_REFERENCE_SEGY_MISSING"
    from geogg_fresh.segy import SegyFile
    live_flat = np.flatnonzero(np.asarray(trace_grid).reshape(-1) >= 0)
    with SegyFile(segy_path) as segy:
        if segy.n_traces != len(live_flat):
            return x_grid, y_grid, "UNAVAILABLE_REFERENCE_TRACE_COUNT_MISMATCH"
        headers = segy.scan_headers(fields=("cdp_x", "cdp_y", "coord_scalar"), show_progress=False)
        x = segy.apply_scalar(headers["cdp_x"], headers["coord_scalar"])
        y = segy.apply_scalar(headers["cdp_y"], headers["coord_scalar"])
    flat_x, flat_y = x_grid.reshape(-1), y_grid.reshape(-1)
    flat_x[live_flat], flat_y[live_flat] = x, y
    finite = np.isfinite(x) & np.isfinite(y)
    return x_grid, y_grid, ("AVAILABLE" if finite.any() else "UNAVAILABLE_NO_FINITE_CDP_COORDINATES")


def _write_geobody_run_length_index(labels: np.ndarray, geometry: dict, output_path: Path,
                                    x_grid: np.ndarray, y_grid: np.ndarray,
                                    slab_rows: int = 16) -> dict:
    """Write a lossless vertical-run index for every positive geobody label."""
    il_min, il_step = int(geometry["il_min"]), int(geometry["il_step"])
    xl_min, xl_step = int(geometry["xl_min"]), int(geometry["xl_step"])
    t0, dt = float(geometry["t0_ms"]), float(geometry["dt_ms"])
    fields = ["geobody_id", "volume_i", "volume_j", "il", "xl", "x", "y",
              "top_time_k", "base_time_k", "twt_top_ms", "twt_base_ms", "voxel_count"]
    voxel_counts: dict[int, int] = {}
    run_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i0 in range(0, labels.shape[0], slab_rows):
            block = np.asarray(labels[i0:i0 + slab_rows])
            active_i, active_j = np.nonzero(np.any(block > 0, axis=2))
            for local_i, j in zip(active_i.tolist(), active_j.tolist()):
                i = i0 + int(local_i)
                trace = block[local_i, j, :]
                ks = np.flatnonzero(trace > 0)
                if not len(ks):
                    continue
                gids = trace[ks].astype(np.int64, copy=False)
                cuts = np.flatnonzero((np.diff(ks) > 1) | (gids[1:] != gids[:-1])) + 1
                for idx in np.split(np.arange(len(ks)), cuts):
                    k0, k1, gid = int(ks[idx[0]]), int(ks[idx[-1]]), int(gids[idx[0]])
                    writer.writerow({"geobody_id": gid, "volume_i": i, "volume_j": int(j),
                        "il": il_min + i * il_step, "xl": xl_min + int(j) * xl_step,
                        "x": float(x_grid[i, j]) if np.isfinite(x_grid[i, j]) else None,
                        "y": float(y_grid[i, j]) if np.isfinite(y_grid[i, j]) else None,
                        "top_time_k": k0, "base_time_k": k1,
                        "twt_top_ms": t0 + k0 * dt, "twt_base_ms": t0 + k1 * dt,
                        "voxel_count": int(len(idx))})
                    voxel_counts[gid] = voxel_counts.get(gid, 0) + int(len(idx))
                    run_count += 1
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(output_path)
    return {"run_rows": run_count, "voxel_counts_by_geobody": voxel_counts,
            "total_exported_voxels": int(sum(voxel_counts.values())),
            "format": "lossless_vertical_run_length", "depth_conversion": "NOT_APPLICABLE_STAGE12_TWT_ONLY"}


def _write_geobody_voxel_xyz(labels: np.ndarray, geometry: dict, output_path: Path,
                             x_grid: np.ndarray, y_grid: np.ndarray,
                             chunk_voxels: int = 250_000) -> dict:
    """Optional chunked per-voxel spatial export; depth conversion is deliberately excluded."""
    if int(chunk_voxels) < 1:
        raise RuntimeError("voxel_export_chunk_size must be a positive integer")
    fields = ["geobody_id", "volume_i", "volume_j", "time_k", "il", "xl", "x", "y", "twt_ms"]
    il_min, il_step = int(geometry["il_min"]), int(geometry["il_step"])
    xl_min, xl_step = int(geometry["xl_min"]), int(geometry["xl_step"])
    t0, dt = float(geometry["t0_ms"]), float(geometry["dt_ms"])
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    total = 0
    with temp.open("w", encoding="utf-8", newline="") as stream:
        pd.DataFrame(columns=fields).to_csv(stream, index=False)
        for i in range(labels.shape[0]):
            slab = np.asarray(labels[i])
            jj, kk = np.nonzero(slab > 0)
            for start in range(0, len(jj), chunk_voxels):
                stop = min(len(jj), start + chunk_voxels)
                j_chunk, k_chunk = jj[start:stop].astype(np.int64), kk[start:stop].astype(np.int64)
                frame = pd.DataFrame({
                    "geobody_id": slab[j_chunk, k_chunk].astype(np.int64),
                    "volume_i": np.full(stop - start, i, dtype=np.int64),
                    "volume_j": j_chunk, "time_k": k_chunk,
                    "il": il_min + i * il_step, "xl": xl_min + j_chunk * xl_step,
                    "x": x_grid[i, j_chunk], "y": y_grid[i, j_chunk],
                    "twt_ms": t0 + k_chunk * dt,
                })
                frame.to_csv(stream, index=False, header=False)
                total += len(frame)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(output_path)
    return {"voxel_rows": total, "chunk_voxels": int(chunk_voxels),
            "depth_conversion": "NOT_APPLICABLE_STAGE12_TWT_ONLY"}


def main() -> None:
    result_path = Path(os.environ["APGCE_RESULT_PATH"]).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path(os.environ.get("APGCE_GEOGG_GEOBODY_CONFIG") or Path(__file__).resolve().parents[2] / "D_geobody.yaml").expanduser().resolve()
    config = _load_yaml(config_path).get("geobody", {})
    if not isinstance(config, dict):
        raise RuntimeError("geobody configuration must be a mapping")

    checks: list[dict] = []
    current = result_path.parent
    jobs_root = next((p for p in current.parents if p.name == "processing_jobs"), None)
    if jobs_root is None:
        raise RuntimeError(f"Cannot locate definitive processing_jobs directory from {result_path}")
    jobs_root = jobs_root.resolve()

    def resolve_job(config_key: str, label: str) -> tuple[str, Path | None]:
        job_id = str(config.get(config_key) or "").strip()
        safe_id = bool(job_id and Path(job_id).name == job_id and job_id not in {".", ".."})
        candidate = jobs_root / job_id if safe_id else None
        safe = bool(candidate and not candidate.is_symlink() and candidate.is_dir()
                     and candidate.resolve().parent == jobs_root)
        checks.append(_check(f"{label.lower()}_job", safe,
                             f"{label} processing job {'found' if safe else 'missing or unsafe'}: {job_id or '<not configured>'}"))
        return job_id, candidate.resolve() if safe else None

    foundation_id, foundation_job = resolve_job("foundation_job_id", "Foundation")
    inference_id, inference_job = resolve_job("inference_job_id", "Inference")
    volume_id, source_job = resolve_job("source_volume_job_id", "Volume-products")

    inference_geometry_path = inference_job / "stages/09_inference/volumes/volume_geometry.json" if inference_job else None
    try:
        inference_geometry = json.loads(inference_geometry_path.read_text(encoding="utf-8")) if inference_geometry_path and inference_geometry_path.is_file() else {}
    except (OSError, ValueError):
        inference_geometry = {}
    recorded_foundation = str(inference_geometry.get("foundation_job_id") or "").strip()
    checks.append(_check("inference_foundation_link", bool(recorded_foundation and recorded_foundation == foundation_id),
                         f"Inference geometry records foundation job {recorded_foundation or '<missing>'}; configured foundation job is {foundation_id}."))

    stage_id = str(config.get("source_stage_id", "11_volume_export")).strip()
    stage_dir = source_job / "stages" / stage_id if source_job else None
    status_path = stage_dir / "validation_summary.json" if stage_dir else None
    manifest_path = stage_dir / "native_volume_export_manifest.json" if stage_dir else None
    try:
        stage_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path and status_path.is_file() else {}
    except (OSError, ValueError):
        stage_status = {}
    checks.append(_check("stage11_status", str(stage_status.get("status", "")).lower() in {"completed", "complete", "conditional"},
                         f"Stage 11 status: {stage_status.get('status', 'UNAVAILABLE')}"))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path and manifest_path.is_file() else {}
    except (OSError, ValueError):
        manifest = {}
    products = manifest.get("products") if isinstance(manifest.get("products"), list) else []
    checks.append(_check("stage11_manifest", bool(products), f"Stage 11 manifest contains {len(products)} registered native products."))
    provenance = manifest.get("provenance") if isinstance(manifest.get("provenance"), dict) else {}
    recorded_inference = str(manifest.get("source_inference_job_id") or provenance.get("source_inference_job_id") or "").strip()
    checks.append(_check("volume_inference_link", bool(recorded_inference and recorded_inference == inference_id),
                         f"Stage 11 manifest records inference job {recorded_inference or '<missing>'}; configured inference job is {inference_id}."))

    source_product = str(config.get("source_product", "prediction_lithology")).strip()
    product_rows = [r for r in products if isinstance(r, dict) and str(r.get("format", "")).lower() == "npy"
                    and str(r.get("product", "")).strip() == source_product]
    valid_rows = [r for r in products if isinstance(r, dict) and str(r.get("format", "")).lower() == "npy"
                  and str(r.get("product", "")).strip() == "prediction_valid"]
    score_path = valid_path = None
    score_row = product_rows[0] if product_rows else None
    valid_row = valid_rows[0] if valid_rows else None
    try:
        if score_row:
            score_path = _safe_product_path(jobs_root, score_row)
        if valid_row:
            valid_path = _safe_product_path(jobs_root, valid_row)
    except RuntimeError as exc:
        checks.append(_check("canonical_npy_paths", False, str(exc)))
    checks.append(_check("canonical_npy", bool(score_path and score_path.is_file() and score_row and
                         (not score_row.get("sha256") or _sha256(score_path) == str(score_row["sha256"]).lower())),
                         f"Canonical {source_product} NPY exists and matches its registered checksum." if score_path and score_path.is_file() and score_row and
                         (not score_row.get("sha256") or _sha256(score_path) == str(score_row["sha256"]).lower()) else
                         f"Canonical NPY for {source_product} is missing or checksum-mismatched."))
    checks.append(_check("validity_mask", bool(valid_path and valid_path.is_file() and valid_row and
                         (not valid_row.get("sha256") or _sha256(valid_path) == str(valid_row["sha256"]).lower())),
                         "Registered prediction_valid NPY exists and matches its checksum." if valid_path and valid_path.is_file() and valid_row and
                         (not valid_row.get("sha256") or _sha256(valid_path) == str(valid_row["sha256"]).lower()) else
                         "A registered, checksummed prediction_valid NPY is required."))

    geometry = manifest.get("geometry") if isinstance(manifest.get("geometry"), dict) else {}
    shape = geometry.get("shape")
    axis_order = geometry.get("axis_order")
    geometry_ok = isinstance(shape, list) and len(shape) == 3 and axis_order == ["inline_bin", "crossline_bin", "time_sample"]
    checks.append(_check("volume_geometry", geometry_ok, f"Stage 11 geometry shape={shape}; axis_order={axis_order}."))
    score_semantics = str(config.get("score_semantics", "unqualified_model_output")).strip().lower()
    semantics_ok = score_semantics in {"calibrated_probability", "validated_continuous_score"}
    checks.append(_check("source_product_semantics", semantics_ok,
                         "Selected product has declared validated continuous-score semantics." if semantics_ok else
                         f"{source_product} is an unqualified model output; segmentation is exploratory, not a calibrated probability map.",
                         required=False))
    threshold_evidence = bool(foundation_job and any((foundation_job / str(x)).is_file()
                              for x in config.get("threshold_evidence_paths", ["stages/07_models/tables/outer_test_predictions.csv"])))
    checks.append(_check("threshold_evidence", threshold_evidence,
                         "Configured development-fold threshold evidence is available." if threshold_evidence else
                         "No configured development-fold threshold evidence found; configured thresholds are exploratory.",
                         required=bool(config.get("require_threshold_evidence", False))))
    depth_ok = str(config.get("depth_conversion", "unavailable")).lower() in {"available", "configured", "true"}
    checks.append(_check("depth_conversion", depth_ok,
                         "Qualified depth conversion configured." if depth_ok else "No qualified time-depth conversion; report TWT/sample units only.",
                         required=False))

    blocked = [c for c in checks if c["status"] == "BLOCKED"]
    conditional = [c for c in checks if c["status"] == "CONDITIONAL"]
    preflight_status = "BLOCKED" if blocked else ("CONDITIONAL" if conditional else "READY")
    preflight = {
        "schema_version": 2, "check": "12_geobody", "status": preflight_status,
        "foundation_job_id": foundation_id or None, "inference_job_id": inference_id or None,
        "source_volume_job_id": volume_id or None, "source_stage_id": stage_id,
        "source_product": source_product, "canonical_format": "NPY",
        "checks": checks, "notes": [
            "A conditional preflight permits extraction but labels all bodies EXPLORATORY_UNCALIBRATED_CANDIDATE.",
            "Candidate bodies are connected components of a seeded low/high threshold mask, not interpreted reservoirs.",
            "No depth, fluid, net-pay, reserves, or volumetric claims are made."
        ]
    }
    (result_path.parent / OUTPUT).write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if blocked:
        raise RuntimeError("Stage 12 geobody preflight BLOCKED:\n" +
                           "\n".join(f"- {c['check']}: {c['detail']}" for c in blocked))

    # ------------------------------ first functional 12A/12B extraction
    if not score_path or not valid_path or not geometry_ok:
        raise RuntimeError("Stage 12 extraction cannot start: canonical score, validity mask, or geometry is unavailable.")
    score = np.load(score_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(int(x) for x in shape)
    if tuple(score.shape) != expected_shape or tuple(valid.shape) != expected_shape:
        raise RuntimeError(f"Source array geometry mismatch: score={score.shape}, valid={valid.shape}, manifest={expected_shape}")
    if score.dtype.kind not in "fi" or valid.dtype.kind not in "biu":
        raise RuntimeError(f"Unsupported source dtypes: score={score.dtype}, valid={valid.dtype}")
    if valid.size and not np.isin(np.unique(valid), [0, 1]).all():
        raise RuntimeError("prediction_valid must be binary (0/1)")
    valid_bool = np.asarray(valid, dtype=bool)
    sample = np.asarray(score[valid_bool])
    if sample.size and (not np.isfinite(sample).all() or float(sample.min()) < 0 or float(sample.max()) > 1):
        raise RuntimeError("Selected scalar score must be finite and within [0, 1] on valid voxels.")

    tau_low = float(config.get("tau_low", 0.35))
    tau_high = float(config.get("tau_high", 0.65))
    min_voxels = int(config.get("min_voxels", 18))
    min_columns = int(config.get("min_distinct_columns", 9))
    min_extent = int(config.get("min_lateral_extent_cells", 3))
    connectivity = int(config.get("connectivity", 6))
    labels, candidate, component_rows = _extract(score, valid_bool, tau_low, tau_high,
                                                   min_voxels, min_columns, min_extent, connectivity)
    catalog_rows = _catalogue(labels, score, valid_bool, component_rows, geometry)

    stage_out = current / "stages" / "12_geobody"
    volumes_dir = stage_out / "volumes"
    tables_dir = stage_out / "tables"
    labels_path = volumes_dir / "geobody_labels.npy"
    mask_path = volumes_dir / "geobody_candidate_mask.npy"
    _atomic_save_npy(labels_path, labels)
    _atomic_save_npy(mask_path, candidate)

    import pandas as pd
    components_path = tables_dir / "geobody_component_audit.csv"
    catalog_path = tables_dir / "geobody_catalog.csv"
    components_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(component_rows).to_csv(components_path, index=False)
    pd.DataFrame(catalog_rows).to_csv(catalog_path, index=False)

    # Specialist-aligned deliverables requested: voxel-boundary meshes, native-geometry
    # SEG-Y mask/ID exports, static visualisations, and a read-only API over these assets.
    mesh_dir = stage_out / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_products = []
    mesh_max_faces = int(config.get("mesh_max_faces", 400000))
    mesh_max_voxels = int(config.get("mesh_max_voxels", 5_000_000))
    mesh_limit = int(config.get("mesh_max_bodies", 20))
    for row in sorted(catalog_rows, key=lambda x: int(x["voxel_count"]), reverse=True)[:mesh_limit]:
        gid = int(row["geobody_id"])
        mesh_path = mesh_dir / f"geobody_{gid:04d}.ply"
        if int(row["voxel_count"]) > mesh_max_voxels:
            mesh_products.append({"product": "geobody_mesh", "geobody_id": gid, "status": "SKIPPED", "reason": "MESH_VOXEL_LIMIT",
                                  "voxel_count": int(row["voxel_count"])})
            continue
        face_count = _write_mesh(labels, gid, mesh_path, geometry, mesh_max_faces)
        if face_count <= 0:
            mesh_products.append({"product": "geobody_mesh", "geobody_id": gid, "status": "SKIPPED", "reason": "MESH_FACE_LIMIT_OR_EMPTY",
                                  "face_count": abs(face_count)})
            mesh_path.unlink(missing_ok=True)
            continue
        mesh_products.append({"product": "geobody_mesh", "geobody_id": gid, "status": "PASS", "format": "PLY",
                              "path": str(mesh_path.relative_to(current)), "stage_relative_path": str(mesh_path.relative_to(stage_out)), "vertices_faces": face_count,
                              "sha256": _sha256(mesh_path), "bytes": mesh_path.stat().st_size,
                              "coordinate_system": "inline_number,crossline_number,-TWT_ms"})
    inference_stage = inference_job / "stages" / "09_inference" if inference_job else None
    if inference_stage is None or not inference_stage.is_dir():
        raise RuntimeError("Stage 09 inference directory is required for trace-grid-based SEG-Y export")
    segy_products = _export_segy_products(
        stage_out, stage_dir, inference_stage, manifest, labels, valid_bool, config
    )

    # Stage 12 owns complete, well-independent spatial exports for every retained body.
    x_grid, y_grid, xy_status = _load_trace_xy(inference_job, stage_dir, manifest, expected_shape)
    run_length_path = tables_dir / "geobody_run_length_index.csv"
    run_export = _write_geobody_run_length_index(labels, geometry, run_length_path, x_grid, y_grid)
    voxel_path = tables_dir / "geobody_voxel_xyz.csv"
    voxel_export = None
    if bool(config.get("export_voxel_xyz", False)):
        voxel_export = _write_geobody_voxel_xyz(
            labels, geometry, voxel_path, x_grid, y_grid,
            int(config.get("voxel_export_chunk_size", 250_000)))
    else:
        voxel_path.unlink(missing_ok=True)

    # Independently recount labels in bounded slabs; avoid allocating a
    # full-volume boolean selection for large seismic cubes.
    label_counts: dict[int, int] = {}
    for i0 in range(0, labels.shape[0], 16):
        block_ids, block_counts = np.unique(np.asarray(labels[i0:i0 + 16]), return_counts=True)
        for gid, count in zip(block_ids.tolist(), block_counts.tolist()):
            if int(gid) > 0:
                label_counts[int(gid)] = label_counts.get(int(gid), 0) + int(count)
    catalogue_counts = {int(row["geobody_id"]): int(row["voxel_count"])
                        for row in catalog_rows if int(row.get("geobody_id", 0)) > 0}
    all_body_ids = sorted(set(label_counts) | set(catalogue_counts))
    reconciliation_rows = [{
        "geobody_id": gid,
        "label_volume_voxel_count": label_counts.get(gid, 0),
        "catalogue_voxel_count": catalogue_counts.get(gid, 0),
        "run_length_export_voxel_count": int(run_export["voxel_counts_by_geobody"].get(gid, 0)),
        "label_minus_catalogue": label_counts.get(gid, 0) - catalogue_counts.get(gid, 0),
        "label_minus_run_length": label_counts.get(gid, 0) -
                                  int(run_export["voxel_counts_by_geobody"].get(gid, 0)),
        "reconciles": (label_counts.get(gid, 0) == catalogue_counts.get(gid, 0) ==
                       int(run_export["voxel_counts_by_geobody"].get(gid, 0))),
    } for gid in all_body_ids]
    reconciliation_path = tables_dir / "geobody_spatial_export_reconciliation.csv"
    import pandas as pd
    pd.DataFrame(reconciliation_rows).to_csv(reconciliation_path, index=False)
    reconciliation_ok = all(row["reconciles"] for row in reconciliation_rows)

    spatial_products = [
        {"product": "geobody_run_length_index", "format": "csv",
         "path": str(run_length_path.relative_to(current)), "sha256": _sha256(run_length_path)},
        {"product": "geobody_spatial_export_reconciliation", "format": "csv",
         "path": str(reconciliation_path.relative_to(current)), "sha256": _sha256(reconciliation_path)},
    ]
    if voxel_export is not None:
        spatial_products.append({"product": "geobody_voxel_xyz", "format": "csv",
            "path": str(voxel_path.relative_to(current)), "sha256": _sha256(voxel_path)})
    figure_products = _write_visualisations(stage_out, labels, catalog_rows)

    n_valid = int(valid_bool.sum())
    n_candidate = int(candidate.sum())
    n_retained = sum(1 for r in component_rows if r["decision"] == "RETAINED")
    n_rejected = len(component_rows) - n_retained
    summary = {
        "schema_version": 1, "stage": "12_geobody", "status": "completed",
        "interpretation_status": "EXPLORATORY_UNCALIBRATED_CANDIDATE",
        "source": {"foundation_job_id": foundation_id, "inference_job_id": inference_id,
                   "volume_job_id": volume_id, "stage_id": stage_id, "product": source_product,
                   "score_sha256": _sha256(score_path), "validity_sha256": _sha256(valid_path)},
        "geometry": {"shape": list(expected_shape), "axis_order": axis_order,
                     "il_min": geometry.get("il_min"), "il_step": geometry.get("il_step"),
                     "xl_min": geometry.get("xl_min"), "xl_step": geometry.get("xl_step"),
                     "t0_ms": geometry.get("t0_ms"), "dt_ms": geometry.get("dt_ms")},
        "method": {"name": "seeded_3d_hysteresis", "connectivity": connectivity,
                   "tau_low": tau_low, "tau_high": tau_high,
                   "min_voxels": min_voxels, "min_distinct_columns": min_columns,
                   "min_lateral_extent_cells": min_extent,
                   "threshold_status": "EXPLORATORY" if not threshold_evidence else "DEVELOPMENT_EVIDENCE_AVAILABLE"},
        "counts": {"valid_voxels": n_valid, "candidate_voxels": n_candidate,
                   "candidate_fraction_of_valid": float(n_candidate / n_valid) if n_valid else 0.0,
                   "raw_low_threshold_components": len(component_rows),
                   "retained_geobodies": n_retained, "rejected_components": n_rejected},
        "products": [
            {"product": "geobody_labels", "format": "npy", "path": str(labels_path.relative_to(current)),
             "dtype": str(labels.dtype), "shape": list(labels.shape), "sha256": _sha256(labels_path)},
            {"product": "geobody_candidate_mask", "format": "npy", "path": str(mask_path.relative_to(current)),
             "dtype": str(candidate.dtype), "shape": list(candidate.shape), "sha256": _sha256(mask_path)},
            {"product": "geobody_catalog", "format": "csv", "path": str(catalog_path.relative_to(current)),
             "sha256": _sha256(catalog_path)},
            {"product": "geobody_component_audit", "format": "csv", "path": str(components_path.relative_to(current)),
             "sha256": _sha256(components_path)}
        ] + spatial_products + segy_products + figure_products + mesh_products,
        "exports": {"spatial": {
                        "run_length_index": {"path": str(run_length_path.relative_to(current)),
                            **{k: v for k, v in run_export.items() if k != "voxel_counts_by_geobody"}},
                        "xy_coordinate_status": xy_status,
                        "xy_coordinate_source": "Stage 11 QC-passed SEG-Y CDP X/Y headers mapped through Stage 09 trace_index_grid",
                        "catalogue_reconciliation": {"path": str(reconciliation_path.relative_to(current)),
                            "geobodies_compared": len(reconciliation_rows),
                            "all_counts_reconcile": reconciliation_ok},
                        "optional_voxel_xyz": ({"path": str(voxel_path.relative_to(current)), **voxel_export}
                                               if voxel_export is not None else {"enabled": False}),
                        "depth_conversion": "TWT_ONLY; well-specific TVDSS conversion remains in Stage 13"},
                    "segy": segy_products, "meshes": mesh_products, "visualisations": figure_products},
        "api": {"base_path": f"/api/v1/geobodies/jobs/{current.name}",
                "endpoints": ["/", "/items", "/items/{geobody_id}", "/items/{geobody_id}/slice",
                              "/items/{geobody_id}/mesh", "/exports/geobody_mask", "/exports/geobody_id",
                              "/figures/{name}"]},
        "limitations": [
            "prediction_lithology score semantics are unqualified; all segmentation is exploratory.",
            "Bodies touching any exported volume face are truncated by the export boundary.",
            "Spatial exports provide IL/XL, CDP X/Y where available, and TWT; TVDSS conversion remains well/model-specific in Stage 13.",
            "Spatial export reconciliation must be checked; a false reconciliation flag indicates an export or catalogue consistency issue.",
            "Connected components are geometric candidates, not proof of reservoir quality, hydrocarbons, fluid, net pay, or reserves."
        ]
    }
    _atomic_write_json(stage_out / "geobody_manifest.json", summary)
    _atomic_write_json(result_path.parent / SUMMARY, summary)
    # Keep the preflight report and add the final extraction status for quick diagnostics.
    preflight["extraction"] = {"status": "COMPLETED", "retained_geobodies": n_retained,
                               "rejected_components": n_rejected, "candidate_voxels": n_candidate}
    _atomic_write_json(result_path.parent / OUTPUT, preflight)

