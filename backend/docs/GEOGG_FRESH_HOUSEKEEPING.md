# GeoGG Fresh Codebase Housekeeping and Specialist Parity Audit

Baseline reviewed: `backend_v103.zip`  
Specialist reference: `AskGeo_Development_v1.zip`  
Scope: `scripts/geogg_fresh/`

## Housekeeping changes

- Removed verified unused imports from the migrated stage modules.
- Replaced duplicated file SHA-256 implementations with the shared streaming helper in `provenance.sha256_file`.
- Made `SpecialistStageContext` logging, check, metric, README and output-writing methods explicit for easier debugging.
- Routed compatibility-layer table, text, JSON and figure outputs through stage-contained path resolvers, rejecting path traversal.
- Refactored Stage 08 evaluation into small selection and group-evaluation helpers without changing its sealed-prediction source or metric definitions.
- Added focused Stage 08 and output-path safety regression tests.
- Kept stage algorithms, scientific thresholds, feature definitions, model architecture, patch contract, inference precision and output schemas unchanged outside the documented Stage 08 readability refactor.

## Specialist functionality comparison

### Stages 00–07

The current backend contains the corresponding specialist stage modules. An AST-level top-level function/class inventory found no specialist top-level functions or classes missing from the current versions of `s00_inventory.py` through `s07_models.py`. The backend versions add APGCE-specific inventory resolution and well-resolution helpers.

The stages continue to use the specialist stage implementations through `SpecialistStageContext`; the backend context owns job identity, paths, inputs, outputs and provenance. This is an intentional execution-adapter change, not a second pipeline/job manager.

The following scientific contracts remain important and are not changed by this housekeeping:

- Stage 03 patch layout remains `[channels, 5, 5, 21]` at 4 ms sampling.
- Stage 04 feature construction and Stage 05 training-table schema remain the source of model features.
- Stage 06/07 feature-selection and model-training implementations remain in place.
- Model releases remain immutable and carry their source-job provenance and patch/feature contracts.

### Stage 08 — Evaluation

**Parity gap identified.** The specialist Stage 08 produces a broader evaluation package, including per-well and per-class metrics, paired comparisons, beyond-depth analysis, conformal intervals, calibration bins, interval-error tables and multiple evaluation figures.

The current backend Stage 08 evaluates sealed outer-test predictions and writes `heldout_metrics.csv` plus `evaluation_summary.json`. It does not currently reproduce the full specialist reporting package. The housekeeping refactor preserves this backend scope; it does not silently claim full specialist parity.

### Stage 09 — Inference

The current backend Stage 09 supports:

- released tabular-model inference;
- deep-model patch replay from the immutable source Stage 03 artifacts, with checksum verification;
- survey-volume inference from the five registered amplitude stacks;
- native NPY prediction volumes, geometry metadata, checkpoint/resume and Stage 03 patch-parity verification.

The specialist Stage 09 additionally produces well-level predictions, section outputs, reserved-well checks, section parity tables and associated figures. Those specialist deliverables are not all reproduced by the current backend Stage 09. Survey-volume inference is a backend extension that draws on specialist full-survey inference concepts.

SEG-Y volume writing/export remains deliberately excluded, per project direction.

### Later specialist stages

The specialist reference also contains Stage 10 summary, Stage 11 volume export, Stage 12 geobody and Stage 13 prospect workflows. They are not part of the current backend `geogg_fresh` stage runner. The project roadmap consolidates the next work into Stage G (summaries and native volume products), Stage H (geobody extraction), and Stage I (prospect volumetrics). Stage J is AskGeo domain intelligence.

## Explicitly retained backend adaptations

- APGCE processing-job UUID and backend-managed `processing_jobs/<UUID>/` layout are authoritative.
- Model releases are stored under backend-managed `model_releases/` and resolved by release provenance.
- Inference resolves original seismic stack paths from the foundation Stage 00 data manifest and verifies SHA-256 checksums.
- No Drive B or Windows filesystem paths are introduced.
- Stage F uses native NPY outputs. SEG-Y writing/export remains on hold.

## Verification

The focused GeoGG regression suite was run after housekeeping:

`PYTHONPATH=scripts python -m pytest -q tests/test_geogg_stage_f.py tests/test_geogg_fresh_foundation.py tests/test_geogg_stage_b.py tests/test_geogg_stage_c.py tests/test_geogg_volume_inference.py tests/test_geogg_model_release.py tests/test_geogg_runner_cli.py`

Result: **70 passed**.

This is a focused regression run, not a claim that every optional dependency or every end-to-end specialist workflow has been executed. The parity inventory is a source-level comparison; scientific equivalence still depends on the existing stage tests and real-data validation.
