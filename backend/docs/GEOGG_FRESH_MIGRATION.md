# GeoGG Fresh and AskGeo migration

## Migration boundary

The specialist `AskGeo_Development_v1` archive is the scientific/domain reference.
The definitive APGCE backend remains the application runtime and source of truth
for project resources, datasets, storage, processing jobs, execution and evidence.

| Concern | Owner after migration |
|---|---|
| Project, survey, well and dataset records | `app/` and existing database |
| Dataset path resolution and `inputs.json` | existing dataset/storage services |
| Processing job lifecycle and local/SLURM execution | existing processing-job service and runners |
| GeoChat intent and assistant interaction | `app/geochat/` |
| GeoGG scientific algorithms and stages | `scripts/geogg_fresh/` |
| User-facing output catalogue and provenance | existing Evidence Ledger, extended with GeoGG provenance |
| Stage-level scientific lineage | `scripts/geogg_fresh/provenance.py` |

The migration must not introduce a second project/dataset registry, scheduler,
job lifecycle, storage layer or competing evidence catalogue.

## Stage A — compatibility foundation

Implemented in Backend v59 and retained in this revision:

- Scientific package boundary under `scripts/geogg_fresh/`.
- Lightweight contracts for stage identity, status, inputs, outputs and results.
- `StageContext` adapter for the existing `APGCE_RESULT_PATH` and
  `APGCE_INPUT_MANIFEST` runner environment.
- Per-job run layout with isolated stage output directories.
- Configuration normalization and deterministic SHA-256 fingerprinting.
- Atomic, non-overwriting stage provenance records and output checksums.
- Separate scientific dependency manifest; no heavy scientific imports at
  package import time.

The specialist has two configuration levels which must not be conflated:
`askgeo.yaml` is its launcher/workspace configuration (`data_root`), while the
pipeline configuration (for example `configs/pipeline_template.yaml`) contains
scientific settings such as `paths.project_root`, `project.output_root` and
stage parameters. The adapter in this revision handles the pipeline structure.
The launcher-level `data_root` mapping will be finalized when Stage 00/01
input requirements are migrated.

## Stage B — input inventory and well foundations

Implemented in Backend v62:

- Migrated the specialist Stage 00 inventory implementation and its supporting
  LAS, deviation, checkshot, SEG-Y, plotting and progress utilities.
- Migrated Stage 01 trajectory, T-Z model selection, leave-one-well-out checks,
  interval-velocity QC and synthetic-to-seismic tie implementation.
- Added a thin `process_A_foundations.py` entry point that executes Stage 00
  and Stage 01 sequentially in one existing processing job. Stage 01 consumes
  Stage 00 outputs from that same job; running them as separate jobs is not the
  supported workflow.
- Added a compatibility context that maps specialist stage output calls onto
  the existing APGCE job directory and writes the existing `result.json` plus
  immutable stage provenance. It does not add a second job/checkpoint manager.
- Added `A_foundations.yaml.example`. Copy it to `site_data/scripts/A_foundations.yaml`
  and edit the paths, expected well list and scientific policies. The runner
  also accepts `APGCE_GEOGG_CONFIG` or `--config`.
- The input manifest is retained in provenance. Stage 00's scientific file
  discovery remains controlled by the editable GeoGG YAML paths and patterns;
  it does not silently guess a mapping from dataset names to scientific roles.

Stage 00/01 are implemented, but acceptance against the project's real data
must still be performed in the target scientific runner environment. Missing
or invalid inputs are reported as CONDITIONAL/BLOCKED rather than fabricated.

## Input and output contract

- Inputs are selected by dataset UUID in the existing application API.
- The existing job service resolves each registered dataset and writes
  `inputs.json`; GeoGG reads those resolved paths without database access.
- No source data are copied into the GeoGG run directory.
- Stage outputs are written only below
  `<processing-job>/stages/<stage-id>/`.
- Stage provenance records configuration fingerprint, input lineage, output
  checksums, status, checks, warnings, limitations and blockers.
- A new run ID is required for reruns; existing stage output and provenance are
  not overwritten.
- The existing runner's `result.json` remains authoritative for processing-job
  status. GeoGG stage status is subordinate stage-level scientific metadata.

## Dependency isolation

The API environment remains lean. Install `scripts/geogg_fresh/requirements-science.txt`
in the dedicated processing environment. CUDA/PyTorch must be installed as a
node-compatible build separately. Future migrated modules must avoid importing
optional scientific dependencies from package `__init__.py`.

## Deferred to later stages

- Stages 00–01 are migrated; stages 02–13 remain pending.
- No automatic translation of all specialist config sections is claimed.
- No new scheduler, checkpoint engine or application database model is added.
- No GeoChat specialist skills, ScenarioLab or EvidenceHub functionality is
  integrated in Stage A.


### Stage 00 inventory authority correction

The APGCE backend database is the sole authoritative dataset registry and UID
issuer. Processing-job `inputs.json` now carries each selected dataset UID, kind,
format, path, metadata, and registered well/survey association. Stage 00 consumes
only these manifest entries; directory discovery and independent dataset identity
creation are disabled. Its scientific inventory tables retain the backend UIDs.
Explicit scientific roles (for example seismic volume names or a well-header
workbook) are resolved through registered dataset UIDs using `dataset_roles` or
registered dataset metadata. Stage 01 consumes Stage 00's outputs and does not
create another inventory.


## Stage C — labels and seismic extraction

Implemented in Backend v77:

- Stage 02 migrates the specialist petrophysical label policy, source verification,
  curve QC, lithology/fluid evidence, confidence grades, task support and audit tables.
- Stage 03 migrates time-bin aggregation, well-position alignment, native-grid
  seismic patch extraction, channel/geometry checks and extraction audit outputs.
- Stage 03 resolves seismic source paths exclusively from the Stage 00 inventory,
  which carries the selected backend dataset UUID and resolved manifest path. It does
  not use legacy `paths.seismic` filesystem configuration.
- Stages 00–03 execute sequentially in one APGCE processing job so upstream outputs
  are available under the same flattened job directory.
- Missing curves, provisional ties, unsupported labels, absent volumes and invalid
  geometry remain explicit limitations/blockers; no labels or seismic values are fabricated.

Stage C is code-migrated in v77. Scientific acceptance still requires a real-data run
that produces `02_labels` and `03_extraction` provenance and passes their gate checks.
