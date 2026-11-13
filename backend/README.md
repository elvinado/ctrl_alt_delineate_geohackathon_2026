# APGCE Geohackathon Backend v8 — Configurable Processing Scripts

FastAPI + SQLAlchemy backend for project, survey, well, and geoscience dataset management. This revision intentionally makes a **breaking schema/API change** to give datasets a real polymorphic model and enforce resource relationships.

## Data model

```text
Project
 ├── Survey
 │    └── SeismicDataset (survey_id lives here)
 ├── Well
 │    ├── WellLogDataset (well_id lives here)
 │    │    └── Curve (stable curve ID)
 │    ├── CheckshotDataset (well_id lives here)
 │    └── DeviationDataset (well_id lives here)
 └── Dataset (base table: project_id + shared file metadata)
      ├── SeismicDataset
      ├── WellLogDataset
      ├── CheckshotDataset
      └── DeviationDataset

AppSetting
ProcessingJob
 └── ProcessingJobInput ── Dataset (many-to-many input references)
```

`Dataset` uses SQLAlchemy joined-table inheritance. Shared metadata and source-file identity are in `datasets`; each supported kind has a subtype table. SQLAlchemy loads the right subtype when a dataset is queried by ID. `kind` is the discriminator and cannot be changed through PATCH.

Every dataset stores its actual `storage_root`, relative `storage_key`, original filename, normalized format, byte size, and SHA-256 checksum. Uploaded files are stored under the active configured storage root at `uploads/<dataset-id>.<extension>`. The storage root is not inside the backend runtime folder. File access is always resolved from the dataset record using `resolve_dataset_path()`; do not reconstruct storage paths from user-supplied filenames.

### Curve design decision

A `Curve` is a **first-class addressable resource owned by a `WellLogDataset`**. It has its own UUID and can be retrieved directly by ID. Its foreign key targets `well_log_datasets.id`, not the generic `datasets` table, so a curve cannot be attached to seismic/checkshot/deviation data at the database level. Numerical samples remain in the original LAS file; the database stores curve definitions and column positions. If a source LAS file is ever replaced, its curve catalog must be regenerated.

## Relationships and integrity rules

- Every survey, well, and dataset belongs to exactly one project.
- A seismic dataset requires a survey from the same project.
- A well-log, deviation, or checkshot dataset requires a well from the same project.
- `datasets` contains `project_id` and common file/resource fields only; it has no `survey_id` or `well_id` columns.
- `seismic_datasets` owns the required `survey_id` foreign key.
- `well_log_datasets`, `checkshot_datasets`, and `deviation_datasets` each own their required `well_id` foreign key.
- A well-associated dataset cannot have `survey_id`. Survey-level activities and grouping remain at Project/Survey level; they are not added to well dataset records.
- Survey and well associations must belong to the same project as their dataset.
- A survey or well cannot be deleted while datasets still depend on it. Delete those datasets first.
- Deleting a project deletes its child resources and stored dataset files.
- Deleting a dataset deletes its subtype row, curves, and source file.
- SQLite foreign-key enforcement is enabled by the connection setup.

## Breaking-change warning

This version changes the database schema and runtime settings contract. `Base.metadata.create_all()` creates missing tables but does not migrate existing tables. **Back up and remove/rename the old SQLite database before running this revision** unless you implement a deliberate migration. Existing uploaded files may remain in the data directory; do not delete them until you have confirmed the backup is complete.

## Install and run

Python 3.12+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Configure environment variables directly if needed.
./run.sh
```

Alternatively:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8888
```

Open `/docs` for the interactive API. Application settings are available at `/api/v1/settings`. Storage paths, ingestion limits, processing defaults, and GeoChat parameters are editable through the Settings API and persisted in the `AppSetting` table.

By default, dataset files are stored in `backend/site_data`; the fallback defaults to `backend/site_data_fallback`. Both paths can be changed in Settings to point to suitable local or mounted storage. At startup and before uploads, the backend checks actual write access to the primary root and selects the fallback only if the primary is unavailable. Existing dataset files can be resolved from either configured root.

The SQLite database is independent of dataset storage and defaults to `backend/data/geohackathon.db`. Its location is bootstrap configuration: set `DATABASE_URL` in `backend/.env` if you need a different database. This cannot be safely stored as an ordinary editable setting because the database location must be known before the settings table can be read.

Example settings update:

```http
PATCH /api/v1/settings
Content-Type: application/json

{
  "storage": {
    "data_root": "/path/to/site_data",
    "fallback_data_root": "/data/epic20/APGCE/apgce_team_10/site_data"
  },
  "expected_version": 1
}
```

Storage paths must be absolute. Each dataset records the actual storage root used at upload time, in addition to its relative `storage_key`. Changing the configured roots affects future uploads; existing datasets continue to resolve from their recorded root, with both configured roots checked as fallback locations.

Useful environment settings:

- `DATABASE_URL` — SQLAlchemy database URL.
- `API_PREFIX` — defaults to `/api/v1`.
- `CORS_ORIGINS` — comma-separated frontend origins.

## Processing jobs and input manifest

`POST /api/v1/processing-jobs` accepts zero or more existing dataset UUIDs. If at least one dataset is supplied, the backend validates each dataset, resolves its source file, records the job-to-dataset relationship, and writes an `inputs.json` manifest in the job directory. The local and SLURM runners pass the manifest to the selected Python script using `--input-manifest`. If no datasets are supplied, no manifest is created and the script runs without that argument. The worker does not need direct database access.

Example request:

```json
{
  "python_script": "process_A_foundations.py",
  "shell_script": "pyrunner.local.sh",
  "dataset_ids": ["<dataset-uuid-1>", "<dataset-uuid-2>"]
}
```

The generated manifest contains an `inputs` array. Each entry includes `dataset_id`, `kind`, `format`, `name`, and the resolved `path`. An empty or omitted `dataset_ids` list is allowed for jobs that do not need datasets. Dataset IDs must be unique and refer to existing datasets whose source files are available.

Custom processing scripts that use datasets should accept the optional `--input-manifest <path>` argument. When it is absent, the script should run without dataset inputs. If the argument is supplied but the manifest file is missing, the runner fails rather than silently ignoring the problem. The manifest and input files must be accessible from the execution environment, including SLURM compute nodes. For this proof of concept, source files are referenced in place rather than copied; keep them available and unchanged while jobs are queued or running.

## Main API routes

### Projects, surveys, and wells

- `POST/GET /api/v1/projects`, `GET/PATCH/DELETE /api/v1/projects/{project_id}`
- `POST /api/v1/projects/{project_id}/surveys`, `GET /api/v1/surveys`, `GET/PATCH/DELETE /api/v1/surveys/{survey_id}`
- `POST /api/v1/projects/{project_id}/wells`, `GET /api/v1/wells`, `GET/PATCH/DELETE /api/v1/wells/{well_id}`

### Datasets

Upload multipart form data to `POST /api/v1/datasets` with:

- `file` (required)
- `kind` (required: `seismic`, `well_log`, `checkshot`, `deviation`)
- `project_id` (required unless inferred from the associated survey/well)
- `survey_id` for seismic only
- `well_id` for well-log, checkshot, and deviation only
- Supplying the association intended for another dataset kind is rejected.
- optional `name`, `crs`, and JSON-object `metadata_json`

Routes:

- `GET /api/v1/datasets` — filter by `kind`, `project_id`, `survey_id`, `well_id`
- `GET /api/v1/datasets/{dataset_id}` — resource metadata and subtype name
- `PATCH /api/v1/datasets/{dataset_id}` — mutable description/association fields only
- `DELETE /api/v1/datasets/{dataset_id}`
- `GET /api/v1/datasets/{dataset_id}/download` — stream the original stored file
- `GET /api/v1/datasets/{dataset_id}/file-info` — verify the stored file exists
- `GET /api/v1/datasets/{dataset_id}/header` — SEG-Y/LAS headers
- `GET /api/v1/datasets/{dataset_id}/data` — frontend-ready parsed payload for seismic, well-log, checkshot, or deviation data
- `GET /api/v1/datasets/{dataset_id}/table` — paginated checkshot/deviation rows
- `GET /api/v1/datasets/{dataset_id}/validation`

### Curves

- `GET /api/v1/datasets/{dataset_id}/curves` — list curves in LAS source order
- `GET /api/v1/curves/{curve_id}` — get one curve by its stable ID
- `GET /api/v1/curves/{curve_id}/data?offset=0&limit=1000` — retrieve samples from the original LAS file

Curve samples are read from the stored source file, not duplicated into SQL. The curve ID is stable for that source-file version; re-ingestion/replacement should create a fresh catalog.

### Parsed data response contracts

`GET /api/v1/datasets/{dataset_id}/data` reads the stored source file on demand. It does not persist the full numerical arrays in SQL.

- `seismic`: `type`, `traces`, `traceHeaders`, `traceCount`, `scannedTraceCount`, `samplesPerTrace`, `sampleIntervalUs`, `formatCode`, optional `inlineRange`/`crosslineRange`, and `textualHeader`.
- `well_log`: `type: "well"`, `curves`, `units`, `rows`, optional `depthStart`/`depthStop`, and `nullValue`.
- `checkshot` and `deviation`: `type: "table"`, `headers`, `rows`, `rowCount`, and `delimiter`.

These are response payloads derived from the original file. Curve definitions remain independently addressable database resources, while numerical curve samples are read from the LAS file.

### Settings, GeoChat, and processing

- `/api/v1/settings` — runtime settings, including editable GeoChat model and generation parameters
- `POST /api/v1/geochat` — OpenAI-compatible LLM chat completion
- `/api/v1/processing-jobs` — local/SLURM job submission and status
- `/api/v1/capabilities` — supported dataset kinds, formats, and implementation status

GeoChat uses the Python OpenAI SDK against the configurable OpenAI-compatible base URL. Defaults are `http://172.28.34.1:8000/v1` and `google/gemma-4-26B-A4B-it`; the placeholder API key is kept in the backend. Configure the model, system prompt, temperature, max tokens, top-p, frequency/presence penalties, and timeout through the `geochat` section of `PATCH /api/v1/settings` or the frontend Settings page.

Example request:

```json
{
  "conversation_id": "c2d5f47e-ecb4-44e7-ae52-7b70c08c2a1a",
  "messages": [
    {"role": "user", "content": "Explain seismic amplitude in simple terms."}
  ]
}
```

Send the current conversation's user/assistant messages on each request; the backend prepends the configured system prompt. Successful conversations are archived as readable UTF-8 text files at `<active site_data root>/geochat/<conversation-uuid>.txt`. The file contains a JSON metadata header (model settings, timestamps, request count, token usage) followed by the conversation transcript. The existing `AppSetting` SQLite table persists GeoChat settings; no new database table is required. Current chat history is managed by the frontend session and is not automatically restored from the archive.

## Parsed dataset payloads for the frontend

`GET /api/v1/datasets/{dataset_id}/data` resolves the source file through the dataset's stored `storage_key` and parses it on demand. Parsed numerical data is not duplicated in the SQL database.

The response includes `dataset_id` plus one of these discriminated payloads:

- **Seismic (`type: "seismic"`)**: `traces`, `traceHeaders` (`inline`, `crossline`), `traceCount`, `scannedTraceCount`, `samplesPerTrace`, `sampleIntervalUs`, `formatCode`, optional `inlineRange` and `crosslineRange`, and `textualHeader`.
- **Well log (`type: "well"`)**: `curves`, `units`, numeric `rows`, optional `depthStart`, `depthStop`, and `nullValue`.
- **Checkshot/deviation (`type: "table"`)**: `headers`, string `rows`, `rowCount`, and `delimiter`.

For SEG-Y, `max_traces` controls how many traces are included in the response (default `500`, maximum `5000`). `traceCount` is the total file trace count; `scannedTraceCount` is the number of sample traces returned. Inline/crossline ranges are scanned from trace headers across the file. LAS and text-table payloads are returned in full subject to safety limits (100,000 LAS rows and 250,000 table rows); larger files return HTTP 422 and should be consumed through the existing paginated curve/table endpoints.

Example requests:

- `/api/v1/datasets/<seismic-id>/data?max_traces=250`
- `/api/v1/datasets/<well-log-id>/data`
- `/api/v1/datasets/<checkshot-or-deviation-id>/data`

## File and parser behavior

| Kind | Accepted extension | Required parent | Read endpoint |
|---|---|---|---|
| `seismic` | `.sgy`, `.segy` | Survey | `/datasets/{id}/data`, `/datasets/{id}/header`, `/download` |
| `well_log` | `.las` | Well | `/datasets/{id}/data`, `/datasets/{id}/header`, `/datasets/{id}/curves` |
| `checkshot` | `.txt` | Well | `/datasets/{id}/data`, `/datasets/{id}/table` |
| `deviation` | `.txt` | Well | `/datasets/{id}/data`, `/datasets/{id}/table` |

Deviation tables expect three numeric columns. Checkshot tables support both the legacy three-column `MDKB, TWT, TVDSS` layout and a two-column `TVDSS, TWT` layout with comment headers such as `#TVDSS (m)` and `#TWT (ms)`. The two-column format is preserved as TVDSS/TWT only; the parser does not synthesize MDKB. The scientific `parse_tdr_table` reader supports the same two-column format for Stage 00/01 processing.

## Tests

```bash
pytest -q
```

The test environment must install the packages listed in `requirements.txt`, including `lasio` and `segyio`.


## Unified processing script directory (v8)

All user-managed Python processing scripts and the local/SLURM runner shell scripts live together under the active Data Directory:

```text
site_data/
  scripts/
    process_A_foundations.py
    pyrunner.local.sh
    pyrunner.sbatch.sh
```

The API resolves `scripts/` dynamically from the active storage root selected by the runtime Storage settings. On every backend startup, all bundled defaults under `backend/scripts/` are synchronised recursively into the active `<data-root>/scripts/` directory. Matching default files are refreshed from the backend package, including nested directories; runtime-only files are retained and are never deleted by synchronisation. Consequently, edits to a bundled default file in `site_data/scripts/` are replaced on the next restart. Keep persistent user-specific settings and custom scripts in files that are not shipped under `backend/scripts/`. If the configured primary root is not writable, the same writable fallback-root selection used by storage is applied.

Python processing scripts are managed through these endpoints:

- `GET /api/v1/scripts` — list `.py` scripts (metadata only).
- `GET /api/v1/scripts/{script_id}` — return script metadata and UTF-8 source.
- `POST /api/v1/scripts` — upload a Python file as multipart field `file`; returns `409` rather than overwriting an existing filename.
- `PUT /api/v1/scripts/{script_id}` — replace source using JSON `{ "content": "..." }`.
- `DELETE /api/v1/scripts/{script_id}` — delete a Python script.

`script_id` is the `.py` filename, not a filesystem path. Script management accepts only safe `.py` filenames, limits source files to 2 MiB, and prevents directory traversal. The listing intentionally returns Python workers only; shell runners remain in the same directory and are selected by the existing `shell_script` field in `POST /api/v1/processing-jobs`. The supported runner filenames remain `pyrunner.local.sh` and `pyrunner.sbatch.sh`.

The Processing Jobs API resolves both the selected Python worker and runner from `<active-data-root>/scripts/`. `backend/scripts/` is the version-controlled source of bundled defaults; startup synchronisation keeps the runtime copy current. Ensure the runner files are executable/readable as appropriate for your environment and that the same data root is visible to SLURM compute nodes.

**Security:** Python source can execute arbitrary code. Restrict API access to trusted users and only upload/modify scripts you intend to execute. This version does not provide a sandbox or per-user authorization.

Example script upload:

```bash
curl -X POST http://localhost:8888/api/v1/scripts \
  -F 'file=@process_A_foundations.py'
```

Example source update:

```bash
curl -X PUT http://localhost:8888/api/v1/scripts/process_A_foundations.py \
  -H 'Content-Type: application/json' \
  -d '{"content":"print(\"hello\")\\n"}'
```

### Processing job inspection

- `GET /api/v1/processing-jobs` lists recent processing jobs (supports `limit` and `offset`).
- `GET /api/v1/processing-jobs/{job_id}` returns job status and metadata.
- `GET /api/v1/processing-jobs/{job_id}/files` lists files in the job directory.
- `GET /api/v1/processing-jobs/{job_id}/files/{file_name}` reads permitted UTF-8 JSON, log, and text files up to 2 MiB.

The file endpoints are scoped to the selected job directory and reject path traversal, symlinks, and unsupported file types. The placeholder dataset quality-check and feature-generation endpoints have been removed.

### Unassigned well-related datasets

Well-log (LAS), deviation (TXT), and checkshot (TXT) datasets may be uploaded before a well is selected. Their `well_id` is nullable and can be assigned later through the dataset association API. On startup, the backend migrates existing SQLite/PostgreSQL subtype tables to allow null well associations. Seismic datasets still require a survey association.

## GeoChat organization

GeoChat runs inside the existing FastAPI application; it is not a second server.
Its implementation is grouped under `app/geochat/`:

- `api.py` exposes `/api/v1/geochat`.
- `agent.py` asks the configured LLM to select a skill and orchestrates execution and answer generation.
- `skills/registry.py` discovers self-contained skill directories at startup.
- Each skill keeps its `SKILL.md` and optional `handler.py` together. A skill may also ship `formatting.py` (`render_markdown(result)`, the fixed answer layout) and `intent.py` (`parse_request(message, history)`, the questions it answers); the registry discovers both by file name.
- Process A-F each have a run skill (`process_<x>_...`, pre-flight only) and a result reader (`process_<x>_..._output_analysis`). Ask Geo routes "Explain the results of Process D" to the reader, finds the latest run, and renders its report; `POST /api/v1/evidence/{job_id}/explain` uses the same reader for the Evidence tab.

The GeoChat endpoint accepts an optional `project_id` alongside `conversation_id` and `messages`. The frontend sends the currently selected project ID so project-specific skills can operate in the correct scope. The response includes the selected `skill`. Conversation transcripts continue to be archived as plain-text files under the configured storage root's `geochat/` directory; no separate chat database or server is introduced.

## GeoChat skills

GeoChat uses an LLM-driven skill selector. Each skill is a self-contained directory under
`app/geochat/skills/` containing a `SKILL.md` description and, when executable, a
`handler.py` exposing `execute(db, project_id, parameters)`. A skill may omit
`handler.py` when it only supplies domain instructions to the LLM.

The registry discovers skill directories at application startup. Add a new skill by
creating a directory with a valid `SKILL.md`; add `handler.py` when the skill must
perform application operations. Restart the backend after adding or changing skills.

Processing skill templates are under `app/geochat/skills/_templates/`:

- Copy `process_x/` and rename it `process_<operation>/` for execution.
- Copy `process_x_output_analysis/` and rename it
  `process_<operation>_output_analysis/` for interpreting that operation's outputs.
- Replace the template instructions and handler implementation before using the skill.

Template directories are intentionally excluded from skill discovery. Processing
handlers must call existing backend processing services; they must not execute
arbitrary LLM-generated Python code. Execution skills should validate inputs and
require user confirmation before submitting jobs.

The frontend sends the currently selected `project_id` with each GeoChat request.
The backend remains the single FastAPI application and uses the existing shared
SQLAlchemy database and configured storage root.

## Linking existing datasets

The Input Data tab can register a file already available on the backend filesystem
without copying it into the project's `site_data/uploads` directory. Configure the
per-deployment allowlist with `DATASET_ALLOWED_ROOTS` in `backend/.env`:

```bash
# Local development
DATASET_ALLOWED_ROOTS=/home/elvinado/Downloads/input_data

# HPC deployment
DATASET_ALLOWED_ROOTS=/data/epic20/APGCE/input_data
```

On Linux, multiple roots may be separated with `:`. Only files whose resolved
paths remain inside an allowed root are accepted; symlink escapes are rejected.
Linked dataset records retain the original path and checksum. Removing a linked
record never deletes the source file. Existing uploaded datasets remain managed
and retain their existing deletion behaviour. The backend must have filesystem
read permission for every linked file.


## GeoGG Fresh / AskGeo migration

The staged migration boundary and Stage A compatibility foundation are documented
in [`docs/GEOGG_FRESH_MIGRATION.md`](docs/GEOGG_FRESH_MIGRATION.md). The initial
scientific package lives in `scripts/geogg_fresh/`; it deliberately does not
replace the existing project, dataset, processing-job, runner, SLURM or GeoChat
infrastructure. Stage implementations are not included until their roadmap
stage is migrated and scientifically validated.


### GeoGG Fresh scientific stages

Backend v62 includes GeoGG Stage 00 input inventory and Stage 01 well foundations.
Install `scripts/geogg_fresh/requirements-science.txt` in the dedicated scientific
runner environment, copy `scripts/A_foundations.yaml.example` to
`site_data/scripts/A_foundations.yaml`, configure the project paths/wells, then run
`process_A_foundations.py` through the existing local or SLURM processing
runner. The two stages execute sequentially within one processing job. The
wrapper accepts the runner's optional `--input-manifest` argument and forwards
it through `APGCE_INPUT_MANIFEST` to the shared GeoGG context.

#### Link local file browser

The Input Data tab's **Link local** dialog uses `GET /datasets/link-local/browse` to browse immediate children of the configured `DATASET_ALLOWED_ROOTS`. The endpoint only exposes configured roots and supported input files, resolves symlinks before listing, and rejects paths outside the allowlist. It returns server-side absolute paths for the existing `POST /datasets/link-local` registration flow; it does not upload or copy file contents.


## Excel well-list import

The Project tab can import `.xlsx` / `.xlsm` well lists. The importer detects a header
row containing `Well Name`, `RT`, `TD`, `X`, and `Y` (the header may follow title
or preamble rows). It creates new wells or updates matching wells by case-insensitive
name within the selected project. Existing well UIDs are preserved. RT and TD are
stored as optional numeric fields on the existing `wells` table; startup adds these
columns to existing databases without replacing the table. The source workbook is
also retained as a project-level `table` dataset with `format=xlsx` and
`metadata.table_type=well_list`.

## Latest update

Backend v71 fixes Stage 01 optional TDR QC reporting and resolves the synthetic-tie full-stack path from the registered Stage 00 inventory. See [docs/UPDATE_v71.md](docs/UPDATE_v71.md).

### Linked dataset storage roots

Linked datasets are resolved by the backend dataset registry, but their source
files must live on a filesystem visible to both the API and processing workers.
Configure `DATASET_ALLOWED_ROOTS` explicitly on every relevant node; do not rely
on developer-home or guessed cluster paths. For Linux, separate multiple roots
with `:`. Use managed dataset uploads when files must be stored under the
backend's configured storage root rather than referenced in-place. Processing
manifests record the API-resolved path and storage provenance for each dataset.

See `docs/STAGE_B_DATA_LIMITATIONS.md` for the current Stage B scientific data
readiness and acceptance criteria.

### Stage F survey-volume inference (v93)

`python scripts/process_B_inference.py --config scripts/B_inference.yaml` supports label-independent inference over registered seismic SEG-Y inputs. The default `inference.input_mode` is `survey_volume`: the processing job must explicitly register/select the full, near, mid, far and ultrafar SEG-Y datasets. Missing inputs now stop with a clear error instead of silently replaying the training-survey patches. Set `inference.input_mode: source_replay` only when intentionally reproducing the original training-survey well-path predictions. Survey-volume inference follows the originating foundation job recorded in the immutable model-release manifest; it validates that job and checks that the registered SEG-Y inputs have the same Stage 00 SHA-256 hashes. No separate foundation job ID is configured. The full-stack geometry is authoritative, per-channel IL/XL geometry is respected, and Stage 03 patch parity is verified before inference. Tabular and deep missing-data policies remain distinct. Outputs are streamed float32 NumPy prediction cubes (`.npy`, memory-mapped during writing) and geometry metadata under `stages/09_inference/volumes/`; see `docs/UPDATE_v93.md` for the contract and scientific boundaries. The default inference config selects the latest CNN3D multi-task release and all three prediction heads.

## GeoGG Stage 14 — Prospect screening

Stage 14 is an independent worker launched with `scripts/process_F_prospect.py` and configured by `scripts/F_prospect.yaml`. It consumes the registered Foundation (Stage 01 trajectories and Stage 02 interpreted logs), Inference (Stage 09 trace grid), Volume-products (Stage 11 manifest), and Geobody (Stage 12 labels/catalogue/spatial exports) jobs. It defines local well-anchored prospects, calculates GRV under regional T-Z models, evaluates well-log NTG/PHIT/SWT, and reports cutoff and block-bootstrap sensitivity. HCPV is only a gated uniform-property screening scenario when interpreted well-log SWT is supported; Stage 11 SWT inference is not required. See `docs/UPDATE_v152.md` for source contracts, outputs, and limitations.

### Backend v153 — Stage 14 anchor-intersection audit

Stage 14 now exports `prospect_well_intersection_audit.csv` for all configured wells and reports missing/unsupported anchor intervals explicitly. Empty anchor-interval dataframes retain a defined schema, preventing the v152 `n_path_samples` AttributeError. Non-anchor wells are never promoted automatically; anchor changes remain explicit YAML configuration decisions. See `docs/UPDATE_v153.md`.


### Backend v154

Stage 14 supports automatic selection of all qualifying prospect anchors from the explicit configured well inventory; strict explicit-anchor mode remains available. See `docs/UPDATE_v154.md`.
