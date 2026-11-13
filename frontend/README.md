# APGCE GeoHackathon 2026 UI

React + TypeScript frontend for the geoscience workflow. The UI connects to the GeoHackathon FastAPI backend and contains no bundled geological sample datasets, synthetic viewer traces, sample chat answers, or sample evidence records.

## Requirements

- Node.js 20 or newer
- The `backend_v7` Python API

## Start the Python backend

From the backend project directory, activate its virtual environment and run:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

The backend health endpoint should respond at `http://127.0.0.1:8080/health`. Keep this terminal running.

## Start the frontend

In a separate terminal, from this UI project directory:

```bash
npm install
npm run dev -- --host 127.0.0.1 --port 4173 --strictPort
```

Open `http://127.0.0.1:4173`. Vite proxies `/api/*` and `/health` to `http://127.0.0.1:8080`, so browser requests use the same origin and do not require backend CORS changes for local development.

## Code organization

The `src/` directory is intentionally kept to four folders:

- `src/api/` — backend HTTP requests and API types (`backend.ts`).
- `src/app/` — application UI, startup entry point, shared frontend data types, and related tests.
- `src/parsers/` — local LAS/SEG-Y parsing and parser tests.
- `src/styles/` — global theme, layout, and viewer CSS files.

Keep this structure small for now. Add files to these existing folders when needed rather than introducing new architectural layers prematurely.

Local LAS parsing is intentional: it gives the Well Viewer immediate curve data and keeps the browser preview independent of backend parser availability. The backend remains responsible for persistence and dataset metadata.

## Backend integration

- On startup, the UI checks backend health and loads the backend's projects. It selects the most recently created project when one exists; it does not create a project automatically, so an empty backend stays empty until you choose **New**.
- Project management covers all five project API operations: list (`GET /api/v1/projects`), create (`POST /api/v1/projects`), retrieve details (`GET /api/v1/projects/{project_id}`), edit name and description (`PATCH /api/v1/projects/{project_id}`), and delete (`DELETE /api/v1/projects/{project_id}`). The sidebar displays the selected project's description and creation date.
- Use the sidebar controls to switch projects, create a project, edit its name and description, delete a project, or refresh the project list. Deleting a project permanently removes its surveys, wells, datasets, and stored uploaded files, so the UI asks for confirmation first. Deleting the last project leaves the project list empty; it does not silently create a replacement.
- The selected project's `.sgy`, `.segy`, and `.las` inventory is loaded from `GET /api/v1/datasets?project_id=...`.
- SEG-Y and LAS files are parsed locally for immediate previews and uploaded to `POST /api/v1/datasets` with the backend-required parent association. SEG-Y requires a survey; LAS, deviation, and checkshot datasets require a well. The local parser remains the viewer data source; backend curve-catalog and curve-sample APIs are still available for dataset inspection.
- Uploaded originals and metadata persist in the backend. Selecting a dataset loaded from the backend downloads the original file from `GET /api/v1/datasets/{dataset_id}/download` and builds its preview locally when a local preview is not already available.
- CSV/TSV files can be previewed locally, but are not uploaded: backend_v7 supports only `seismic`, `well_log`, `checkshot`, and `deviation` dataset kinds.
- Ask Geo and Evidence remain explicit empty states because the current backend does not provide chat or evidence endpoints. The UI does not fabricate responses or evidence.

## Validate

```bash
npm test
npm run build
```

Note: the frontend build could not be executed in the packaging environment because `npm ci` timed out while fetching dependencies. The modified TypeScript/TSX files were syntax-checked, but a full dependency-backed build remains to be run on your machine.

### Survey API integration

Survey management is scoped to the selected backend project in the workspace sidebar. The frontend uses the backend_v7 FastAPI survey endpoints:

- `POST /api/v1/projects/{project_id}/surveys` — create a survey with name, description, CRS, and metadata.
- `GET /api/v1/surveys?project_id={project_id}` — list surveys for the selected project.
- `GET /api/v1/surveys/{survey_id}` — retrieve survey details.
- `PATCH /api/v1/surveys/{survey_id}` — update survey name, description, CRS, and metadata.
- `DELETE /api/v1/surveys/{survey_id}` — delete a survey; linked datasets are retained and detached.
- `GET /api/v1/surveys/{survey_id}/datasets` — inspect datasets linked to a survey.

Create/edit forms validate that metadata is a JSON object. Deletion requires confirmation and explains that associated datasets are not deleted. Survey list refresh is available from the workspace sidebar. The dataset-link inspection action reports the linked dataset count; it does not change dataset associations.

### Assign seismic datasets to surveys

In **Input Data → Data inventory**, select an uploaded seismic dataset to open its file preview. The preview panel includes a **Survey association** selector. Choose a survey to link the dataset, or choose **Unassigned — no survey** to detach it. The selection is saved immediately through `PATCH /api/v1/datasets/{dataset_id}` using `survey_id`.

Once linked, the dataset appears under that survey in Seismic Viewer and is identified by survey name in Data inventory. Only surveys in the currently selected project are offered.

### Seismic Viewer: assigning unassigned data

The Seismic Viewer includes an **Unassigned seismic** selector so seismic datasets without a survey association can be opened directly in the viewer. Select an unassigned dataset, inspect its preview, then use **Assign selected seismic to survey** to persist the association through `PATCH /api/v1/datasets/{dataset_id}`. After a successful assignment, the selected survey becomes active and the dataset appears in that survey's linked seismic list. This avoids requiring a dataset to already belong to a survey before it can be viewed and assigned.


## Seismic Viewer UI cleanup (v16)

The Seismic Viewer uses a consistent page title, a single survey selector, an unassigned-seismic selector, and a linked-dataset selector. Unassigned datasets can be previewed while a destination survey remains selected; use the single “Assign selected dataset” action to link the current file. Survey editing remains available from the header.

### Data inventory actions

The Data inventory now includes **Edit** and **Delete** actions for backend datasets. Edit lets you change the dataset display name and choose an existing survey association (or `Unassigned`). Delete requires confirmation and permanently removes the backend dataset record, stored original file, and associated curve rows. These actions require a connected backend.

## Well APIs (v18)

The project workspace now supports well CRUD using the FastAPI well endpoints:

- `POST /api/v1/projects/{project_id}/wells` — create a well
- `GET /api/v1/wells?project_id=...` and `GET /api/v1/wells/{well_id}` — list/retrieve wells
- `PATCH /api/v1/wells/{well_id}` — edit name, UWI, X/Y coordinates, CRS, and JSON metadata
- `DELETE /api/v1/wells/{well_id}` — delete a well; linked datasets are retained and detached
- `PATCH /api/v1/datasets/{dataset_id}` with `well_id` — link/unlink a LAS dataset to a well

Well-log dataset association is available from Input Data > Data Inventory's selected-file preview. The well management controls are in the project workspace sidebar. Coordinate values may be left blank; metadata must be a JSON object.

## Processing and pipeline APIs (v27)

The Processing workspace exposes the backend's resource-oriented pipeline endpoints:

- `POST /api/v1/datasets/{dataset_id}/quality-checks` (QC)
- `POST /api/v1/datasets/{dataset_id}/features` (FEATGEN)
- `POST /api/v1/processing-jobs` (MLPROC)

It loads datasets from the selected project, displays pipeline readiness from `GET /api/v1/capabilities`, allows testing the operation endpoints, and records the request outcomes in a session-only log. The current backend registers these three operations as HTTP `501 Not Implemented` placeholders. The UI reports that state explicitly; it does not claim that processing runs or jobs are created.

Navigation update: Processing is a primary navigation item immediately below Well Viewer. System Status and Runtime Settings are no longer primary tabs; both are in the sidebar footer, with Runtime Settings below System Status.


## Backend v7 alignment

This frontend is aligned to `backend_v7.zip`'s polymorphic dataset model:

- Dataset `kind` is immutable after creation. Edit actions update the display name, CRS, metadata, and valid parent association only.
- Uploads use the backend's required relationship rules: `seismic` → `survey_id`; `well_log`, `deviation`, and `checkshot` → `well_id`. The upload flow asks you to select an existing survey or well.
- Dataset inventory uses the backend's paginated response (`count`, `limit`, `offset`, `items`) and requests up to 200 records.
- Parsed table paging follows the v7 response shape (`columns`, `row_count`, `returned_rows`, `rows`).
- Processing job submission is represented by `POST /api/v1/processing-jobs` with `python_script`, `shell_script`, and optional `dataset_ids`; status is read from `GET /api/v1/processing-jobs/{job_id}`. QC and feature endpoints remain reserved operations and may return `501 Not Implemented`.

The archive intentionally excludes `node_modules`. Install dependencies in the frontend directory with `npm ci`, then validate with `npm run build` and `npm test`.

## Frontend tab organization (v24 refactor)

The main navigation now uses dedicated tab boundary components under `src/app/tabs/`:

- `ProjectTab.tsx`
- `InputDataTab.tsx`
- `SeismicViewerTab.tsx`
- `WellViewerTab.tsx`
- `AskGeoTab.tsx`
- `EvidenceTab.tsx`

These components preserve the existing page CSS classes and provide explicit component boundaries while the stateful application workflows remain coordinated in `App.tsx`. Processing continues to use the existing `ProcessingPage` component. This conservative pass avoids moving stateful handlers and API operations across module boundaries before they can be independently typed and regression-tested.

## v25 frontend refactor pass

- Moved the complete Processing page implementation and its page-local state/effects into `src/app/tabs/ProcessingTab.tsx`.
- Replaced the Ask Geo wrapper with a standalone tab component that receives its query state and send handler through props.
- Replaced the Evidence wrapper with a standalone tab component containing its full placeholder UI.
- Kept backend requests and application-wide project/dataset state in their existing locations to avoid changing current behavior.
- Validation: all 14 non-declaration TypeScript/TSX files passed TypeScript syntax transpilation. A full type-check/build could not be completed because the dependency installation in this environment is incomplete.

The Project, Input Data, Seismic Viewer, and Well Viewer tab JSX still require deeper extraction in a subsequent pass; they remain rendered by `App.tsx` behind their current tab boundaries.

## Refactor status (v26)

This release continues from v25 without changing application behavior. The tab boundary components remain in `src/app/tabs/`; `ProcessingTab`, `AskGeoTab`, and `EvidenceTab` contain their extracted implementations. The Project, Input Data, Seismic Viewer, and Well Viewer markup still depends on substantial shared state and handlers in `App.tsx`, so those four implementations have not yet been fully moved into their own files. They should be extracted with explicit typed props in a follow-up pass, then checked with a clean dependency install and production build.

Generated TypeScript incremental-build files are intentionally excluded from this source archive.


## v27 refactor checkpoint

The seven tab files are currently present, but four remain wrapper components whose main JSX is still authored in `App.tsx`: Project, Input Data, Seismic Viewer, and Well Viewer. This archive does not claim those four tabs have been fully extracted. Moving their JSX safely requires passing the exact state, handlers, and helper components used by each tab; avoid a blind move that changes behavior or hides type errors behind `any`.

Validation: archive integrity is checked when packaging. A clean TypeScript check, production build, and browser runtime check have not been completed in this environment.


## v28 tab extraction

The Project, Input Data, Seismic Viewer, and Well Viewer JSX has now been moved into their respective `src/app/tabs/*Tab.tsx` modules. `App.tsx` continues to own application state and cross-tab handlers, which are passed through a shared tab context. This is a behavior-preserving structural refactor; validate against the backend and representative project/data workflows before release.

## v30 function ownership refactor

- Extracted the seismic preview, well-track preview, table preview, and paginated deviation/checkshot table viewer into `src/app/tabs/ViewerComponents.tsx` so the tab files own their display components rather than receiving them from `App.tsx`.
- Extracted runtime settings UI and its local load/edit/save state into `src/app/SettingsPage.tsx`.
- Moved backend dataset-to-inventory mapping, dataset size formatting, and SEG-Y detection into `src/app/data.ts` for reuse.
- Moved the Input Data tab's dataset-kind icon helper into `InputDataTab.tsx`.
- Kept project/survey/well CRUD, uploads, dataset API operations, navigation, and shared inventory state in `App.tsx` because they coordinate shared application state and backend workflows.
- Validation: TypeScript syntax transpilation passed. A full build/type check was unavailable because dependencies (`react`, `vite`, and related packages) are not installed in this environment; run `npm ci`, `npm run build`, and `npm test` locally.


## Ask Geo

Ask Geo sends the current conversation to `POST /api/v1/geochat`. The backend uses its persisted OpenAI-compatible model configuration and archives successful chat turns under `site_data/geochat` (or the active fallback storage root). The Settings page exposes the system prompt, model, endpoint, temperature, token limit, top-p, penalties, and request timeout. Project datasets are not yet automatically added to chat context.


## v33 settings alignment

The Settings page clarifies primary/fallback storage selection and explains that the database URL is bootstrap configuration, not a database-backed editable setting. The default SQLite path is `backend/data/geohackathon.db`; override it with `DATABASE_URL` in `backend/.env` when required.


## Excel well-list import

The Project tab includes **Load well list** for `.xlsx` / `.xlsm` workbooks.
The importer detects `Well Name`, `RT`, `TD`, `X`, and `Y` columns, creates
new wells, and updates matching project wells by case-insensitive name while
preserving their existing UIDs. The uploaded workbook is retained as a
project-level Table dataset (`kind=table`, Excel format, `table_type=well_list`).
