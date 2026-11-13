# Process: Process A Foundations

## Description
Run `process_A_foundations.py` (operation `geogg_fresh_stages_00_07`). It builds the foundations of the seismic AI engine from wells and seismic in eight stages:

| Stage | Plain name | What it produces |
|---|---|---|
| 00 | Data check-in | Inventory of wells, logs and seismic volumes, with grid checks |
| 01 | Depth-to-time conversion | Well paths and a time-depth relation for every well |
| 02 | Rock and fluid labels | Sand, shale and fluid labels from interpreted logs |
| 03 | Seismic cut-outs | A small seismic cube at each point along each well |
| 04 | Seismic attributes | Amplitude, frequency, continuity and contrast measurements |
| 05 | Training tables | Inputs joined to answers, with whole wells assigned to test groups |
| 06 | Attribute selection | Attributes that behave consistently from well to well |
| 07 | Model training | Tuned candidate models and a packaged model release |

The intended result is a completed job folder plus a model release, ready for the held-out evaluation process. Process A itself reports practice scores only.

## Use this skill when
- The user explicitly asks to run Process A, the foundations process, or stages 00 to 07.
- The user asks what Process A needs, or to check whether their project data are ready for it.
- The user asks to configure or prepare this operation for execution.

To explain a run that has already finished, use `process_a_foundations_output_analysis`.

## Required inputs
All inputs are project datasets. The process works on all selected wells together; it has no target well.

| Input | Need | Why it matters |
|---|---|---|
| Well list table (.xlsx) | Required, exactly one | Well names, surface X/Y, rotary-table elevation, total depth |
| Well logs (.las), one per well | Required, at least 4 wells | Interpreted VSH, PHIT and SWT are the answers the models learn from |
| Deviation survey (.txt), one per well | Required for every well with logs | Converts measured depth to true vertical depth and map position |
| Time-depth or checkshot tables (.txt) | At least 1, 3 or more recommended | Anchor the depth-to-time conversion |
| Full-stack seismic (.sgy) | Required | Defines the survey grid and the reference volume |
| Near, mid, far, ultra-far stacks (.sgy) | Recommended | Amplitude-contrast attributes |
| P-impedance, S-impedance, Vp/Vs (.sgy) | Optional | Conditional inputs for the stacks-plus-inversion comparison |

Processing parameters: none. Cutoffs, patch size, resolutions and model grids are fixed inside the script and recorded in each run's `config_snapshot.yaml`.

## Preconditions
- Every well log, deviation survey and time-depth table is associated with its well in the project.
- All seismic volumes belong to one survey and share one grid.
- Wells and seismic use the same coordinate system. The process does not transform coordinates.
- Logs are indexed in measured depth and contain interpreted VSH, PHIT and SWT curves.
- No earlier processing step is required. Process A is the first step of the engine.

## Instructions
1. Confirm the active project and gather the datasets the user wants to use. If they say "everything", take all wells and seismic volumes of the challenge data.
2. Call the handler with `parameters.datasets` set to those dataset descriptors. It checks coverage and returns `ready`, a list of `issues` and a plan. With no datasets it returns the requirements list.
3. If there are blocking issues, explain each one in a sentence and say how to fix it, using `how_to_fix`. Do not submit.
4. If there are only warnings, explain what each will cost in quality and let the user decide.
5. Present the plan in plain words: how many wells, which seismic volumes, what the eight stages will do, and what will come out. Obtain explicit confirmation before execution.
6. Submit only through the approved processing service, with the script name and the dataset IDs returned by the handler.
7. Return the job identifier and submission status. Offer to explain the results with `process_a_foundations_output_analysis` once the job has succeeded.

## How to talk to the user
- Use the plain stage names from the table above and mention stage numbers only as a reference.
- Say what a missing input will cost, not just that it is missing. For example: "Without sonic and density logs the well ties cannot be verified, so results stay exploratory."
- Set expectations before the run: it produces training foundations and practice scores, not a final accuracy and not a map of sand or hydrocarbons.
- Keep the confirmation message short: inputs, what will happen, where the outputs go.

## What to tell the user to expect
- Stages normally finish as `CONDITIONAL`. That means the stage passed its self-checks and its results rest on stated assumptions. It is not a failure.
- If no well has sonic and density logs, no synthetic well tie is possible and every tie is provisional.
- Oil and gas classes are trained only when independent fluid evidence exists. Otherwise the fluid task is "water-bearing versus hydrocarbon potential".
- The Total Sand Probability volume is never used as a model input, because it is too close to the answer.

## Response format
The pre-flight answer is rendered by `formatting.render_markdown`: readiness line, what was checked, issues with their fixes, the inputs table, what the run will do, expected outputs and the next step.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`. The handler validates and plans; it does not submit the job.

## Safety
Never fabricate input datasets, parameter values, or job status. Never bypass backend validation or user confirmation. Never submit when the handler reports a blocking issue. Never execute user-provided Python source. Do not promise a run time or a result quality before the job has finished.
