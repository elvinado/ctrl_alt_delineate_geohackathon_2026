# Process: Process E Well Geobody Analysis

## Description
Run `process_E_well_geobody_analysis.py` (stage 13). It lays the well paths over the candidate geobodies of a completed Process D run. It trains nothing and predicts nothing.

| Step | What it does |
|---|---|
| Preflight | Confirms the source jobs link up and the well paths, label volume and catalogue match their checksums |
| Intersections | For each well, records which path samples lie inside a body and groups them into continuous stretches |
| Selected body | For each well, takes the largest body it meets and tries to give it a depth range and predicted porosity and water-saturation statistics |

The intended result is a table of which wells pass through which bodies. It is geometry only: it does not compare the bodies with well logs.

## Use this skill when
- The user explicitly asks to run Process E, the well geobody analysis, or stage 13.
- The user asks which wells cross the geobodies and no completed Process E run exists yet.
- The user asks what Process E needs, or whether a Process D run is ready for it.

To explain a run that has already finished, use `process_e_well_geobody_analysis_output_analysis`.

## Required inputs
| Input | Need | Why it matters |
|---|---|---|
| A completed Process D (geobody) job | Required | Supplies the body label volume, the catalogue and the per-trace index |
| The Process A job it descends from | Required, linked automatically | Well positions in seismic time come from its depth-to-time conversion |
| A list of wells | Optional | Without one, every well with a complete path is analysed |
| Porosity and water-saturation cubes | Optional | Needed only for the property statistics |

Pass the Process D job ID to the handler as `source_geobody_job_id`, and optionally `wells` as a list of names.

## Preconditions
- The Process D job has status `succeeded`, its exports reconcile, and its label volume, catalogue and index are still in the job folder.
- The Process A, B and C jobs it descends from still exist, because the preflight verifies the links.
- Each requested well has a path in the Process A job.

## Instructions
1. Confirm the active project and which Process D job to analyse. If there are several, ask; do not guess.
2. Call the handler with `parameters.source_geobody_job_id` and, if the user named wells, `parameters.wells`. It returns `ready`, a list of `issues` and a plan. With no job ID it returns the requirements.
3. If there are blocking issues, explain each in a sentence with its `how_to_fix`. Do not submit.
4. If the handler warns `one_dominant_body`, tell the user plainly before they confirm: every well will meet the same background body and the result will not tell the wells apart. Recommend fixing Process D first.
5. If it warns `property_cube_missing`, say that the porosity and saturation statistics will be skipped.
6. Present the plan briefly: which job, which wells, what the steps do, how long the reference run took. Obtain explicit confirmation.
7. Submit only through the approved processing service, with the script name returned by the handler.
8. Return the job identifier and submission status. Offer `process_e_well_geobody_analysis_output_analysis` once the job has succeeded.

## How to talk to the user
- Keep it short: which job, which wells, any warning, how long.
- Say what the run will not do: it does not check the bodies against the logs, and it makes no fluid or volume claim.
- Say "passes through a body", not "confirms a body".

## What to tell the user to expect
- For each well the selected body is the largest one it meets, by voxel count. That is not a quality ranking.
- Depths are given only inside the time range the time-depth models cover. A body reaching deeper gets no depth.
- Without validated well ties, each well's position in the cube is an estimate.
- Every result is labelled exploratory.

## Response format
The pre-flight answer is rendered by `formatting.render_markdown`: readiness line, what was checked, issues with their fixes, the inputs table, what the run will do, expected outputs and the next step.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`. The handler validates and plans; it does not submit the job.

## Safety
Never fabricate a job ID, well names, or job status. Never bypass backend validation or user confirmation. Never submit when the handler reports a blocking issue, and never analyse a Process D job whose exports are inconsistent. Never execute user-provided Python source. Quote the reference run time as an observation, not a promise.
