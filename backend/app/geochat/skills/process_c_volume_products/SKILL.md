# Process: Process C Volume Products

## Description
Run `process_C_volume_products.py`. It is the last step of the seismic AI engine and packages the results of a completed Process B (inference) run. It trains nothing and predicts nothing.

| Stage | Plain name | What it produces |
|---|---|---|
| 10 | Audit and summary | Checksum audit of inputs and outputs, cube statistics, imported held-out scores, dashboard, section figures, model cards |
| 11 | Export and validation set-up | One verified SEG-Y file per prediction cube, frozen predictions and a blind-well test protocol |

The intended result is a set of cubes that interpretation software can load, with a record of where they came from.

## Use this skill when
- The user explicitly asks to run Process C, the volume products process, or stages 10 to 11.
- The user asks to export the prediction cubes to SEG-Y, or to audit and package the results.
- The user asks what Process C needs, or whether a Process B run is ready for it.

To explain a run that has already finished, use `process_c_volume_products_output_analysis`.

## Required inputs
| Input | Need | Why it matters |
|---|---|---|
| A completed Process B (inference) job | Required | Supplies the cubes, their grid and the held-out scores |
| All cube tiles processed, every cube file present | Required | A partial cube would be exported as if complete |
| The full-stack SEG-Y used for the predictions | Required, already linked | Its headers are copied so the cubes load with the same geometry |
| Free disk for one SEG-Y file per cube | Required | Each file is about as large as the full-stack volume |

The process takes no dataset selection and no numeric parameters. Pass the Process B job ID to the handler as `source_inference_job_id`.

## Preconditions
- The Process B job has status `succeeded`, no failed self-checks and no blockers.
- Its cube files, validity mask, geometry file and evaluation tables are still in the job folder.
- The Process A job and model release it was built on still exist, because the audit verifies them.

## Instructions
1. Confirm the active project and which Process B job to package. If there are several, ask; do not guess.
2. Call the handler with `parameters.source_inference_job_id`. It returns `ready`, a list of `issues` and a plan. With no job ID it returns the requirements.
3. If there are blocking issues, explain each in a sentence with its `how_to_fix`. Do not submit.
4. If the handler warns that a cube's model showed no held-out skill, tell the user before they confirm. The export will make that cube look official.
5. Present the plan briefly: which job, what the two stages do, the disk space from `source.expected_segy_gb_total`, and what comes out. Obtain explicit confirmation.
6. Submit only through the approved processing service, with the script name returned by the handler.
7. Return the job identifier and submission status. Offer `process_c_volume_products_output_analysis` once the job has succeeded.

## How to talk to the user
- Keep it short: which job, what happens, how much disk, where the outputs go.
- Say plainly that this step checks and packages files. It does not improve or re-test the predictions.
- Use the plain stage names and mention stage numbers only as a reference.

## What to tell the user to expect
- The audit usually ends as `conditional`: no check failed, and limitations of earlier stages are inherited.
- Every cube is exported, whatever its quality. A verified file is not a verified prediction.
- Blind-well validation is prepared but not run. It needs independent wells to be registered afterwards.
- The original NumPy cubes stay in the Process B job; they are referenced, not copied.

## Response format
The pre-flight answer is rendered by `formatting.render_markdown`: readiness line, what was checked, issues with their fixes, the inputs table, what the run will do, expected outputs and the next step.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`. The handler validates and plans; it does not submit the job.

## Safety
Never fabricate a job ID, parameter values, or job status. Never bypass backend validation or user confirmation. Never submit when the handler reports a blocking issue, and never package a failed, running or partly completed Process B job. Never execute user-provided Python source. Quote the reference run time as an observation, not a promise.
