# Process: Process B Inference

## Description
Run `process_B_inference.py` (operation `geogg_stage_f_08_09`). It is the second step of the seismic AI engine and builds on a completed Process A (foundations) run:

| Stage | Plain name | What it produces |
|---|---|---|
| 08 | Final exam on unseen wells | Held-out scores for every Process A model, with intervals, paired comparisons, calibration and prediction bands |
| 09 | Prediction cubes for the whole survey | One cube each for sand versus shale, shale volume and porosity, plus a validity mask and the cube geometry |

The intended result is an honest measure of how well the models work on wells they never trained on, and survey-wide prediction cubes for interpretation.

## Use this skill when
- The user explicitly asks to run Process B, the inference process, the evaluation, or stages 08 to 09.
- The user asks to produce prediction cubes or volumes from the trained models.
- The user asks what Process B needs, or whether a Process A run is ready for it.
- The user asks to configure or prepare this operation for execution.

To explain a run that has already finished, use `process_b_inference_output_analysis`.

## Required inputs
| Input | Need | Why it matters |
|---|---|---|
| A completed Process A (foundations) job | Required | Supplies the trained models, the sealed held-out predictions and the seismic cut-out definition |
| Its model release, status `ready` | Required | The packaged models are what Process B loads |
| The registered seismic volumes used by Process A | Required, already linked | The cubes are predicted from the full, near, mid, far and ultra-far stacks |

The process takes no dataset selection and no numeric parameters. Pass the Process A job ID to the handler as `foundation_job_id`.

## Preconditions
- The Process A job has status `succeeded`, all eight stages recorded, no failed self-checks and no blockers.
- Its sealed predictions (`outer_test_predictions.csv`), model registry and deep model file are still in the job folder.
- The seismic volumes registered for Process A are still available at the same location.
- A GPU node is available. In the reference run the whole-survey prediction took about 9 hours and wrote about 25 GB.

## Instructions
1. Confirm the active project and identify which Process A job to build on. If the user has several, ask which one; do not guess.
2. Call the handler with `parameters.foundation_job_id`. It checks the job and returns `ready`, a list of `issues` and a plan. With no job ID it returns the requirements list.
3. If there are blocking issues, explain each one in a sentence and say how to fix it, using `how_to_fix`. Do not submit.
4. If there are only warnings, explain what they mean for the results. In particular, tell the user when the Process A job has no validated well ties: the run will work, and every result stays exploratory.
5. Present the plan in plain words: which Process A job, what the two stages will do, how long the reference run took, and what will come out. Obtain explicit confirmation before execution.
6. Submit only through the approved processing service, with the script name returned by the handler.
7. Return the job identifier and submission status. Offer to explain the results with `process_b_inference_output_analysis` once the job has succeeded.

## How to talk to the user
- Use the plain stage names from the table above and mention stage numbers only as a reference.
- Be clear about the cost before asking for confirmation: this is a long GPU job that writes large files.
- Set expectations: stage 08 gives the honest scores, and those scores may be modest. The cubes are only as trustworthy as the model that makes them.
- Say what the run will not do: it does not retrain or reselect models, and it does not map oil or gas.
- Keep the confirmation message short: which foundation job, what will happen, how long, where the outputs go.

## What to tell the user to expect
- Stage 08 never refits a model. It only scores predictions that were sealed during Process A, so the exam cannot be re-sat.
- The cubes are built by the deep model on the seismic stacks. That model is not necessarily the best-scoring one for every property.
- The cubes cover the whole survey and all time samples, including parts with no well control. Those parts are extrapolation.
- Both stages finish as `COMPLETED`. That means they ran; quality is read from the held-out scores.
- If the job is interrupted during stage 09, it can resume from its checkpoint.

## Response format
The pre-flight answer is rendered by `formatting.render_markdown`: readiness line, what was checked, issues with their fixes, the inputs table, what the run will do, expected outputs and the next step.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`. The handler validates and plans; it does not submit the job.

## Safety
Never fabricate a job ID, parameter values, or job status. Never bypass backend validation or user confirmation. Never submit when the handler reports a blocking issue, and never build on a failed, running or partly completed Process A job. Never execute user-provided Python source. Quote the reference run time as an observation, not a promise.
