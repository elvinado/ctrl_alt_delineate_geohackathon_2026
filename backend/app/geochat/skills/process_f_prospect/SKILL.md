# Process: Process F Prospect

## Description
Run `process_F_prospect.py` (stage 14). It builds on the candidate geobodies of a completed Process D run and on the well paths and interpreted logs of the Process A run behind it. It trains nothing and predicts nothing.

| Step | What it does |
|---|---|
| Prospect definition | Each continuous stretch of a well inside a body becomes one prospect: the part of the body within that time window |
| Geometry and volume | Measures each prospect and tries to compute gross rock volume under several time-depth models |
| Well-log characterisation | Reads net-to-gross, porosity and water saturation from the logs crossing each prospect, tests cutoffs, computes low, middle and high cases |
| Gates | Decides per prospect what may be reported. A hydrocarbon pore volume is released only if every gate passes |

The intended result is a table of exploratory prospects with their geometry, log properties and, where the gates allow, a volume scenario.

## Use this skill when
- The user explicitly asks to run Process F, the prospect step, or stage 14.
- The user asks for prospect definitions or volumetrics and no completed Process F run exists yet.
- The user asks what Process F needs, or whether volumes can be produced from the current data.

To explain a run that has already finished, use `process_f_prospect_output_analysis`.

## Required inputs
| Input | Need | Why it matters |
|---|---|---|
| A completed Process D (geobody) job | Required | Supplies the body label volume, the catalogue and the per-trace index |
| The Process A job it descends from | Required, linked automatically | Supplies well paths, the time-depth model and the interpreted logs |
| Confirmed map coordinate units | Required for any volume | Without them gross rock volume is blocked |
| A list of anchor wells | Optional | Without one, every well that crosses a body is used |

Pass the Process D job ID to the handler as `source_geobody_job_id`, and optionally `wells` as a list of names.

## Preconditions
- The Process D job has status `succeeded`, its exports reconcile, and its label volume, catalogue and index are still in the job folder.
- The Process A job still holds its well paths, time-depth model tables and interpreted logs.
- The Process B job still holds its trace index grid.

## Instructions
1. Confirm the active project and which Process D job to build on. If there are several, ask; do not guess.
2. Call the handler with `parameters.source_geobody_job_id` and, if the user named wells, `parameters.wells`. It returns `ready`, a list of `issues`, a `forecast` and a plan. With no job ID it returns the requirements.
3. If there are blocking issues, explain each in a sentence with its `how_to_fix`. Do not submit.
4. If the handler warns `one_dominant_body`, tell the user plainly before they confirm: every prospect will be a survey-wide slice of the background, and the reference run of that kind took 4.4 hours and wrote 188 GB. Recommend fixing Process D first.
5. If `forecast.volumes_can_be_released` is false, say so: the run will report geometry and log properties, and no volume.
6. Present the plan briefly: which job, which wells, the forecast, the cost. Obtain explicit confirmation.
7. Submit only through the approved processing service, with the script name returned by the handler.
8. Return the job identifier and submission status. Offer `process_f_prospect_output_analysis` once the job has succeeded.

## How to talk to the user
- Keep it short: which job, which wells, what will and will not come out, the cost.
- Say that "prospect" here means a slice of a candidate body, not a drillable prospect.
- Present a blocked volume as the system protecting the user from a meaningless number.

## What to tell the user to expect
- The run can finish as `COMPLETED_WITH_GRV_BLOCKERS`. That is a completed run in which volumes were withheld.
- Any volume that is released is a most-optimistic scenario: the whole body filled, total porosity, uniform properties. It is not a resource estimate.
- Prospects overlap. Nothing may be added across them.
- Every result is labelled exploratory.

## Response format
The pre-flight answer is rendered by `formatting.render_markdown`: readiness line, what was checked, issues with their fixes, the inputs table, what the run will do, expected outputs and the next step.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`. The handler validates, forecasts and plans; it does not submit the job.

## Safety
Never fabricate a job ID, well names, units or job status. Never bypass backend validation or user confirmation. Never submit when the handler reports a blocking issue. Never tell the user the map units are metres unless the data owner has confirmed it. Never execute user-provided Python source. Quote the reference run time and size as observations, not promises.
