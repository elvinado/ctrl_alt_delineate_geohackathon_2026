# Process: Process D Geobody

## Description
Run `process_D_geobody.py` (stage 12). It turns the sand-score cube into candidate geobodies: connected 3-D groups of voxels. It builds on a completed Process C (volume products) run and trains nothing.

| Step | What it does |
|---|---|
| Preflight | Confirms the three source jobs link up and the cube matches its checksum |
| Segmentation | Keeps voxels above a low threshold, joins touching ones, and keeps groups that contain a voxel above a high threshold and meet a minimum size |
| Exports | Body catalogue, label volumes, two SEG-Y files (mask and body number), 3-D meshes and figures |

The intended result is a list of candidate bodies to inspect on the seismic. They are exploratory shapes, not interpreted reservoirs.

## Use this skill when
- The user explicitly asks to run Process D, the geobody extraction, or stage 12.
- The user asks what Process D needs, or whether a Process C run is ready for it.
- The user asks what would happen at a different low threshold.

To explain a run that has already finished, use `process_d_geobody_output_analysis`.

## Required inputs
| Input | Need | Why it matters |
|---|---|---|
| A completed Process C (volume products) job | Required | Registers the sand-score cube and its validity mask with verified checksums |
| The Process B job it packaged, with its cube files | Required, linked automatically | The cube itself stays in the Process B job |
| Free disk for label volumes, two SEG-Y files and an index | Required | Several times the size of one seismic volume |

Pass the Process C job ID to the handler as `source_volume_job_id`. Optionally pass `low_threshold` (between 0 and 1) to forecast the outcome at that setting. The thresholds themselves are set in the process configuration, not through this skill.

## Preconditions
- The Process C job has status `succeeded` and its export stage (11) is `completed`.
- The sand-score cube and validity mask are registered, passed their checksum, and still exist in the Process B job.
- The Process A and Process B jobs it was built on still exist, because the preflight verifies the links.

## Instructions
1. Confirm the active project and which Process C job to segment. If there are several, ask; do not guess.
2. Call the handler with `parameters.source_volume_job_id`. It returns `ready`, a list of `issues`, a `forecast` and a plan. With no job ID it returns the requirements.
3. If there are blocking issues, explain each in a sentence with its `how_to_fix`. Do not submit.
4. If the handler warns `giant_body_expected`, tell the user plainly before they confirm: at this setting the run will produce one survey-wide body and little else. Give the suggested threshold from `how_to_fix`.
5. Present the plan briefly: which job, what the steps do, the disk space, and what comes out. Obtain explicit confirmation.
6. Submit only through the approved processing service, with the script name returned by the handler.
7. Return the job identifier and submission status. Offer `process_d_geobody_output_analysis` once the job has succeeded.

## How to talk to the user
- Keep it short: which job, what happens, the forecast, how much disk.
- Explain the giant-body risk with a picture: "switch on two thirds of the cells in a block and they will always link into one piece".
- Call the results "candidate bodies" or "leads", not sands, reservoirs or prospects.

## What to tell the user to expect
- The preflight usually ends `CONDITIONAL`: the sand score is not a calibrated probability and there is no qualified depth conversion. The run proceeds and labels every body exploratory.
- Sizes come in voxels, grid cells and milliseconds. There is no thickness in metres, no area and no volume.
- The bodies inherit every weakness of the sand cube they are cut from.

## Response format
The pre-flight answer is rendered by `formatting.render_markdown`: readiness line, what was checked, issues with their fixes, the inputs table, what the run will do, expected outputs and the next step.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`. The handler validates, forecasts and plans; it does not submit the job and does not change the thresholds.

## Safety
Never fabricate a job ID, threshold values, or job status. Never bypass backend validation or user confirmation. Never submit when the handler reports a blocking issue, and never segment a cube whose checksum failed. Never execute user-provided Python source. Present the forecast as an estimate from the cube's percentiles, and the reference run time as an observation, not a promise.
