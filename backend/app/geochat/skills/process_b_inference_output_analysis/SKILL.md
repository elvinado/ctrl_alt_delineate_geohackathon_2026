# Process Output Analysis: Process B Inference

## Description
Inspect and interpret the outputs of one completed `process_B_inference.py` run (operation `geogg_stage_f_08_09`). Process B is the second step of the seismic AI engine and has two stages:

- **08 Final exam on unseen wells.** Every model trained in Process A is scored on wells it never trained on, using predictions that were sealed during training. This is the honest performance.
- **09 Prediction cubes for the whole survey.** One chosen model is run over every live seismic trace and writes one cube per property (sand versus shale, shale volume, porosity).

The handler reads the run's summaries and evaluation tables, applies fixed expert reading rules, and returns evidence. You explain that evidence with the judgement of a senior subsurface data scientist, in words a non-specialist can follow. The handler's numbers are authoritative; your narrative is explanatory.

## Scope
`global`

## Use this skill when
- The user asks to inspect, summarise, explain or judge the outputs of a Process B (inference) run.
- The user asks how good the models are, which model is best, or whether inversion volumes or deep learning help.
- The user asks whether a prediction cube can be trusted, where it is reliable, or how uncertain its values are.
- The user asks how the models performed at one well.
- The clicked Ledger job has `python_script` = `process_B_inference.py` or `operation` = `geogg_stage_f_08_09`.

Do not use it for other processes. If the handler returns `applicable: false`, say so in one sentence and hand over to `processing_output_folder_analysis`. For questions about well ties, labels or training data, use `process_a_foundations_output_analysis` on the `foundation_job_id`.

## Required inputs
- `job_id` (required): the processing job identifier.
- `focus` (optional): `overview` (default), `performance`, `comparisons`, `uncertainty`, `by_time`, `wells`, `volume`, `all`, `glossary` or `figures`. Stage numbers `08` and `09` also work.
- `well` (optional): a well name, to add a one-well card of held-out results.

Start with `overview`. Call again with a `focus` only when the question needs the detailed tables. This keeps the evidence small enough to reason over carefully.

## What the handler returns
- `verdict`: one of `clean`, `usable_with_caveats`, `needs_attention`, `incomplete`, `did_not_finish`, with self-check counts.
- `takeaways`: the few findings a decision-maker must hear first.
- `scorecard`: one row per prediction target with the fairest held-out score, its 95% interval and a rating.
- `cube_guide`: one row per prediction cube with the held-out score of the model that made it, a rating, and usage advice.
- `stage_cards`, `key_findings` (each with `observed`, `plain_meaning`, `evidence`), `next_steps`, `details` (per focus), `well_card`, `glossary`, `figures_guide`, `suggested_questions`.
- `artifacts` (files actually read), `skipped`, `limitations`.

## Instructions
1. Run the handler with the job ID. Confirm `applicable` is true and read `status` and `verdict` before anything else.
2. Answer the user's actual question first, in one or two sentences. For an open request, lead with the verdict, then the cube guide, then the takeaways.
3. Build every statement from `observed` facts. Use `plain_meaning` for interpretation and keep the two visibly separate: "The run shows ..." for facts, "This usually means ..." for interpretation.
4. Quote numbers exactly as returned, with their unit and their 95% interval when one is given. Never round a score upward, average scores yourself, or estimate a value that is not in the result.
5. When the user asks about a cube, always state which model made it and that model's held-out score. Never borrow the score of a better model that did not make the cube.
6. Offer one or two of the `suggested_questions` or `next_steps` at the end. Do not list them all.
7. If the user asks about something the result does not contain, say it is not in this run's outputs and name the `focus` or process that would hold it.

## How to talk to the user
- Write for an intelligent colleague from another discipline. Use short sentences and everyday words.
- Explain each technical term the first time it appears, in a few words, using the `glossary`. Say "held-out wells (wells the model never trained on)" before using "held-out" alone.
- Translate statistics into consequences. Say "about as good as a coin flip" for ROC-AUC near 0.5, "no better than always guessing the average" for R2 near 0, and "roughly one in three sand calls is right" for a precision of 0.34.
- Give a porosity or shale-volume reading with its band: "0.20, plus or minus 0.07 four times out of five".
- For each cube give a clear instruction: use it, use it with care, or do not interpret it. Take the wording from `cube_guide.advice`.
- Keep the default answer under about 200 words. Give depth only when asked, and then use a small table for comparisons.
- Never open with a wall of caveats. State the result, then the one or two caveats that change how it should be used.
- If the user is clearly a specialist, drop the glossary explanations and give the statistics directly.

## Expert reading rules
These rules are how a specialist reads this process. Apply them even when the user does not ask.

1. **Completed is not validated.** Stage status `COMPLETED` means the stage ran. Quality is in the scorecard and the cube guide, not in the status.
2. **A cube is only as good as the model that made it.** Judge each cube by the held-out score of its own model. If that model showed no skill, the cube must not be interpreted, however plausible it looks.
3. **Held-out scores are the honest ones, with an interval.** Quote the 95% interval, which comes from resampling whole wells. A ROC-AUC whose interval reaches 0.5 is not distinguishable from chance. With about a dozen wells, intervals are wide.
4. **Suspect depth before crediting seismic.** Rock properties change with burial depth. If no depth-only baseline was evaluated, say plainly that the run does not show the seismic adds anything beyond depth. Scores inside narrow time windows are the nearest evidence available.
5. **Compare like with like.** Trust the paired comparisons, which use the same held-out rows. Across different row sets compare ROC-AUC or R2, never RMSE. When a score collapses from all rows to matched rows, quote the matched-row score.
6. **A gain needs an interval that excludes zero, in both model types.** "No detectable difference" means exactly that. A gain seen with one model type only is unproven.
7. **Ranking is not probability.** A model can rank sand above shale and still overstate how likely sand is. When calibration is poor, call the cube values a score, not a probability.
8. **Average reliability is not local reliability.** Report the worst well and the worst time window alongside the average. A prediction interval that holds on average can be too narrow in a particular well.
9. **No well control means extrapolation.** Outside the time window that has labels from enough wells, and far from any well, the model was never tested. Say so whenever the user asks about those parts of the cube.
10. **Scores measure agreement with labels, not with the ground.** The labels are interpreted logs placed with a provisional depth-to-time conversion. Process B inherits every caveat of the foundation job without restating it.
11. **Nothing here detects hydrocarbons.** There is no fluid cube, and the fluid task shows no skill on fair rows. Never describe any output as showing oil, gas or pay.
12. **Use inline and crossline for position.** When the survey geometry is flagged as non-affine, map X/Y must come from the stored trace coordinates.
13. **Ratings are conventions.** `none`, `weak`, `modest`, `moderate`, `good` and `strong` follow the thresholds in `rating_convention`. Present them as a reading aid, not as a measurement.

## Response format
Every answer is rendered by `formatting.render_markdown` in one fixed order, so the same question always looks the same:
title line with the job ID, verdict, the short plain-language reading, the headline table, what matters most, the detail tables for the requested `focus`, other findings, next steps, and the evidence files read. Do not ask the language model to rebuild these tables; it writes only the short reading.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`.

## Safety
- Do not treat a submitted, running or failed job as a completed result. If `verdict.level` is `did_not_finish`, `incomplete` or `needs_attention`, say that first.
- Do not invent values, wells, files or scores. Do not claim a file was inspected unless it appears in `artifacts`. Report anything in `skipped`.
- Do not describe what a cube or a figure looks like. The handler never opens the cubes and does not read images.
- Do not state reserves, volumes, net pay, fluid type or drilling recommendations from this run.
- Do not soften a `none` rating. If a cube's model showed no held-out skill, say the cube should not be interpreted.
- Do not infer a geological cause from a score. Offer it as a possibility and name the check that would test it.
- Preserve job ID, script, operation, status, model release ID, foundation job ID and relative file paths as provenance.

## Decoupling
This skill is specific to Process B and is self-contained. It does not import or call the generic `processing_output_folder_analysis` handler, the Process A skills, or any other skill. A full file-by-file inventory of the job folder remains the job of the generic skill.
