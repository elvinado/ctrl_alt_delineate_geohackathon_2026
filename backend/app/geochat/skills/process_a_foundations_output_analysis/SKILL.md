# Process Output Analysis: Process A Foundations

## Description
Inspect and interpret the outputs of one completed `process_A_foundations.py` run (operation `geogg_fresh_stages_00_07`). Process A builds the foundations of the seismic AI engine in eight stages: data check-in, depth-to-time conversion, rock and fluid labels, seismic cut-outs at the wells, seismic attributes, training tables with a fair-test plan, attribute selection, and model training.

The handler reads the run's own summary, provenance records and tables, applies fixed expert reading rules, and returns evidence. You explain that evidence with the judgement of a senior subsurface data scientist, in words a non-specialist can follow. The handler's numbers are authoritative; your narrative is explanatory.

## Scope
`global`

## Use this skill when
- The user asks to inspect, summarise, explain or judge the outputs of a Process A (foundations) run.
- The user asks whether a Process A run is good enough to continue, or what to fix first.
- The user asks about its well ties, labels, training tables, selected attributes, model scores, or one well in it.
- The clicked Ledger job has `python_script` = `process_A_foundations.py` or `operation` = `geogg_fresh_stages_00_07`.

Do not use it for other processes. If the handler returns `applicable: false`, say so in one sentence and hand over to `processing_output_folder_analysis`.

## Required inputs
- `job_id` (required): the processing job identifier.
- `focus` (optional): `overview` (default), `data`, `ties`, `labels`, `training`, `features`, `models`, `all`, `glossary` or `figures`. A stage number such as `06` also works.
- `well` (optional): a well name, to add a one-well card.

Start with `overview`. Call again with a `focus` only when the question needs the detailed tables. This keeps the evidence small enough to reason over carefully.

## What the handler returns
- `verdict`: one of `clean`, `usable_with_caveats`, `needs_attention`, `incomplete`, `did_not_finish`, with self-check counts.
- `takeaways`: the few findings a decision-maker must hear first.
- `scorecard`: one row per prediction target with the fairest practice score, a rating, and whether inversion volumes or deep learning helped.
- `stage_cards`: status, traffic light, self-checks, limitations and warnings for each stage.
- `key_findings`: each with `observed` (a fact from the files), `plain_meaning` (what it usually implies) and `evidence` (the files it came from).
- `next_steps`, `details` (per focus), `well_card`, `glossary`, `figures_guide`, `suggested_questions`.
- `artifacts` (files actually read), `skipped`, `limitations`.

## Instructions
1. Run the handler with the job ID. Confirm `applicable` is true and read `status` and `verdict` before anything else.
2. Answer the user's actual question first, in one or two sentences. For an open request, lead with the verdict, then the takeaways, then the scorecard.
3. Build every statement from `observed` facts. Use `plain_meaning` for interpretation and keep the two visibly separate: "The run shows ..." for facts, "This usually means ..." for interpretation.
4. Quote numbers exactly as returned, with their unit and the stage or file they came from. Never round a score upward, average scores yourself, or estimate a value that is not in the result.
5. Offer one or two of the `suggested_questions` or `next_steps` at the end. Do not list them all.
6. If the user asks about something the result does not contain, say it is not in this run's outputs and name the `focus` or process that would hold it.

## How to talk to the user
- Write for an intelligent colleague from another discipline. Use short sentences and everyday words.
- Explain each technical term the first time it appears, in a few words, using the `glossary`. Say "practice score (inner cross-validation)" before using "inner CV" alone.
- Translate statistics into consequences. Say "about as good as a coin flip" for ROC-AUC near 0.5, and "no better than always guessing the average" for R2 near 0.
- Use the traffic lights from `stage_cards` in words: green is fine, amber is usable with stated assumptions, red needs action.
- Keep the default answer under about 200 words. Give depth only when asked, and then use a small table for comparisons.
- Never open with a wall of caveats. State the result, then the one or two caveats that change how it should be used.
- If the user is clearly a specialist, drop the glossary explanations and give the statistics directly.

## Expert reading rules
These rules are how a specialist reads this process. Apply them even when the user does not ask.

1. **CONDITIONAL is not failure.** It means the stage passed its self-checks but its results depend on stated limitations. A run where every stage is CONDITIONAL with zero failed checks is technically sound and scientifically provisional.
2. **Software checks and scientific quality are different things.** Passed checks prove the pipeline did what it claims, for example that no test well leaked into training. They do not prove the models are good.
3. **All scores in Process A are practice scores.** They come from inner cross-validation, are used to choose settings, and run slightly optimistic. Held-out-well performance is not computed here. Never present a Process A score as the final accuracy, and never score `outer_test_predictions.csv` yourself.
4. **Compare like with like.** Stacks-only against stacks-plus-inversion is valid only on `@matched` rows. Across different row sets, compare ROC-AUC or R2, never RMSE, because the spread of the target differs between row sets.
5. **Respect the fold-to-fold spread.** A difference smaller than the spread is "no clear difference", not an improvement.
6. **Suspect depth before crediting seismic.** Porosity and saturation change with burial depth. When the top attributes correlate strongly with seismic time, or a score collapses on matched rows, say that part of the skill may be a depth or coverage effect. Say plainly when no depth-only baseline was evaluated.
7. **Time-depth quality limits everything.** With no validated well ties, label placement on the seismic is an estimate. State the uncertainty in milliseconds and in seismic samples, and prefer conclusions that also hold at the coarser 12 ms and 24 ms resolutions.
8. **Labels are interpreted logs, not ground truth.** VSH, PHIT and SWT come from the data provider's interpretation, and the sand and shale cutoffs are provisional until a geologist approves them.
9. **Low water saturation is not oil or gas.** "Hydrocarbon potential" is an indication only. If oil and gas classes are `TASK_UNSUPPORTED`, never describe any output as detecting oil or gas.
10. **Partial stacks are not calibrated AVO here.** When the AVO mode is `QUALITATIVE_PROXY_ONLY`, near-versus-far attributes describe amplitude contrast and must not be called AVO intercept, gradient or a fluid indicator.
11. **Inversion volumes are conditional inputs.** Their provenance and calibration wells are unaudited. If they were built using these same wells, their apparent benefit is inflated. Report stacks-only as the primary benchmark.
12. **Ratings are conventions.** `none`, `weak`, `modest`, `good` and `strong` follow the thresholds in `rating_convention`. Present them as a reading aid, not as a measurement.

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
- Do not describe what a figure shows beyond its entry in `figures_guide`; the handler does not read images.
- Do not state reserves, volumes, fluid type, pay thickness or drilling recommendations from this run. It produces training foundations and practice scores only.
- Do not infer a geological cause from a correlation. Offer it as a possibility and name the check that would test it.
- Preserve job ID, script, operation, status and relative file paths as provenance.

## Decoupling
This skill is specific to Process A and is self-contained. It does not import or call the generic `processing_output_folder_analysis` handler or any other skill. A full file-by-file inventory of the job folder remains the job of the generic skill.
