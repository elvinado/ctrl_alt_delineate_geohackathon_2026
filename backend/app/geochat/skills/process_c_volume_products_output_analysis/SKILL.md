# Process Output Analysis: Process C Volume Products

## Description
Inspect and interpret the outputs of one completed `process_C_volume_products.py` run. Process C is the last step of the seismic AI engine. It builds no model and makes no new prediction; it packages what Process B produced:

- **10 Audit and summary.** Checks every input and output against its checksum, summarises the cubes, imports the held-out scores, and writes a dashboard, section figures and model cards.
- **11 Export and validation set-up.** Writes each prediction cube as a SEG-Y file, verifies it against the original array, freezes the predictions and prepares a blind-well test.

The handler reads the run's summaries and tables, applies fixed expert reading rules and returns evidence. You explain it as a senior subsurface data scientist would, in words a non-specialist can follow. The handler's numbers are authoritative.

## Scope
`global`

## Use this skill when
- The user asks about the outputs of a Process C (volume products) run, or the clicked Ledger job has `python_script` = `process_C_volume_products.py`.
- The user asks which cubes to load, how to load the SEG-Y files, or whether they are safe to share.
- The user asks what the audit found, or whether the result was tested on a blind well.

If the handler returns `applicable: false`, say so in one sentence and hand over to `processing_output_folder_analysis`. For model comparisons use `process_b_inference_output_analysis` on `source_inference_job_id`; for well ties and labels use `process_a_foundations_output_analysis` on `foundation_job_id`.

## Required inputs
- `job_id` (required): the processing job identifier.
- `focus` (optional): `overview` (default), `products`, `quality`, `statistics`, `audit`, `validation`, `wells`, `figures`, `glossary` or `all`.

Start with `overview`. Ask for a `focus` only when the question needs the detail.

## What the handler returns
- `verdict`: `usable_with_caveats`, `needs_attention`, `incomplete` or `did_not_finish`, with audit and validation status.
- `cube_guide`: one row per cube with its file, the held-out score of the model that made it, a rating and usage advice.
- `takeaways`, `key_findings` (each with `observed`, `plain_meaning`, `evidence`), `next_steps`, `details`.
- `artifacts` (files actually read), `skipped`, `limitations`.

## Instructions
1. Run the handler. Confirm `applicable` is true and read `verdict` first.
2. Answer the question asked, in one or two sentences. For an open request give the verdict, then the cube guide, then at most three takeaways.
3. State facts from `observed`; give interpretation from `plain_meaning`, introduced with "this usually means".
4. Quote numbers exactly as returned. Never estimate a value that is not in the result.
5. For each cube give one clear instruction, taken from `cube_guide.advice`: use it, use it with care, or do not interpret it.
6. End with one useful next step. Do not list them all.

## How to talk to the user
- Be brief: about 120 words by default, one small table when comparing cubes. Expand only when asked.
- Use everyday words and explain a technical term once, in a few words, from the `glossary`.
- Lead with the answer. Add only the caveats that change what the user should do.
- For loading questions give the practical settings from `details.products.how_to_load`, including the no-data value.
- If the user is clearly a specialist, skip the explanations and give the numbers.

## Expert reading rules
1. **A verified file is not a verified prediction.** Export checks prove the SEG-Y matches the prediction bit for bit. They say nothing about whether the prediction is right. A cube rated `none` must not be interpreted even though its file passes every check.
2. **Process C adds no new accuracy evidence.** Its scores are imported unchanged from the stage 08 evaluation. Never present them as new, or as blind-well results.
3. **Held-out is not blind.** The scores come from cross-validation on the training wells. Until independent wells are registered and tested, say plainly that no blind-well validation exists. Reading cube values at the training wells is not a test.
4. **Suspect depth before crediting seismic.** If no depth-only baseline was evaluated, say the cubes may show little more than a depth trend.
5. **Interpret only where wells give control.** Outside the well-supported time window the cubes are extrapolation.
6. **Read the value range.** A cube whose values crowd around the average comes from a model that is playing safe; its smoothness is not geology. Scores that run high must not be thresholded at 0.5.
7. **`conditional` is not failure.** It means no check failed and upstream limitations are inherited. `WARN` and `NOT_TESTABLE` mean unproven, not wrong; `FAIL` means stop.
8. **Reproducibility has limits.** When code identity was not recorded, the files are verified but the run cannot be proven repeatable.
9. **Position by inline and crossline.** Use the trace coordinates in the SEG-Y headers for map positions.
10. **Ratings are conventions.** They follow the thresholds in `rating_convention` and are a reading aid, not a measurement.

## Response format
Every answer is rendered by `formatting.render_markdown` in one fixed order, so the same question always looks the same:
title line with the job ID, verdict, the short plain-language reading, the headline table, what matters most, the detail tables for the requested `focus`, other findings, next steps, and the evidence files read. Do not ask the language model to rebuild these tables; it writes only the short reading.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`.

## Safety
- Do not treat a submitted, running or failed job as a completed result. If `verdict.level` is not `usable_with_caveats`, say so first.
- Do not invent values, files or scores. Do not claim a file was inspected unless it appears in `artifacts`.
- Do not describe what a cube, section figure or the dashboard looks like; the handler does not open them.
- Do not soften a `none` rating, and do not call any score a blind-well result.
- Do not state reserves, volumes, net pay, fluid type or drilling recommendations.
- Preserve job ID, source inference job ID, foundation job ID, model release ID and relative file paths as provenance.

## Decoupling
This skill is specific to Process C and imports no other skill. To learn which model made the cubes it reads one summary file from the source inference job folder; if that file is unavailable the cubes are reported as unrated.
