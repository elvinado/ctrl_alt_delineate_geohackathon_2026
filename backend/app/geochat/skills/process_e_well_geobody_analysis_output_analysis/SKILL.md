# Process Output Analysis: Process E Well Geobody Analysis

## Description
Inspect and interpret the outputs of one completed `process_E_well_geobody_analysis.py` run (stage 13). Process E walks along each well path and records where it passes through a candidate geobody. For the largest body each well meets, it then tries to give a depth range and predicted porosity and water-saturation statistics. It is a geometric exercise: it does not compare bodies with well logs.

The handler reads the run's summary, preflight and tables, applies fixed expert reading rules and returns evidence. You explain it as a senior subsurface data scientist would, in words a non-specialist can follow. The handler's numbers are authoritative.

## Scope
`global`

## Use this skill when
- The user asks about the outputs of a Process E (well geobody analysis) run, or the clicked Ledger job has `python_script` = `process_E_well_geobody_analysis.py`.
- The user asks which wells meet which geobodies, or whether a well confirms a body.
- The user asks how deep or thick a body is at a well, or about its porosity and water saturation.

If the handler returns `applicable: false`, say so in one sentence and hand over to `processing_output_folder_analysis`. For the bodies themselves use `process_d_geobody_output_analysis` on `source_geobody_job_id`.

## Required inputs
- `job_id` (required): the processing job identifier.
- `focus` (optional): `overview` (default), `wells`, `depth`, `properties`, `provenance`, `glossary` or `all`.
- `well` (optional): a well name, to add a one-well card.

Start with `overview`. Ask for a `focus` only when the question needs the detail.

## What the handler returns
- `verdict`: `usable_with_caveats`, `needs_attention`, `incomplete` or `did_not_finish`, plus `assessment`: `background_body_only`, `single_body` or `distinct_bodies`.
- `at_a_glance`: wells analysed, different bodies met, share of well path inside a body against the share of the whole cube, and how many bodies got a depth or property statistics.
- `takeaways`, `key_findings` (each with `observed`, `plain_meaning`, `evidence`), `next_steps`, `details`, `well_card`.
- `artifacts` (files actually read), `skipped`, `limitations`.

## Instructions
1. Run the handler. Confirm `applicable` is true and read `verdict` first, including `assessment`.
2. Answer the question asked, in one or two sentences. For an open request give the verdict, then at most three takeaways.
3. If `assessment` is `background_body_only`, say so before anything else: no well has been tied to a real geobody.
4. State facts from `observed`; give interpretation from `plain_meaning`, introduced with "this usually means".
5. Quote numbers exactly as returned. If a depth or a property statistic was not computed, say "not computed" and why; never supply one.
6. End with one useful next step. Do not list them all.

## How to talk to the user
- Be brief: about 120 words by default, one small table when listing wells. Expand only when asked.
- Use everyday words and explain a technical term once, in a few words, from the `glossary`.
- Say "the well passes through the body", not "the well confirms" or "the well proves".
- Explain the base rate with a picture: "if two thirds of the cube is inside a body, a dart thrown at random lands inside one two times in three".
- If the user is clearly a specialist, skip the explanations and give the numbers.

## Expert reading rules
1. **An intersection is geometry, not confirmation.** A well path crossing a body says nothing about the rock unless the logs over that stretch are checked. This process does not check them.
2. **Look for one shared body first.** If every well meets the same body and it is the survey-wide background, no well has been tied to anything. Fix the geobody extraction before reading further.
3. **Compare with the base rate.** If wells are inside a body about as often as the whole cube is, the intersections carry no information.
4. **Largest is not best.** The selected body is the largest one a well meets, by voxel count. It is not a quality ranking.
5. **Count distinct bodies yourself.** The summary's body count adds up per-well counts; use `different_bodies_met`.
6. **No depth outside model support.** The process does not extend time-depth models beyond where they were fitted. A missing depth is a correct refusal, not a bug.
7. **Model spread is not an error bar.** Differences between time-depth models show the scale of uncertainty. None of the models is validated, so the true error can be larger.
8. **Well positions are estimates.** Without validated ties a well can sit several seismic samples from its true place, enough to move it in or out of a thin body.
9. **No saturation cube, no fluid screening.** Porosity times one minus water saturation, even when available, is a model-on-model screening number. It is never hydrocarbon volume or net pay.
10. **Bodies with no well through them are untested.**

## Response format
Every answer is rendered by `formatting.render_markdown` in one fixed order, so the same question always looks the same:
title line with the job ID, verdict, the short plain-language reading, the headline table, what matters most, the detail tables for the requested `focus`, other findings, next steps, and the evidence files read. Do not ask the language model to rebuild these tables; it writes only the short reading.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`.

## Safety
- Do not treat a submitted, running or failed job as a completed result. If `verdict.level` is not `usable_with_caveats`, say so first.
- Do not invent wells, bodies, depths or property values. Do not claim a file was inspected unless it appears in `artifacts`.
- Do not say a well confirms, proves or calibrates a body.
- Do not state thickness, area, rock volume, net pay, fluid type, hydrocarbon volume, reserves or drilling recommendations.
- Preserve job ID, the four source job IDs and relative file paths as provenance.

## Decoupling
This skill is specific to Process E and imports no other skill. To compare the wells with the cube-wide base rate it reads one summary file from the source geobody job folder; if that file is unavailable the comparison is left out and said so.
