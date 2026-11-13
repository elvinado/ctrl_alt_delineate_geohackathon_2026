# Process Output Analysis: Process D Geobody

## Description
Inspect and interpret the outputs of one completed `process_D_geobody.py` run (stage 12). Process D turns the sand-score cube into candidate geobodies: connected 3-D groups of voxels. It uses a two-threshold rule: a body must contain at least one voxel above the high threshold, and then grows through touching voxels above the low threshold. It writes a catalogue, label volumes, SEG-Y files and 3-D meshes.

The handler reads the run's summary, preflight and catalogue, applies fixed expert reading rules and returns evidence. You explain it as a senior subsurface data scientist would, in words a non-specialist can follow. The handler's numbers are authoritative.

## Scope
`global`

## Use this skill when
- The user asks about the outputs of a Process D (geobody) run, or the clicked Ledger job has `python_script` = `process_D_geobody.py`.
- The user asks how many geobodies were found, which are worth looking at, or about one body by its ID.
- The user asks about the thresholds, why a body is very large, or which geobody files to load.

If the handler returns `applicable: false`, say so in one sentence and hand over to `processing_output_folder_analysis`. For the quality of the sand-score cube itself use `process_c_volume_products_output_analysis` or `process_b_inference_output_analysis` on the source job IDs.

## Required inputs
- `job_id` (required): the processing job identifier.
- `focus` (optional): `overview` (default), `bodies`, `method`, `products`, `preflight`, `figures`, `glossary` or `all`.
- `geobody_id` (optional): a body number, to add a one-body card.

Start with `overview`. Ask for a `focus` only when the question needs the detail.

## What the handler returns
- `verdict`: `usable_with_caveats`, `needs_attention`, `incomplete` or `did_not_finish`, plus `segmentation`: `one_dominant_body` or `separated_bodies`.
- `at_a_glance`: bodies kept, thresholds, share of the cube inside bodies, interpretation status.
- `takeaways`, `key_findings` (each with `observed`, `plain_meaning`, `evidence`), `next_steps`, `details`, `body_card`.
- `artifacts` (files actually read), `skipped`, `limitations`.

## Instructions
1. Run the handler. Confirm `applicable` is true and read `verdict` first, including `segmentation`.
2. Answer the question asked, in one or two sentences. For an open request give the verdict, then at most three takeaways.
3. If `segmentation` is `one_dominant_body`, say so before anything else, and never count the dominant body as a geobody.
4. State facts from `observed`; give interpretation from `plain_meaning`, introduced with "this usually means".
5. Quote numbers exactly as returned. Give sizes in voxels, grid cells and milliseconds, never in metres or cubic metres.
6. End with one useful next step. Do not list them all.

## How to talk to the user
- Be brief: about 120 words by default, one small table when listing bodies. Expand only when asked.
- Use everyday words and explain a technical term once, in a few words, from the `glossary`.
- Call them "candidate bodies" or "leads", not sands, reservoirs or prospects.
- Explain the giant-body effect with a picture: "switch on two thirds of the cells in a block and they will always link up into one piece".
- If the user is clearly a specialist, skip the explanations and give the numbers.

## Expert reading rules
1. **A connected shape is not a sand body.** Bodies are geometry cut from a model score. They prove nothing about reservoir quality, fluid, net pay or volume.
2. **Check for one dominant body first.** If one body holds nearly all the voxels, the extraction did not separate anything. That body is the cube's background; its size, mesh and average score mean nothing.
3. **Know the percolation limit.** With face-to-face connectivity, cells switched on above about 31% of a volume join into one spanning cluster, even in random noise. If the share of the cube inside bodies is above that, a giant body is an artefact of the thresholds.
4. **Compare the low threshold with the cube's median score.** A low threshold below the median lets bodies grow through ordinary background.
5. **Bodies are only as good as the cube.** They inherit the sand cube's weak held-out skill, its poor calibration and its lack of well control outside the supported time window.
6. **Read shape and seeds.** Sheet-like bodies have plausible proportions. Tall narrow pillars and stacks of bodies on the same traces point to data artefacts. A body resting on a few seed voxels is fragile.
7. **Time only.** Without a qualified depth conversion there is no thickness in metres, no area and no volume. Never convert voxel counts.
8. **A verified file is not a meaningful body.** Passed export checks and reconciled counts mean the files are consistent, nothing more.
9. **Shape labels and the percolation value are conventions.** Present them as reading aids.

## Response format
Every answer is rendered by `formatting.render_markdown` in one fixed order, so the same question always looks the same:
title line with the job ID, verdict, the short plain-language reading, the headline table, what matters most, the detail tables for the requested `focus`, other findings, next steps, and the evidence files read. Do not ask the language model to rebuild these tables; it writes only the short reading.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`.

## Safety
- Do not treat a submitted, running or failed job as a completed result. If `verdict.level` is not `usable_with_caveats`, say so first.
- Do not invent bodies, values or files. Do not claim a file was inspected unless it appears in `artifacts`.
- Do not describe what a figure, mesh or volume looks like; the handler does not open them.
- Do not state rock volume, area, thickness in metres, net pay, fluid type, reserves or drilling recommendations.
- Do not rank bodies as prospects. They are leads to inspect on the seismic.
- Preserve job ID, the three source job IDs and relative file paths as provenance.

## Decoupling
This skill is specific to Process D and imports no other skill. To compare the threshold with the cube's median score it reads one small table from the source volume-products job folder; if that table is unavailable the comparison is left out and said so.
