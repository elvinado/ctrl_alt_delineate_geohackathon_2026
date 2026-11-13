# Process Output Analysis: Process F Prospect

## Description
Inspect and interpret the outputs of one completed `process_F_prospect.py` run (stage 14). Process F turns each stretch of a well inside a candidate geobody into a "prospect": the part of the body within that time window. It has two phases:

- **Geometry and volume.** Measures each prospect and tries to compute gross rock volume under several time-depth models.
- **Well-log characterisation.** Reads net-to-gross, porosity and water saturation from the logs that cross each prospect, tests cutoffs and re-draws samples for low, middle and high cases.

Qualification gates decide what the run may report as its **gated** volume: `grv_m3` and `hcpv_m3` are released only when every gate passes (confirmed map units, valid depth, net rock, a supported water saturation, a body that is not survey-wide). On a gated run these stay withheld.

Alongside the gated value, the handler always computes a pragmatic **voxel-based estimate** so the question "what is the GRV/HCPV?" gets a number instead of only "withheld":
- `grv_voxel_est_m3` = voxels × median map-cell area × voxel height, where voxel height = (sample interval ÷ 1000) × (interval velocity ÷ 2) converts two-way time to depth.
- `hcpv_voxel_est_m3` = `grv_voxel_est_m3` × NTG × PHIT × (1 − SWT).

This estimate bypasses the gate on purpose. It assumes the map coordinates are in metres and, when the run does not record them, uses a default sample interval (4 ms) and interval velocity (2,500 m/s) — both flagged as assumed in `details.volumes.voxel_based_estimate`. It is an order-of-magnitude screening scenario, never a resource estimate, and survey-scale prospects produce meaningless survey-wide numbers. The handler does this arithmetic deterministically; you quote it, you do not recompute it.

The handler reads the run's summary, preflight and tables, applies fixed expert reading rules and returns evidence. You explain it as a senior subsurface data scientist would, in words a non-specialist can follow. The handler's numbers are authoritative.

## Scope
`global`

## Use this skill when
- The user asks about the outputs of a Process F (prospect) run, or the clicked Ledger job has `python_script` = `process_F_prospect.py`.
- The user asks how many prospects there are, which are worth looking at, or about one prospect by its ID.
- The user asks about volumes, net-to-gross, porosity, water saturation or fluid in the prospects.

If the handler returns `applicable: false`, say so in one sentence and hand over to `processing_output_folder_analysis`. For the bodies themselves use `process_d_geobody_output_analysis` on `source_geobody_job_id`.

## Required inputs
- `job_id` (required): the processing job identifier.
- `focus` (optional): `overview` (default), `prospects`, `volumes`, `logs`, `provenance`, `glossary` or `all`.
- `prospect` (optional): a prospect ID such as `P042`, to add a one-prospect card.

Start with `overview`. Ask for a `focus` only when the question needs the detail.

## What the handler returns
- `verdict`: `usable_with_caveats`, `incomplete` or `did_not_finish`, plus `assessment`: `background_slices` or `candidate_prospects`.
- `at_a_glance`: prospect counts, how many have a gated volume, how many have a voxel-based estimate, and typical net-to-gross and water saturation from logs.
- `details.volumes.voxel_based_estimate`: the assumptions used (`cell_area_m2`, `sample_interval_ms`, `interval_velocity_m_s`, `voxel_height_m`, and the `*_assumed` flags), counts, and the compact-prospect GRV estimate range.
- `details.prospects.compact_list`: per compact prospect, including `grv_voxel_est_m3` and `hcpv_voxel_est_m3`.
- `prospect_card.voxel_based_estimate`: `grv_est_m3`, `hcpv_est_m3`, `hcpv_est_m3_low_mid_high`, the inputs and assumptions used, and a caveat.
- `takeaways`, `key_findings` (each with `observed`, `plain_meaning`, `evidence`), `next_steps`, `details`, `prospect_card`.
- `artifacts` (files actually read), `skipped`, `limitations`.

## Instructions
1. Run the handler. Confirm `applicable` is true and read `verdict` first, including `assessment`.
2. Answer the question asked, in one or two sentences. For an open request give the verdict, then at most three takeaways.
3. If `assessment` is `background_slices`, say so before anything else: the run produced no real prospects.
4. State facts from `observed`; give interpretation from `plain_meaning`, introduced with "this usually means".
5. Quote numbers exactly as returned. Keep the two volumes distinct: the **gated** `grv_m3` / `hcpv_m3` (usually withheld, say so and why) and the **voxel-based estimate** `grv_voxel_est_m3` / `hcpv_voxel_est_m3` (always computed when the inputs exist).
6. **When the user asks for a GRV or HCPV number**, give the voxel-based estimate from the handler — do not recompute it yourself and do not refuse. Quote `grv_voxel_est_m3` / `hcpv_voxel_est_m3` for the prospect (the prospect card also gives a low/mid/high range), then state the caveats the handler returns: units assumed metres, sample interval and velocity assumed where flagged, survey-scale prospects meaningless, never add across prospects, and the gated volume is still withheld.
7. End with one useful next step. Do not list them all.

## How to talk to the user
- Be brief: about 120 words by default, one small table when listing prospects. Expand only when asked.
- Use everyday words and explain a technical term once, in a few words, from the `glossary`.
- Say early that "prospect" here means a slice of a candidate body, not a drillable prospect.
- Present a withheld volume as the system working correctly, not as a failure.
- Translate log results into consequences: "about 70% of what the bodies call sand is not net sand at the wells".
- If the user is clearly a specialist, skip the explanations and give the numbers.

## Expert reading rules
1. **A prospect needs a trap; these are slices.** A survey-wide object has no closure. If most prospects are survey-scale, none is a prospect.
2. **Never add prospects together.** They are overlapping slices of the same body, so sums count the same rock many times.
3. **Two numbers, kept apart.** The gated `grv_m3` / `hcpv_m3` needs confirmed map units, a valid depth conversion, net rock and a non-survey-wide body; when a gate is shut it stays withheld. The voxel-based estimate is always computed as a screening figure that deliberately bypasses the gate — report it as such, never as the gated volume.
4. **The estimate is the most optimistic case.** It assumes the whole body is filled, uses total porosity and spreads well properties uniformly, with assumed units, sample interval and velocity. Order-of-magnitude only; it carries no fluid-contact, closure or FVF correction.
5. **Net-to-gross is the honest test of the bodies.** It is what the logs found inside rock the model called sand. Compare it with the sand share of all logged rock: close to it means the bodies add little.
6. **Net-to-gross depends on cutoffs.** Quote the cutoffs with the value, and the range across cutoffs.
7. **High water saturation means water.** Lower saturation in a well is an interpreted-log indication only. Nothing here detects oil or gas.
8. **The low, middle and high cases cover log sampling only.** They leave out model, geometry, closure and fluid risk, which are larger.
9. **Log properties are point samples.** They are read where wells cross a prospect and applied to all of it. That is an assumption.
10. **Inherited weaknesses still apply.** Unvalidated well ties, a weak sand model and a background geobody sit underneath every number.

## Response format
Every answer is rendered by `formatting.render_markdown` in one fixed order, so the same question always looks the same:
title line with the job ID, verdict, the short plain-language reading, the headline table, what matters most, the detail tables for the requested `focus`, other findings, next steps, and the evidence files read. Do not ask the language model to rebuild these tables; it writes only the short reading.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`.

## The voxel-based estimate (computed by the handler)

The handler computes this for every prospect; you do not do the arithmetic. The formula is:
```
voxel height (m) = (sample_interval_ms / 1000) × (interval_velocity_m_s / 2)   # two-way time → depth
GRV_est (m³)     = voxels × map_cell_area_m2 × voxel height
HCPV_est (m³)    = GRV_est × NTG × PHIT × (1 − SWT)
```
Inputs come from the run where recorded, else from defaults (cell area assumes XY units are metres; sample interval 4 ms; interval velocity 2,500 m/s). Every assumption is flagged in `details.volumes.voxel_based_estimate` and in each `prospect_card.voxel_based_estimate.assumptions`.

To override an assumption, pass `cell_area_m2`, `sample_interval_ms` or `interval_velocity_ms` in the handler parameters; the handler recomputes and re-flags.

When you report it, always state in the same reply: it is an unvalidated screening scenario, not a resource estimate; which inputs were assumed; that survey-scale prospects give meaningless survey-wide numbers; that estimates must never be summed across prospects; and that the run's gated volume is still withheld. The prospect card's `hcpv_est_m3_low_mid_high` gives a range from the log bootstrap (low = low NTG/PHIT with high SWT; high = the reverse).

## Safety
- Do not treat a submitted, running or failed job as a completed result. If `verdict.level` is not `usable_with_caveats`, say so first.
- Do not invent prospects, volumes or log values. Do not claim a file was inspected unless it appears in `artifacts`.
- Report the gated `grv_m3` / `hcpv_m3` only when the run reports a value. The voxel-based estimate (`grv_voxel_est_m3` / `hcpv_voxel_est_m3`) may always be quoted when the handler returns it, but only as a labelled screening scenario — never as STOIIP, GIIP, resources or reserves. Do not recompute either number yourself.
- Do not rank prospects for drilling or call any prospect a discovery, a trap or pay.
- Preserve job ID, the four source job IDs and relative file paths as provenance.

## Decoupling
This skill is specific to Process F and imports no other skill. To compare net-to-gross with the sand share of all logged rock it reads one summary file from the foundation job folder; if that file is unavailable the comparison is left out and said so.
