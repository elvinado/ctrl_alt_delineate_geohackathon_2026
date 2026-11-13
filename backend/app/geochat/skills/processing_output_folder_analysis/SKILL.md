# Processing Output Folder Analysis

## Description
Inspect every safe, regular file in a processing run output folder and produce a grounded inventory and an LLM-written plain-language explanation of its contents. The LLM narrative is explanatory; deterministic inspection remains authoritative.

## Scope
`global`

## Instructions
- Treat the complete processing job folder as the unit of analysis, not only the file clicked in the Ledger.
- Read `job.json` and `result.json` as provenance metadata; do not treat them as scientific measurements.
- Summarize supported tabular, JSON, and text outputs using deterministic file inspection. Report file names, formats, sizes, row/column counts, fields, numeric ranges, and relevant categorical counts where available.
- Identify visualizations and opaque/binary files without claiming to interpret their contents.
- Use inspected facts to describe the likely processing purpose and the role of important outputs. If purpose is unclear, state that uncertainty rather than guessing. Keep the LLM explanation grounded in deterministic findings.
- Distinguish observed facts from possible meaning. Do not infer scientific conclusions unsupported by the files.
- State when files are skipped because they are too large, unreadable, or unsupported. Never follow symlinks or paths outside the job folder.
- Preserve job ID, script, target well, status, and relative artifact paths as provenance.


## Decoupling
This generic folder-inspection skill must remain process-agnostic. Do not import or call any process-specific output-analysis handler. Specialized interpretation belongs in a separately registered `process_*_output_analysis` skill.
