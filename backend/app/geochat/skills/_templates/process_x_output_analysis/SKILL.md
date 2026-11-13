# Process Output Analysis: <PROCESS_NAME>

## Description
Inspect and interpret outputs produced by the corresponding `<PROCESS_NAME>` operation.

## Use this skill when
- The user asks to inspect, summarise, compare, or interpret outputs from this process.
- The user asks about the result or quality of a completed processing job.

## Required inputs
- Processing job identifier, or enough context to identify the relevant completed job.
- Output dataset or output-file references returned by the backend.

## Instructions
1. Identify the relevant processing job and verify its completion status.
2. Retrieve output metadata and accessible output files through approved backend services.
3. Inspect only supported output formats and report any limitations.
4. Separate measured observations from interpretation and uncertainty.
5. Summarise relevant quality indicators, anomalies, and recommended follow-up checks.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`.

## Safety
Do not treat a submitted or failed job as a completed result. Do not invent output values or claim that files were inspected when they were inaccessible.
