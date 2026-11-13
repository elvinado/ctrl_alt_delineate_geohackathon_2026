# Processing Job History

## Description
List completed processing jobs across the workspace, including their Python worker, job ID, status, summary, and generated outputs.

## Scope
`global`

## Instructions
Use this skill when the user asks to list, show, or review completed processing jobs or processing history. It is workspace-wide because processing jobs are recorded in a shared processing-job registry. Return the authoritative job records and format them as a Markdown table with columns: Python file, ID, Status, Description or summary, and Outputs/results. Do not invent missing descriptions or outputs; use a clear unavailable marker.
