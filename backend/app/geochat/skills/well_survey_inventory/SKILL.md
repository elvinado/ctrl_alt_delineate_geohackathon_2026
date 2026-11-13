# Well and Survey Inventory

## Description
List well and survey entities registered in the active project, independently of dataset inventory.

## Use this skill when
- The user asks what wells or boreholes are in the project (for example, "what well do I have?" or "list the wells").
- The user asks what surveys are in the project (for example, "what survey do I have?" or "list available surveys").
- The user asks which wells or surveys are registered or available.

Do not use this skill for questions about well data, well logs, seismic data, files, or datasets. Those belong to `data_inventory`.

## Inputs
- `project_id` is supplied by the application and is required.
- `entity_type` must be `wells` or `surveys`.

## Instructions
1. Require an active project.
2. Query only the requested entity table, scoped to the active project.
3. Return all matching records, sorted by name.
4. Render the result using `formatting.render_markdown`; do not ask the LLM to reformat it.
5. If no records exist, explicitly state that none are registered and report a total of zero.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
