# Data Inventory

## Description
Retrieve the authoritative inventory of datasets registered in the active project.

## Use this skill when
- The user asks what datasets, files, well data, or data associated with wells/surveys are available.
- The user asks which wells or surveys have a particular data type.
- The user asks what data is associated with a named well (for example, "what data do I have for M-1?").
- The user asks whether a dataset is registered or its source file is accessible.
- Treat broad phrases such as "well data", "well datasets", and "well files" as all well-associated types: well logs, deviation surveys, and checkshots. Do not silently narrow these phrases to LAS/well logs only.
- Do not use this skill merely to list entity names. "What well(s) do I have?" and "what survey(s) do I have?" belong to `well_survey_inventory`.

## Inputs
- `project_id` is supplied by the application and is required.
- `data_type` is an optional dataset kind or list of kinds: `seismic`, `well_log`, `deviation`, or `checkshot`.
- `well_name` and `survey_name` are optional exact-name filters.

## Instructions
1. Require an active project. If none is selected, ask the user to select one; never attempt an unscoped inventory query.
2. Interpret natural-language filters before calling the handler. Normalize aliases to canonical kinds and support multiple kinds in one `data_type` list.
3. For broad "well data" requests, filter to `well_log`, `deviation`, and `checkshot`; for a specific well, also pass `well_name`.
4. Call the `data_inventory` handler with the supplied project context and filters.
5. Treat its result as the source of truth. Do not infer that unlisted datasets exist. A successful empty query must produce the explicit zero-result response, never an empty assistant message.
6. Build each description from the parsed file header: SEG-Y textual/binary header and trace dimensions; LAS well/index/curve metadata; or parsed deviation/checkshot columns and row count. If the file is unavailable, parsing fails, or the header provides no useful details, use the concise type-specific default description.
7. Render every inventory response using `formatting.render_markdown`; do not ask the LLM to reformat the records.
8. Use the same table columns and order for full inventories and filtered results. Keep the original filename visible in the Dataset column, wrapped in inline code; never replace it with a viewer label or action.
9. Report the total row count. For an empty result, state that no matching datasets are registered and report a total of zero.

## Response format
Use one compact Markdown table with these columns, in this order:

| Dataset | Type | Well / Survey | Description |
|---|---|---|---|

The filename must remain visible as the first cell. The frontend may add a viewer action beside it; do not replace or rewrite the filename.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.

## Terminology
Consult `../SYNONYMS.md` when interpreting dataset names, aliases, and availability language.
