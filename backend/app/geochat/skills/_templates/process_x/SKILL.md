# Process: <PROCESS_NAME>

## Description
Describe the processing operation and its intended result.

## Use this skill when
- The user explicitly requests this processing operation.
- The user asks to configure or prepare this operation for execution.

## Required inputs
- List required project datasets and their roles.
- List required processing parameters and valid ranges.

## Preconditions
- Identify dataset format, association, and availability requirements.
- Identify any prerequisite processing steps.

## Instructions
1. Confirm the active project and identify the requested inputs.
2. Validate prerequisites and parameter values using the handler.
3. Ask the user to resolve missing or ambiguous inputs.
4. Present the proposed configuration and obtain confirmation before execution.
5. Submit work only through the approved processing service.
6. Return the job identifier and submission status.

## Execution
Python handler: `handler.execute`.
Markdown renderer: `formatting.render_markdown`.
Request parser: `intent.parse_request`.

## Safety
Never fabricate input datasets, parameter values, or job status. Never bypass backend validation or user confirmation.
