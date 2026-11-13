# Ask Geo Capabilities

## Description
Explain the currently implemented Ask Geo capabilities and give examples of supported requests based on the live skill registry and application workflows.

## Use this skill when
- The user asks what Ask Geo can do, how to use it, or what they can ask.
- The user asks for help, examples, or a list of supported Ask Geo functions.

## Instructions
1. Build the capability list from the current registered skills and the explicitly implemented frontend actions.
2. Describe only implemented capabilities. Do not imply support for unimplemented visualizations, arbitrary processing, model training, or unrestricted code execution.
3. Distinguish project-aware data operations, geoscience knowledge, processing workflows, and viewer navigation.
4. Give concise example prompts and mention that a project must be selected for project-specific operations.
5. If the implementation changes, update this catalog rather than leaving stale capability claims.

## Execution
Python handler: `handler.execute`.
