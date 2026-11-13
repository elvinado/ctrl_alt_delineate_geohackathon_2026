# GeoChat Domain Synonyms and Canonical Terms

Use these mappings when interpreting user messages and preparing skill parameters. Match terms by meaning and context, not capitalization, spacing, punctuation, singular/plural form, or underscores.

## Dataset types

| Canonical dataset kind | Common terms and aliases |
|---|---|
| `seismic` | seismic, seismic data, seismic volume, seismic cube, 3D seismic, 2D seismic, SEG-Y, SEGY, SGY, seismic file, seismic survey data |
| `well_log` | `well_log`, well log, well logs, well-log, well logging, wireline log, wireline logs, log data, LAS, LAS file, LAS files, petrophysical logs |
| `checkshot` | checkshot, check shot, check-shot, checkshots, check shot survey, checkshot data, time-depth data, time depth data, T-D survey |
| `deviation` | deviation, well deviation, deviation survey, directional survey, directional data, borehole trajectory, well trajectory, inclination/azimuth survey |

Canonical values must be used when calling dataset inventory filters: `seismic`, `well_log`, `checkshot`, and `deviation`.

Do not classify every TXT file as deviation or checkshot based only on its extension. Use its registered dataset kind or surrounding context.

## Geological objects and associations

| Canonical concept | Common terms and aliases |
|---|---|
| `well` | well, wells, borehole, boreholes, wellbore, wellbores |
| `survey` | survey, surveys, seismic survey, acquisition survey, seismic project area (when referring to a registered survey) |
| `dataset` | dataset, data set, data, file, files, input data, source data, uploaded data |
| `project` | project, workspace, study, current project, active project |

A well and a survey are different objects. Well logs, checkshot, and deviation data are normally associated with wells; seismic datasets are normally associated with surveys. Broad phrases such as "well data", "well datasets", or "well files" mean all three well-associated dataset kinds (`well_log`, `deviation`, and `checkshot`), unless the user names one specific type.

## Processing concepts

| Concept | Common terms and aliases |
|---|---|
| `inversion` | inversion, seismic inversion, acoustic impedance inversion, AI inversion, impedance inversion |
| `migration` | migration, seismic migration, imaging, migrated image, post-stack migration, pre-stack migration |
| `deconvolution` | deconvolution, decon, seismic deconvolution |
| `velocity_analysis` | velocity analysis, velocity picking, velocity model analysis, semblance analysis |
| `processing_output` | output, outputs, result, results, processing result, processing output, generated files, job output, run output |
| `processing_job` | job, processing job, run, execution, task, submitted process |

Processing terminology does not by itself prove that a corresponding processing skill is installed or executable. Select only skills listed in the available skill catalogue.

## Availability and status language

| Intended meaning | Common terms and aliases |
|---|---|
| Registered | registered, recorded, listed, in inventory, uploaded, known to the project |
| Physically accessible | available, accessible, present on disk, exists, readable, can be opened |
| Missing | missing, absent, unavailable, not found, deleted, inaccessible (distinguish inaccessible from missing where possible) |
| Ready | ready, valid, usable, prepared, completed validation |
| Failed | failed, error, invalid, rejected |

Do not treat “registered” and “physically accessible” as equivalent. Report the distinction when relevant.

## Interpretation rules

1. Normalize capitalization, whitespace, hyphens, and underscores before matching known aliases.
2. Use the canonical dataset kind in handler parameters; never pass a display synonym such as `well logs` directly to a database filter.
3. Prefer the most specific phrase. For example, `checkshot data` means `checkshot`, not generic `well_log`.
4. Interpret abbreviations such as `LAS`, `SGY`, and `SEG-Y` using the dataset context above.
5. Do not infer a dataset type from a well name, survey name, or file extension alone when the meaning is ambiguous.
6. If a term could refer to multiple concepts, ask a concise clarification rather than guessing.
7. Synonyms improve language understanding; database records remain the source of truth for actual project inventory.

## Well and survey entity inventory

- "what well do I have?", "which wells are available?", "list wells" -> `well_survey_inventory` with `entity_type=wells`.
- "what survey do I have?", "which surveys are available?", "list surveys" -> `well_survey_inventory` with `entity_type=surveys`.
- These entity-list requests are distinct from dataset questions. "well data", "well logs", "survey data", and named-well data requests -> `data_inventory`.
