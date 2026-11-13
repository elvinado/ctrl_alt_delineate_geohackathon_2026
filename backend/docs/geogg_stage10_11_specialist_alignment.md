# GeoGG Stage 10 / Stage 11 alignment

## Stage 10 — audit, claims, scorecards and model cards

Stage 10 consumes immutable Stage F outputs from `source_inference_job_id`; it does not refit or rerun inference. The definitive backend now writes:

- Stage 00–09 provenance scorecard and artifact-level input/output audit, including expected/actual SHA-256 and explicit missing/unhashed states.
- Source lineage checks, prediction-product integrity checks, aggregate acceptance checks and an evidence-linked claims register.
- Native NPY volume statistics and prediction results scorecard.
- Release-matched Stage 08 evaluation scorecard and copied source evidence, explicitly marked as imported and not recalculated.
- Per-model cards tied to the immutable release manifest and matching evaluation rows.
- Environment/confidentiality status, figures, reproduction script, offline dashboard, handoff and package checksums.

The audit does not convert absent provenance into success. Code fingerprints/versions are surfaced when recorded, but code currency is not independently verified without a trusted expected-code registry. Stage 08 outer held-out metrics are not described as Stage 09 blind-well validation.

## Stage 11 — native volume export, SEG-Y interoperability and QC

Stage 11 remains present as `stages/11_volume_export/`. It publishes a native-NPY handoff manifest, product inventory, checksum-integrity table, value-domain QC, geometry QC, validation summary and handoff notes. Canonical arrays remain in the source Stage 09 inference job and are referenced by storage-root-relative paths rather than copied, avoiding a second multi-GB data set.

Native NPY remains canonical. Stage 11 can additionally export derived prediction SEG-Y files using the registered full-stack source as the geometry/header reference. Export is configurable (`volume_products.segy_export.enabled`, enabled by default), writes IEEE float32 products, uses an explicit -999.25 no-data value, and preserves source trace headers except the required sample-count/delay updates. Every product is reopened with the package reader and an independent byte parser; trace headers, geometry, sample axis and bit-exact sample roundtrip are checked. Per-product JSON sidecars record source NPY and reference SEG-Y checksums. Export uses disk-backed staging and chunked writes to avoid a second full-volume RAM allocation.

## Remaining specialist boundaries

- Independent blind-well evaluation still requires a frozen prediction / independent truth protocol.
- Complete specialist in-process audit context cannot be reconstructed if it was not persisted in upstream provenance.
- Code currency requires a trusted expected-code identity to compare against recorded fingerprints.
- Full geological volumetric interpretation, Petrel handoff and specialist-specific dashboard features remain separate work.
