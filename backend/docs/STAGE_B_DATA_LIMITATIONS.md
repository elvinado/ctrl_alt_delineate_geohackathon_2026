# Stage B data limitations and acceptance record

This document records observed limitations from the successful GeoGG foundations
run reviewed on 2026-11-08. These are input-data/provenance limitations, not
reasons to fabricate or force synthetic ties.

## Observed inventory

- 13 registered wells; 13/13 LAS files and 13/13 deviation surveys found.
- 9/9 registered seismic volumes read and geometry products generated.
- 0/13 wells have both an accepted sonic curve (DT/DTCO) and density curve (RHOB).
- Three T-Z tables were found. Their provenance was `team export`; none was
  approved as a measured checkshot.
- Stage 01 computed trajectories and T-Z functions for 13/13 wells.
- Tie statuses were 10 `provisional_linear_velocity` and 3 `unvalidated`;
  validated measured or synthetic ties: 0/13.

## Synthetic-versus-full-stack availability

A conventional acoustic synthetic requires a valid sonic curve and density
curve to derive acoustic impedance, reflectivity, and the synthetic trace.
The current LAS inventory reports neither required curve for any well.
Consequently, synthetic-versus-full-stack panels correctly report `NOT AVAILABLE`
and list the missing prerequisites. This is intentional fail-closed behaviour.
Do not create substitute traces or mark ties as validated from incomplete logs.

Before retrying synthetic generation:

1. Obtain the missing sonic and density logs, or inspect the source LAS headers
   for documented mnemonic aliases.
2. If aliases are used, add explicit, unit-aware alias mappings and verify curve
   semantics; do not treat arbitrary curves as DT or RHOB.
3. Re-run Stage 00 and confirm the expected curves are recognised for each well.
4. Re-run Stage 01 and inspect synthetic QC before assigning `synthetic_tied`.

## Checkshot provenance

The three available T-Z tables are currently classified as team exports, not
independently verified measured checkshots. They must remain provisional until
source provenance and acquisition metadata are established. Approval should
record, at minimum:

- original source/provider and source file identity;
- acquisition method (measured checkshot, VSP, or another explicitly named type);
- time datum/reference and depth datum/reference;
- MD/TVD/TVDSS basis and KB/DF/GL reference where applicable;
- units, well association, and any processing/correction history;
- reviewer, review date, and explicit approval decision.

A file's presence or a plausible-looking T-Z curve is not sufficient evidence
for `measured_checkshot` status. Until approval, retain provisional/unvalidated
status and downstream limitations.

## Remaining scientific QC

- Resolve the T-Z datum and MDKB metadata gaps before comparing T-Z tables with
  trajectories on a common datum.
- Review strong T-Z extrapolation beyond control points (reported for m-1 and
  np-1) before using those intervals downstream.
- Review irregular/curved XY geometry warnings for full/near/mid/far/ultrafar
  volumes; do not assume a simple affine IL/XL-to-XY transform.

## Stage B acceptance gates

Stage B is operationally migrated and executes, but remains scientifically
conditional until the following are addressed or formally accepted as data
limitations:

- [ ] Sonic and density coverage assessed; aliases validated where applicable.
- [ ] Checkshot provenance reviewed and approved or explicitly retained as
      provisional with supporting evidence.
- [ ] T-Z datum/depth-reference metadata resolved or documented as unavailable.
- [ ] Extrapolation and irregular geometry limitations reviewed for downstream use.

A missing source dataset can remain an explicit limitation; it must not be
silently converted into synthetic or validated evidence.
