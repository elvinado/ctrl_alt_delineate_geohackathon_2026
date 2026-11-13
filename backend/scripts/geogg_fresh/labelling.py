"""Label rules for stage 02: curve QC masks, lithology, fluid evidence, classes, grades.

Why a separate module?  The rules decide what the models will later learn as
"truth".  Keeping them here - small, vectorised and unit-tested - makes every
rule easy to read, review and change through the YAML config.

The decision chain for one depth sample
---------------------------------------
1. **Curve QC** - is VSH / PHIT / SWT a usable value?  Placeholders (VSH = 0 and
   PHIT = 0 together; PHIT above the physical bound) are masked, never clipped.
2. **Lithology** - from VSH with two cutoffs and a dead band:
   ``VSH <= vsh_sand_max -> sand`` (if porosity confirms a porous rock),
   ``VSH >= vsh_shale_min -> shale``, anything in between -> ``heterolithic``.
3. **Fluid evidence (sand only)** - from interpreted SWT and the provider pay flags:
   ``water_bearing``, ``hydrocarbon_potential``, ``unresolved`` or ``conflict``.
   Low SWT is *hydrocarbon potential*, never oil or gas by itself.
4. **Class** - shale / brine_sand / undetermined_sand (+ oil_sand / gas_sand only
   if the provider flags were accepted after a named review AND enough wells
   support the class) / unlabelled.
5. **Confidence grade** - ``A`` independent evidence (none exists in this dataset),
   ``B`` interpreted log far from cutoffs (and corroborated), ``C`` interpreted log
   near cutoffs or uncorroborated, ``U`` unlabelled.

Class IDs follow AGENTS.md A1: 0 shale, 1 brine_sand, 2 oil_sand, 3 gas_sand,
4 undetermined_sand, 5 other_rock, -1 unlabelled.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from .petrophysics import PHI_MAX_CUBIC_PACKING

# --------------------------------------------------------------------------- classes
CLASS_TABLE = [
    # id, name, meaning, minimum evidence (AGENTS.md A1 / brief 3.1)
    (0, "shale", "shale-dominated rock",
     "reviewed lithology or approved Vsh criterion + log context"),
    (1, "brine_sand", "sand-dominated and water-bearing",
     "sand evidence + validated high water saturation and/or water test"),
    (2, "oil_sand", "sand with specifically evidenced oil",
     "oil-specific test, sample, production or reviewed fluid interpretation"),
    (3, "gas_sand", "sand with specifically evidenced gas",
     "gas-specific test, sample, production or reviewed fluid interpretation"),
    (4, "undetermined_sand", "sand-dominated, fluid type unresolved",
     "sand evidence without definitive fluid attribution"),
    (5, "other_rock", "confirmed non-sand / non-shale lithology",
     "explicit distinct lithology (e.g. core, lithology log)"),
    (-1, "unlabelled", "absent, unreliable or conflicting ground truth",
     "kept for inference; excluded from supervised losses"),
]
CLASS_ID = {name: cid for cid, name, _, _ in CLASS_TABLE}
CLASS_NAME = {cid: name for cid, name, _, _ in CLASS_TABLE}
GRADE_SCORE = {"A": 3, "B": 2, "C": 1, "U": 0}

# --------------------------------------------------------------------------- policy
DEFAULT_POLICY: dict[str, Any] = {
    "policy_version": "labels_v1",
    "class_map_version": "geogg_fresh_classes_v1",
    "provenance_note": ("Baseline proxy cutoffs chosen by the pipeline author from the data distribution and the "
                        "provider's own net-pay cutoffs; NOT geologist-approved. Formation-specific values need "
                        "formation tops, which the organiser marks as not available."),
    # curve family (see io_wells.CURVE_FAMILIES) feeding each provider input
    "curve_families": {"vsh": "VSH", "phit": "PHIT", "swt": "SWT",
                       "gas_flag": "HC_INDICATOR_GAS", "oil_flag": "HC_INDICATOR_OIL"},
    "bounds": {"vsh": [0.0, 1.0], "phit": [0.0, PHI_MAX_CUBIC_PACKING], "swt": [0.0, 1.0]},
    "placeholders": {
        # VSH = 0 and PHIT = 0 at the same depth = no shale AND no pore space: either a value the
        # interpretation did not compute, or a tight/coal/carbonate streak.  Not usable as truth.
        "mask_zero_vsh_and_zero_phit": True,
        # PHIT exactly 0 is not a measured total porosity of a sedimentary rock at these depths.
        "mask_phit_exactly_zero": True,
        # PHIT above the physical bound: the joint VSH/PHIT/SWT model is unreliable at that depth
        # (placeholder, bad hole) -> mask all three targets there.
        "phit_out_of_bounds_masks_all": True,
    },
    "lithology": {
        "vsh_sand_max": 0.40,         # = provider's implied net-pay Vsh cutoff (see provider_flag_audit)
        "vsh_shale_min": 0.60,
        "phit_sand_min": 0.03,        # low-Vsh rock with less porosity: tight streak, lithology unconfirmed
        "confidence_margin": 0.10,    # grade B if Vsh is at least this far from the nearest cutoff
        "baseline_vsh_cutoff": 0.50,  # historic VSH50 proxy - comparison column only
    },
    "fluid": {
        "swt_water_min": 0.80,        # sand with SWT >= this and no pay flag -> water_bearing
        "swt_hc_potential_max": 0.70, # sand with SWT <= this -> hydrocarbon_potential (NOT oil/gas)
        "phit_min_for_swt": 0.05,     # Sw is ill-conditioned at very low porosity
        "brine_from_interpreted_swt": True,   # False -> water-bearing sand becomes undetermined_sand
        "grade_b_swt_water_min": 0.90,
        "grade_b_swt_hc_max": 0.60,
        "provider_flag_threshold": 0.5,       # PS_GIP / PS_OIP >= this counts as set
        "accept_provider_fluid_flags": False, # oil/gas classes from PS_OIP / PS_GIP ...
        "provider_flags_reviewed_by": None,   # ... only with a named reviewer AND accept = true
    },
    "support": {"min_wells_per_class": 3, "min_thickness_m_per_well": 3.0},
    "targets": {"swt_exclude_shale": True},
    "cutoff_sensitivity_pairs": [[0.50, 0.50], [0.45, 0.55], [0.40, 0.60], [0.35, 0.65], [0.30, 0.70]],
}


def _deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def effective_policy(cfg: dict) -> dict:
    """DEFAULT_POLICY overridden by the ``labels.policy`` block of the config, then sanity-checked."""
    pol = _deep_merge(DEFAULT_POLICY, ((cfg.get("labels") or {}).get("policy") or {}))
    lith, fl = pol["lithology"], pol["fluid"]
    if not 0 <= lith["vsh_sand_max"] <= lith["vsh_shale_min"] <= 1:
        raise ValueError("labels: need 0 <= vsh_sand_max <= vsh_shale_min <= 1")
    if not 0 <= fl["swt_hc_potential_max"] <= fl["swt_water_min"] <= 1:
        raise ValueError("labels: need 0 <= swt_hc_potential_max <= swt_water_min <= 1")
    pol["fluid"]["provider_flags_accepted"] = bool(fl["accept_provider_fluid_flags"]) and bool(
        str(fl.get("provider_flags_reviewed_by") or "").strip())
    return pol


def reason_codes(pol: dict) -> dict[str, str]:
    """Human-readable meaning of every reason code written to the label tables."""
    lp, fp, b = pol["lithology"], pol["fluid"], pol["bounds"]
    return {
        "vsh_ge_shale_min": f"VSH >= {lp['vsh_shale_min']} -> shale",
        "vsh_in_dead_band": f"{lp['vsh_sand_max']} < VSH < {lp['vsh_shale_min']}: heterolithic, lithology uncertain",
        "vsh_le_sand_max_and_porous": f"VSH <= {lp['vsh_sand_max']} and PHIT >= {lp['phit_sand_min']} -> sand",
        "low_vsh_but_not_porous": (f"VSH <= {lp['vsh_sand_max']} but PHIT missing, zero or < {lp['phit_sand_min']}: "
                                   "tight streak, coal or carbonate possible - lithology unconfirmed"),
        "vsh_missing": "VSH missing (LAS null)",
        "vsh_out_of_bounds": f"VSH outside {b['vsh']}",
        "vsh_placeholder_zero_vsh_and_zero_phit": "VSH = 0 and PHIT = 0 together: not a usable interpretation",
        "vsh_artifact_phit_out_of_bounds": f"PHIT outside {b['phit']}: whole interpretation at this depth masked",
        "swt_ge_water_min": f"sand with interpreted SWT >= {fp['swt_water_min']} and no provider pay flag",
        "swt_le_hc_max": f"sand with interpreted SWT <= {fp['swt_hc_potential_max']} (hydrocarbon POTENTIAL, not oil/gas)",
        "swt_le_hc_max_and_provider_flag": "low SWT corroborated by provider pay flag (still not an oil/gas truth)",
        "provider_flag_swt_not_decisive": "provider pay flag set; SWT missing or between the two SWT cutoffs",
        "provider_flag_vs_high_swt": f"provider pay flag set but SWT >= {fp['swt_water_min']}: conflict",
        "provider_gas_and_oil_both_set": "PS_GIP and PS_OIP both set at the same depth: conflict",
        "swt_invalid_or_mid_range_no_flag": (f"SWT missing/invalid or between {fp['swt_hc_potential_max']} and "
                                             f"{fp['swt_water_min']}, no provider pay flag"),
    }


# --------------------------------------------------------------------------- helpers
def sample_thickness(depth: np.ndarray) -> np.ndarray:
    """Thickness represented by each sample: half the distance to each neighbour (m)."""
    d = np.asarray(depth, dtype=float)
    if d.size < 2:
        return np.zeros_like(d)
    gaps = np.diff(d)
    gaps = np.where(np.isfinite(gaps) & (gaps > 0), gaps, 0.0)
    left = np.r_[gaps[0], gaps]
    right = np.r_[gaps, gaps[-1]]
    return 0.5 * (left + right)


def _flag_state(values: np.ndarray, threshold: float) -> np.ndarray:
    """-1 not evaluated (NaN), 0 not set, 1 set."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, -1, dtype=int)
    ok = np.isfinite(v)
    out[ok] = (v[ok] >= threshold).astype(int)
    return out


# --------------------------------------------------------------------------- step 1: QC
def qc_curves(df: pd.DataFrame, pol: dict) -> pd.DataFrame:
    """Add validity masks and text flags for VSH, PHIT, SWT and the provider pay flags.

    Expects columns ``vsh_raw, phit_raw, swt_raw, gas_flag_raw, oil_flag_raw`` (NaN = missing).
    Raw values are never modified - they stay in the ``*_raw`` columns for the audit.
    """
    out = df.copy()
    v, p, s = (out[c].to_numpy(dtype=float) for c in ("vsh_raw", "phit_raw", "swt_raw"))
    (vlo, vhi), (plo, phi_), (slo, shi) = pol["bounds"]["vsh"], pol["bounds"]["phit"], pol["bounds"]["swt"]
    ph = pol["placeholders"]

    zero_zero = (v == 0) & (p == 0) if ph["mask_zero_vsh_and_zero_phit"] else np.zeros(v.shape, bool)
    phit_oob = np.isfinite(p) & ((p < plo) | (p > phi_))
    artifact = zero_zero | (phit_oob if ph["phit_out_of_bounds_masks_all"] else np.zeros(v.shape, bool))

    # ---- VSH
    vsh_flag = np.full(v.shape, "ok", dtype=object)
    vsh_flag[~np.isfinite(v)] = "missing"
    vsh_oob = np.isfinite(v) & ((v < vlo) | (v > vhi))
    vsh_flag[vsh_oob] = "out_of_bounds"
    vsh_flag[phit_oob & np.isfinite(v)] = "artifact_phit_out_of_bounds"
    vsh_flag[zero_zero] = "placeholder_zero_vsh_and_zero_phit"
    vsh_valid = np.isfinite(v) & ~vsh_oob & ~artifact
    vsh_flag[vsh_valid & (v == vlo)] = "valid_at_lower_limit"
    vsh_flag[vsh_valid & (v == vhi)] = "valid_at_upper_limit"

    # ---- PHIT
    phit_flag = np.full(p.shape, "ok", dtype=object)
    phit_flag[~np.isfinite(p)] = "missing"
    phit_zero = np.isfinite(p) & (p == 0) & bool(ph["mask_phit_exactly_zero"])
    phit_flag[phit_zero] = "zero_value"
    phit_flag[phit_oob] = "out_of_bounds"
    phit_flag[zero_zero] = "placeholder_zero_vsh_and_zero_phit"
    phit_valid = np.isfinite(p) & ~phit_oob & ~phit_zero & ~artifact

    # ---- SWT (needs a valid, not-too-small porosity)
    swt_flag = np.full(s.shape, "ok", dtype=object)
    swt_flag[~np.isfinite(s)] = "missing"
    swt_oob = np.isfinite(s) & ((s < slo) | (s > shi))
    swt_flag[swt_oob] = "out_of_bounds"
    low_phi = phit_valid & (p < pol["fluid"]["phit_min_for_swt"])
    swt_flag[np.isfinite(s) & ~swt_oob & ~phit_valid] = "porosity_invalid"
    swt_flag[np.isfinite(s) & ~swt_oob & low_phi] = "porosity_below_min_for_sw"
    swt_flag[artifact & np.isfinite(s)] = "artifact"
    swt_valid = np.isfinite(s) & ~swt_oob & phit_valid & ~low_phi & ~artifact
    swt_flag[swt_valid & (s == shi)] = "valid_at_upper_limit"

    # ---- provider pay flags (screening evidence, never truth by themselves)
    thr = float(pol["fluid"]["provider_flag_threshold"])
    g_raw = out["gas_flag_raw"].to_numpy(dtype=float)
    o_raw = out["oil_flag_raw"].to_numpy(dtype=float)
    gas, oil = _flag_state(g_raw, thr), _flag_state(o_raw, thr)
    pay = np.full(v.shape, "not_evaluated", dtype=object)
    evaluated = (gas >= 0) | (oil >= 0)
    pay[evaluated] = "none"
    pay[gas == 1] = "gas"
    pay[oil == 1] = "oil"
    pay[(gas == 1) & (oil == 1)] = "gas_and_oil_conflict"
    fractional = (np.isfinite(g_raw) & (g_raw > 0) & (g_raw < 1)) | (np.isfinite(o_raw) & (o_raw > 0) & (o_raw < 1))

    out["artifact_interval"] = artifact
    out["vsh_valid"], out["vsh_flag"] = vsh_valid, vsh_flag
    out["phit_valid"], out["phit_flag"] = phit_valid, phit_flag
    out["swt_valid"], out["swt_flag"] = swt_valid, swt_flag
    out["provider_pay_flag"] = pay
    out["provider_flag_fractional"] = fractional
    return out


# --------------------------------------------------------------------------- steps 2-5
def assign_labels(df: pd.DataFrame, pol: dict) -> pd.DataFrame:
    """Lithology, fluid evidence, class, status, grade and training targets (after qc_curves)."""
    out = df.copy()
    lith_p, fl_p = pol["lithology"], pol["fluid"]
    v = out["vsh_raw"].to_numpy(dtype=float)
    p = out["phit_raw"].to_numpy(dtype=float)
    s = out["swt_raw"].to_numpy(dtype=float)
    vsh_ok = out["vsh_valid"].to_numpy(bool)
    phit_ok = out["phit_valid"].to_numpy(bool)
    swt_ok = out["swt_valid"].to_numpy(bool)
    pay = out["provider_pay_flag"].to_numpy(dtype=object)
    n = len(out)

    # ---- step 2: lithology ------------------------------------------------------------
    lith = np.full(n, "not_interpretable", dtype=object)
    lith_reason = np.array([f"vsh_{f}" for f in out["vsh_flag"]], dtype=object)
    sand_v = vsh_ok & (v <= lith_p["vsh_sand_max"])
    shale_v = vsh_ok & (v >= lith_p["vsh_shale_min"])
    het = vsh_ok & ~sand_v & ~shale_v
    porous = phit_ok & (p >= lith_p["phit_sand_min"])
    lith[shale_v], lith_reason[shale_v] = "shale", "vsh_ge_shale_min"
    lith[het], lith_reason[het] = "heterolithic", "vsh_in_dead_band"
    lith[sand_v & porous], lith_reason[sand_v & porous] = "sand", "vsh_le_sand_max_and_porous"
    tight = sand_v & ~porous
    lith[tight], lith_reason[tight] = "tight_nonshale", "low_vsh_but_not_porous"
    is_sand = lith == "sand"

    # ---- step 3: fluid evidence in sand -----------------------------------------------
    fluid = np.full(n, "", dtype=object)
    fluid_reason = np.full(n, "", dtype=object)
    ev_type = np.full(n, "", dtype=object)
    flagged = np.isin(pay, ["gas", "oil"])
    conflict_flags = pay == "gas_and_oil_conflict"
    water_sw = swt_ok & (s >= fl_p["swt_water_min"])
    hc_sw = swt_ok & (s <= fl_p["swt_hc_potential_max"])

    fluid[is_sand], fluid_reason[is_sand] = "unresolved", "swt_invalid_or_mid_range_no_flag"
    m = is_sand & water_sw & ~flagged & ~conflict_flags
    fluid[m], fluid_reason[m], ev_type[m] = "water_bearing", "swt_ge_water_min", "interpreted_log_swt"
    m = is_sand & hc_sw & ~conflict_flags
    fluid[m], fluid_reason[m], ev_type[m] = "hydrocarbon_potential", "swt_le_hc_max", "interpreted_log_swt"
    m = is_sand & flagged & ~water_sw
    fluid[m] = "hydrocarbon_potential"
    fluid_reason[m] = np.where(hc_sw[m], "swt_le_hc_max_and_provider_flag", "provider_flag_swt_not_decisive")
    ev_type[m] = np.where(hc_sw[m], "interpreted_log_swt+provider_pay_flag", "provider_pay_flag")
    m = is_sand & flagged & water_sw
    fluid[m], fluid_reason[m], ev_type[m] = "conflict", "provider_flag_vs_high_swt", "conflicting"
    m = is_sand & conflict_flags
    fluid[m], fluid_reason[m], ev_type[m] = "conflict", "provider_gas_and_oil_both_set", "conflicting"

    # ---- step 4: class -------------------------------------------------------------------
    cls = np.full(n, CLASS_ID["unlabelled"], dtype=int)
    cls[lith == "shale"] = CLASS_ID["shale"]
    cls[is_sand] = CLASS_ID["undetermined_sand"]
    if fl_p["brine_from_interpreted_swt"]:
        cls[is_sand & (fluid == "water_bearing")] = CLASS_ID["brine_sand"]
    if fl_p.get("provider_flags_accepted"):
        hc = is_sand & (fluid == "hydrocarbon_potential")
        cls[hc & (pay == "gas")] = CLASS_ID["gas_sand"]
        cls[hc & (pay == "oil")] = CLASS_ID["oil_sand"]

    status = np.full(n, "unlabelled", dtype=object)
    status[cls == CLASS_ID["shale"]] = "provisional_interpreted_log"
    status[cls == CLASS_ID["brine_sand"]] = "provisional_interpreted_log"
    status[cls == CLASS_ID["undetermined_sand"]] = "sand_fluid_unresolved"
    status[np.isin(cls, [CLASS_ID["oil_sand"], CLASS_ID["gas_sand"]])] = "provider_fluid_flag_reviewed"

    # ---- step 5: confidence grade ------------------------------------------------------------
    margin = float(lith_p["confidence_margin"])
    lith_grade = np.full(n, "U", dtype=object)
    lith_grade[lith == "shale"] = np.where(v[lith == "shale"] >= lith_p["vsh_shale_min"] + margin, "B", "C")
    lith_grade[is_sand] = np.where(v[is_sand] <= lith_p["vsh_sand_max"] - margin, "B", "C")
    fluid_grade = np.full(n, "C", dtype=object)
    wb = fluid == "water_bearing"
    fluid_grade[wb] = np.where((s[wb] >= fl_p["grade_b_swt_water_min"]) & (pay[wb] == "none"), "B", "C")
    hp = fluid == "hydrocarbon_potential"
    fluid_grade[hp] = np.where((s[hp] <= fl_p["grade_b_swt_hc_max"]) & np.isin(pay[hp], ["gas", "oil"]), "B", "C")
    grade = lith_grade.copy()

    def score(arr):
        return pd.Series(arr, dtype=object).map(GRADE_SCORE).to_numpy(dtype=int)

    worse = is_sand & (score(fluid_grade) < score(lith_grade))
    grade[worse] = fluid_grade[worse]
    grade[cls == CLASS_ID["unlabelled"]] = "U"

    # ---- training targets (masks keep NaN, never imputed) -----------------------------------
    out["lithology_label"], out["lithology_reason"] = lith, lith_reason
    out["fluid_label"], out["fluid_reason"] = fluid, fluid_reason
    out["fluid_evidence_type"] = ev_type
    out["class_id"] = cls
    out["class_name"] = [CLASS_NAME[c] for c in cls]
    out["class_status"] = status
    out["confidence_grade"] = grade
    out["label_source"] = np.where(cls == CLASS_ID["unlabelled"], "", f"provider_logs+{pol['policy_version']}")
    out["y_lithology"] = np.select([lith == "sand", lith == "shale"], [1, 0], default=-1)
    out["y_sand_fluid"] = np.select([fluid == "water_bearing", fluid == "hydrocarbon_potential"], [0, 1], default=-1)
    out["vsh_fraction"] = np.where(vsh_ok, v, np.nan)
    out["phit_fraction"] = np.where(phit_ok, p, np.nan)
    swt_target = swt_ok.copy()
    if pol["targets"]["swt_exclude_shale"]:
        swt_target &= lith != "shale"
    out["swt_fraction"] = np.where(swt_target, s, np.nan)
    out["vsh_mask"], out["phit_mask"], out["swt_mask"] = vsh_ok, phit_ok, swt_target
    base = float(lith_p["baseline_vsh_cutoff"])
    out["lith_vsh50_baseline"] = np.select([np.isfinite(v) & (v < base), np.isfinite(v) & (v >= base)],
                                           ["sand", "shale"], default="")
    return out


# --------------------------------------------------------------------------- support
def class_thickness_by_well(df: pd.DataFrame, column: str = "class_name") -> pd.DataFrame:
    """MD thickness (m) per well and value of ``column`` (needs ``md_step_m``)."""
    return (df.groupby(["well", column], sort=False)["md_step_m"].sum()
            .unstack(fill_value=0.0))


def task_support(df: pd.DataFrame, pol: dict, development_wells: list[str]) -> pd.DataFrame:
    """Decide which learning tasks the labels can support (AGENTS.md A1 feasibility rule).

    A class is supported when at least ``min_wells_per_class`` development wells each
    carry at least ``min_thickness_m_per_well`` metres (MD) of it.
    """
    sup = pol["support"]
    min_w, min_t = int(sup["min_wells_per_class"]), float(sup["min_thickness_m_per_well"])
    dev = df[df["well"].isin(development_wells)]
    thick = class_thickness_by_well(dev, "class_name") if len(dev) else pd.DataFrame()

    def wells_with(name: str) -> list[str]:
        if name not in thick.columns:
            return []
        return sorted(thick.index[thick[name] >= min_t].tolist())

    accepted = bool(pol["fluid"].get("provider_flags_accepted"))
    sand = dev[dev["lithology_label"] == "sand"]
    cand = {}
    for fluid in ("gas", "oil"):
        t = sand[sand["provider_pay_flag"] == fluid].groupby("well")["md_step_m"].sum()
        cand[fluid] = sorted(t.index[t >= min_t].tolist())

    rows = []
    for name in ("shale", "brine_sand", "undetermined_sand", "oil_sand", "gas_sand", "other_rock"):
        wells = wells_with(name)
        ok = len(wells) >= min_w
        if name in ("oil_sand", "gas_sand") and not accepted:
            fl = "oil" if name == "oil_sand" else "gas"
            status = "TASK_UNSUPPORTED"
            reason = (f"No independent {fl} evidence (no test, sample or production data supplied). The provider "
                      f"PS_{'OIP' if fl == 'oil' else 'GIP'} flag is a net-pay flag derived from the same "
                      f"VSH/PHIT/SWT cutoffs with undocumented fluid typing; it is set in >= {min_t} m of sand in "
                      f"{len(cand[fl])} development wells ({', '.join(cand[fl]) or 'none'}) - candidates only, "
                      "usable after a named review (labels.policy.fluid.accept_provider_fluid_flags).")
        elif name == "other_rock":
            status = "TASK_UNSUPPORTED"
            reason = "No lithology log or core to confirm non-sand/non-shale rock; tight/zero intervals stay unlabelled."
        elif name == "brine_sand" and not pol["fluid"]["brine_from_interpreted_swt"]:
            status = "TASK_UNSUPPORTED"
            reason = "brine_from_interpreted_swt = false and no water tests supplied."
        else:
            status = "SUPPORTED_PROVISIONAL" if ok else "TASK_UNSUPPORTED"
            reason = (f"{len(wells)} development wells with >= {min_t} m (need {min_w}); labels come from "
                      "interpreted logs, not independent tests" if ok else
                      f"only {len(wells)} development wells with >= {min_t} m (need {min_w})")
        rows.append({"task": f"class:{name}", "status": status, "n_wells_supporting": len(wells),
                     "wells_supporting": ";".join(wells), "criterion": f">= {min_t} m MD in >= {min_w} wells",
                     "reason": reason})

    def status_of(name):
        return next(r["status"] for r in rows if r["task"] == f"class:{name}")

    five_ok = all(status_of(c) != "TASK_UNSUPPORTED" for c in ("shale", "brine_sand", "oil_sand", "gas_sand"))
    rows.append({"task": "five_class_flat", "status": "SUPPORTED_PROVISIONAL" if five_ok else "TASK_UNSUPPORTED",
                 "n_wells_supporting": np.nan, "wells_supporting": "", "criterion": "all of shale/brine/oil/gas supported",
                 "reason": "" if five_ok else "oil_sand and/or gas_sand unsupported -> use the hierarchical tasks below"})
    lith_wells = sorted(set(wells_with("shale")) & set(
        thick.index[(thick.get("brine_sand", 0) + thick.get("undetermined_sand", 0) + thick.get("oil_sand", 0)
                     + thick.get("gas_sand", 0)) >= min_t].tolist() if len(thick) else []))
    rows.append({"task": "hierarchical_1_lithology_sand_vs_shale",
                 "status": "SUPPORTED_PROVISIONAL" if len(lith_wells) >= min_w else "TASK_UNSUPPORTED",
                 "n_wells_supporting": len(lith_wells), "wells_supporting": ";".join(lith_wells),
                 "criterion": f"sand and shale each >= {min_t} m in >= {min_w} wells",
                 "reason": "target y_lithology (1 sand, 0 shale, -1 masked)"})
    fl_thick = (dev[dev["y_sand_fluid"] >= 0].groupby(["well", "fluid_label"])["md_step_m"].sum()
                .unstack(fill_value=0.0)) if len(dev) else pd.DataFrame()
    fw = sorted(fl_thick.index[(fl_thick.get("water_bearing", 0) >= min_t)
                               & (fl_thick.get("hydrocarbon_potential", 0) >= min_t)].tolist()) if len(fl_thick) else []
    rows.append({"task": "hierarchical_2_sand_water_vs_hc_potential",
                 "status": "SUPPORTED_PROVISIONAL" if len(fw) >= min_w else "TASK_UNSUPPORTED",
                 "n_wells_supporting": len(fw), "wells_supporting": ";".join(fw),
                 "criterion": f"water_bearing and hydrocarbon_potential each >= {min_t} m in >= {min_w} wells",
                 "reason": "target y_sand_fluid (0 water-bearing, 1 hydrocarbon potential); interpreted-log basis"})
    for tgt in ("vsh", "phit", "swt"):
        t = dev[dev[f"{tgt}_mask"]].groupby("well")["md_step_m"].sum()
        wells = sorted(t.index[t >= min_t].tolist())
        rows.append({"task": f"regression_{tgt}",
                     "status": "SUPPORTED_PROVISIONAL" if len(wells) >= min_w else "TASK_UNSUPPORTED",
                     "n_wells_supporting": len(wells), "wells_supporting": ";".join(wells),
                     "criterion": f">= {min_t} m of valid target in >= {min_w} wells",
                     "reason": f"target {tgt}_fraction with {tgt}_mask (provider interpreted curve)"})
    return pd.DataFrame(rows)


def demote_unsupported_classes(df: pd.DataFrame, support: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Oil/gas samples whose class lacks multi-well support fall back to undetermined_sand."""
    out = df.copy()
    demoted = []
    for name in ("oil_sand", "gas_sand", "brine_sand"):
        row = support[support["task"] == f"class:{name}"]
        if len(row) and row.iloc[0]["status"] == "TASK_UNSUPPORTED":
            m = out["class_id"] == CLASS_ID[name]
            if m.any():
                out.loc[m, "class_id"] = CLASS_ID["undetermined_sand"]
                out.loc[m, "class_name"] = "undetermined_sand"
                out.loc[m, "class_status"] = f"demoted_from_{name}_unsupported"
                demoted.append(name)
    return out, demoted


def cutoff_sensitivity(df: pd.DataFrame, pol: dict, development_wells: list[str]) -> pd.DataFrame:
    """Sand / shale / heterolithic thickness for alternative Vsh cutoff pairs (valid VSH only)."""
    dev = df[df["well"].isin(development_wells) & df["vsh_valid"]]
    min_t = float(pol["support"]["min_thickness_m_per_well"])
    rows = []
    for sand_max, shale_min in pol["cutoff_sensitivity_pairs"]:
        v = dev["vsh_raw"].to_numpy(float)
        porous = dev["phit_valid"].to_numpy(bool) & (dev["phit_raw"].to_numpy(float) >= pol["lithology"]["phit_sand_min"])
        sand = (v <= sand_max) & porous
        shale = v >= shale_min
        t = dev["md_step_m"].to_numpy(float)
        per_well_sand = pd.Series(t * sand).groupby(dev["well"].to_numpy()).sum()
        rows.append({"vsh_sand_max": sand_max, "vsh_shale_min": shale_min,
                     "sand_md_m": float((t * sand).sum()), "shale_md_m": float((t * shale).sum()),
                     "heterolithic_md_m": float((t * (~sand & ~shale & (v > sand_max))).sum()),
                     "n_wells_sand_ge_min": int((per_well_sand >= min_t).sum()),
                     "is_active_policy": (sand_max == pol["lithology"]["vsh_sand_max"]
                                          and shale_min == pol["lithology"]["vsh_shale_min"])})
    return pd.DataFrame(rows)
