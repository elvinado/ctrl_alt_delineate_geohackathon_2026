"""Standard petrophysical equations used by stage 02_labels.

Every function is a plain, unit-documented formula.  Stage 02 applies a method
ONLY when (a) the input curves really exist in the LAS files and (b) the
parameters are given in the config with a provenance note.  Otherwise the
method is reported as ``NOT_APPLIED`` with the reason - it never fills a gap
with default numbers disguised as measurements.

Formulas
--------
* Gamma-ray index           IGR = (GR - GR_clean) / (GR_shale - GR_clean)
* Linear Vsh                Vsh = IGR
* Larionov, Tertiary rocks  Vsh = 0.083 * (2**(3.7 * IGR) - 1)
* Larionov, older rocks     Vsh = 0.33  * (2**(2.0 * IGR) - 1)
* Density porosity          PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)
* Neutron-density average   PHIND = (PHID + NPHI) / 2          (simple, lithology-dependent)
* Archie (clean sand)       Sw = (a * Rw / (Rt * phi**m)) ** (1/n)
* Bulk volume water         BVW = PHIT * SWT

All saturations and porosities are fractions (v/v), densities g/cm3, resistivity ohm.m.
"""

from __future__ import annotations

import numpy as np

# Theoretical maximum porosity of a packing of equal spheres (cubic packing).
# Used as the upper physical bound for an interpreted total porosity.
PHI_MAX_CUBIC_PACKING = 0.476


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def gamma_ray_index(gr, gr_clean: float, gr_shale: float) -> np.ndarray:
    """IGR in [0, 1] (values outside are clipped and should be flagged by the caller).

    ``gr_clean`` and ``gr_shale`` must be formation-specific endpoints picked by a
    petrophysicist (e.g. P5/P95 of GR within one formation) - never global defaults.
    """
    if not gr_shale > gr_clean:
        raise ValueError(f"GR_shale ({gr_shale}) must be greater than GR_clean ({gr_clean})")
    return np.clip((_arr(gr) - gr_clean) / (gr_shale - gr_clean), 0.0, 1.0)


def vsh_linear(igr) -> np.ndarray:
    """Linear Vsh = IGR (an upper bound on shale volume for most rocks)."""
    return np.clip(_arr(igr), 0.0, 1.0)


def vsh_larionov_tertiary(igr) -> np.ndarray:
    """Larionov (1969) for young, unconsolidated (Tertiary) rocks."""
    return np.clip(0.083 * (2.0 ** (3.7 * np.clip(_arr(igr), 0, 1)) - 1.0), 0.0, 1.0)


def vsh_larionov_older(igr) -> np.ndarray:
    """Larionov (1969) for older, consolidated rocks."""
    return np.clip(0.33 * (2.0 ** (2.0 * np.clip(_arr(igr), 0, 1)) - 1.0), 0.0, 1.0)


def density_porosity(rhob, rho_matrix: float, rho_fluid: float) -> np.ndarray:
    """PHID (fraction).  Not clipped: values < 0 or > 0.476 are QC flags for the caller.

    RHOB in g/cm3 (convert kg/m3 by /1000 first).  The matrix/fluid densities are
    assumptions that must be stated (2.65 quartz, 2.71 calcite; 1.0-1.1 brine filtrate).
    """
    if not rho_matrix > rho_fluid:
        raise ValueError("rho_matrix must exceed rho_fluid")
    return (rho_matrix - _arr(rhob)) / (rho_matrix - rho_fluid)


def neutron_density_porosity(phid, nphi) -> np.ndarray:
    """Arithmetic neutron-density average (lithology/gas sensitive; a screening estimate)."""
    return 0.5 * (_arr(phid) + _arr(nphi))


def archie_sw(rt, phi, rw: float, a: float = 1.0, m: float = 2.0, n: float = 2.0) -> np.ndarray:
    """Clean-sand Archie water saturation (fraction), clipped to [0, 1].

    Only meaningful in clean sands with a documented Rw (at formation temperature),
    a deep resistivity measured in the uninvaded zone and a porosity definition
    compatible with how a, m, n were calibrated.  Returns NaN where Rt or phi <= 0.
    """
    rt = _arr(rt)
    phi = _arr(phi)
    out = np.full(np.broadcast(rt, phi).shape, np.nan)
    ok = (rt > 0) & (phi > 0)
    out[ok] = (a * rw / (rt[ok] * phi[ok] ** m)) ** (1.0 / n)
    return np.clip(out, 0.0, 1.0)


def bulk_volume_water(phit, swt) -> np.ndarray:
    """BVW = PHIT * SWT (fraction of rock volume filled with water)."""
    return _arr(phit) * _arr(swt)


# Registry used by stage 02 to report every method, applied or not.
METHODS = [
    # name,               target, required curve families, required config keys
    ("vsh_gr_linear",      "vsh",  ("GR",),               ("gr_clean", "gr_shale")),
    ("vsh_gr_larionov",    "vsh",  ("GR",),               ("gr_clean", "gr_shale", "larionov_variant")),
    ("phit_density",       "phit", ("RHOB",),             ("rho_matrix", "rho_fluid")),
    ("phit_neutron_density", "phit", ("RHOB", "NPHI"),    ("rho_matrix", "rho_fluid")),
    ("swt_archie",         "swt",  ("RT_DEEP",),          ("rw", "a", "m", "n")),
]
