"""Minimum-curvature well trajectories:  MD  ->  TVD below RT  ->  TVDSS  and  X / Y.

Why this matters
----------------
Seismic lives on an (inline, crossline, TWT) grid, logs live on measured depth
(MD) along a possibly *deviated* hole.  s-1 reaches ~67 deg inclination and
o-3/no-4 >50 deg, so their reservoir sections sit 1-2 km sideways from the
surface location.  Treating them as vertical would put every label on the wrong
seismic trace.

Method - minimum curvature (industry standard)
----------------------------------------------
Between two survey stations the well is assumed to follow a circular arc.
With unit direction vectors t1, t2 and dogleg angle beta = arccos(t1 . t2)::

    dP = (dMD / 2) * RF * (t1 + t2),     RF = (2 / beta) * tan(beta / 2)   (RF = 1 if beta = 0)

For points *between* stations we evaluate the same arc exactly::

    P(s) = P1 + R * [ t1 sin(phi) + n (1 - cos(phi)) ],   R = dMD / beta,  phi = s / R,
    n = (t2 - t1 cos beta) / sin beta

(this reproduces the station-to-station formula at s = dMD - see tests).

Conventions written into every output
-------------------------------------
* MD is measured from the rotary table (files say ``MDKB``; header column ``RT``).
* ``tvd_rt_m``  = true vertical depth below RT, positive down.
* ``tvdss_m``   = ``tvd_rt_m - RT_elevation`` -> depth below mean sea level, positive down.
* ``x = X_surface + east``, ``y = Y_surface + north``.  This **assumes azimuths are
  referenced to grid north of the (undisclosed, randomised) X/Y system**.  If the
  organisers rotated the coordinates, deviated paths would be rotated too; this
  cannot be verified without an independent well path, so it is flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def direction_vectors(incl_deg, azim_deg) -> np.ndarray:
    """Unit vectors (north, east, down) for inclination/azimuth in degrees."""
    inc = np.radians(np.asarray(incl_deg, dtype=float))
    azi = np.radians(np.asarray(azim_deg, dtype=float))
    return np.column_stack([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])


def minimum_curvature(md, incl_deg, azim_deg):
    """Station positions (north, east, tvd) by minimum curvature.

    Returns ``(positions[n,3], dogleg_rad[n-1], dls_deg_per_30m[n-1])``.
    """
    md = np.asarray(md, dtype=float)
    t = direction_vectors(incl_deg, azim_deg)
    dmd = np.diff(md)
    beta = np.arccos(np.clip(np.sum(t[:-1] * t[1:], axis=1), -1.0, 1.0))
    rf = np.ones_like(beta)
    curved = beta > 1e-9
    rf[curved] = 2.0 / beta[curved] * np.tan(beta[curved] / 2.0)
    steps = (dmd / 2.0 * rf)[:, None] * (t[:-1] + t[1:])
    pos = np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])
    with np.errstate(divide="ignore", invalid="ignore"):
        dls = np.where(dmd > 0, np.degrees(beta) / dmd * 30.0, np.nan)
    return pos, beta, dls


class WellTrajectory:
    """A surveyed well path that can be evaluated at any MD."""

    def __init__(self, well: str, stations: pd.DataFrame, rt_m: float, x0: float, y0: float,
                 azimuth_correction_deg: float = 0.0):
        self.well = well
        self.rt_m = float(rt_m)
        self.x0, self.y0 = float(x0), float(y0)
        self.flags: list[str] = []
        md = stations["md_m"].to_numpy(dtype=float)
        inc = stations["incl_deg"].to_numpy(dtype=float)
        azi = np.mod(stations["azim_deg"].to_numpy(dtype=float) + azimuth_correction_deg, 360.0)
        if md[0] > 0:  # vertical tie-on from the rotary table
            md, inc, azi = np.r_[0.0, md], np.r_[0.0, inc], np.r_[azi[0], azi]
            self.flags.append("vertical_tie_on_added_at_md0")
        keep = np.r_[True, np.diff(md) > 0]
        if not keep.all():
            self.flags.append(f"dropped_{int((~keep).sum())}_duplicate_md_stations")
        self.md, self.inc, self.azi = md[keep], inc[keep], azi[keep]
        self.t = direction_vectors(self.inc, self.azi)
        self.pos, self.beta, self.dls = minimum_curvature(self.md, self.inc, self.azi)

    # ------------------------------------------------------------------ evaluation
    def at_md(self, md_query) -> pd.DataFrame:
        """Position and direction at arbitrary MD values (vectorised, exact arcs)."""
        q = np.atleast_1d(np.asarray(md_query, dtype=float))
        n = len(self.md)
        pos = np.full((q.size, 3), np.nan)
        tan = np.full((q.size, 3), np.nan)
        extrap = np.zeros(q.size, dtype=bool)

        inside = (q >= 0) & (q <= self.md[-1])
        if n == 1:
            inside[:] = False
        if inside.any():
            qi = q[inside]
            idx = np.clip(np.searchsorted(self.md, qi, side="right") - 1, 0, n - 2)
            s = qi - self.md[idx]
            dmd = self.md[idx + 1] - self.md[idx]
            beta = self.beta[idx]
            t1, t2, p1 = self.t[idx], self.t[idx + 1], self.pos[idx]
            p = np.empty_like(t1)
            tv = np.empty_like(t1)
            straight = beta < 1e-9
            if straight.any():
                p[straight] = p1[straight] + s[straight, None] * t1[straight]
                tv[straight] = t1[straight]
            curved = ~straight
            if curved.any():
                b = beta[curved]
                radius = dmd[curved] / b
                phi = s[curved] / radius
                nvec = (t2[curved] - t1[curved] * np.cos(b)[:, None]) / np.sin(b)[:, None]
                p[curved] = p1[curved] + radius[:, None] * (t1[curved] * np.sin(phi)[:, None]
                                                            + nvec * (1 - np.cos(phi))[:, None])
                tv[curved] = t1[curved] * np.cos(phi)[:, None] + nvec * np.sin(phi)[:, None]
            pos[inside], tan[inside] = p, tv

        below = q > self.md[-1]
        if below.any():  # straight-line extrapolation with the last direction (flagged)
            pos[below] = self.pos[-1] + (q[below] - self.md[-1])[:, None] * self.t[-1]
            tan[below] = self.t[-1]
            extrap[below] = True

        tdown = np.clip(tan[:, 2], -1.0, 1.0)
        incl = np.degrees(np.arccos(tdown))
        azim = np.mod(np.degrees(np.arctan2(tan[:, 1], tan[:, 0])), 360.0)
        return pd.DataFrame({
            "md_m": q,
            "tvd_rt_m": pos[:, 2],
            "tvdss_m": pos[:, 2] - self.rt_m,
            "north_m": pos[:, 0],
            "east_m": pos[:, 1],
            "x": self.x0 + pos[:, 1],
            "y": self.y0 + pos[:, 0],
            "incl_deg": incl,
            "azim_deg": azim,
            "extrapolated_below_survey": extrap,
        })

    def stations_table(self) -> pd.DataFrame:
        """Survey stations with computed positions and dogleg severity."""
        df = pd.DataFrame({
            "well": self.well, "md_m": self.md, "incl_deg": self.inc, "azim_deg": self.azi,
            "tvd_rt_m": self.pos[:, 2], "tvdss_m": self.pos[:, 2] - self.rt_m,
            "north_m": self.pos[:, 0], "east_m": self.pos[:, 1],
            "x": self.x0 + self.pos[:, 1], "y": self.y0 + self.pos[:, 0],
        })
        df["dls_deg_per_30m"] = np.r_[np.nan, self.dls]
        return df

    def summary(self) -> dict:
        horizontal = np.hypot(self.pos[:, 0], self.pos[:, 1])
        return {
            "well": self.well, "n_stations": int(len(self.md)), "md_td_m": float(self.md[-1]),
            "tvd_rt_td_m": float(self.pos[-1, 2]), "tvdss_td_m": float(self.pos[-1, 2] - self.rt_m),
            "max_incl_deg": float(np.max(self.inc)),
            "max_dls_deg_per_30m": float(np.nanmax(self.dls)) if len(self.dls) else 0.0,
            "max_horizontal_displacement_m": float(horizontal.max()),
            "bottom_hole_x": float(self.x0 + self.pos[-1, 1]),
            "bottom_hole_y": float(self.y0 + self.pos[-1, 0]),
            "flags": ";".join(self.flags),
        }
