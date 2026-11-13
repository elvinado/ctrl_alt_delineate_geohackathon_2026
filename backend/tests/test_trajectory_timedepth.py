"""Trajectory geometry and T-Z physics with analytic answers.  Software tests only."""

import unittest

import _fixtures  # noqa: F401

import numpy as np
import pandas as pd

from geogg_fresh.timedepth import (PolynomialTZ, TableTZ, fit_anchored_polynomial, fit_linear_velocity,
                                   leave_one_well_out, physical_extrapolation_check, velocity_qc)
from geogg_fresh.trajectory import WellTrajectory


def _survey(md, inc, azi):
    return pd.DataFrame({"md_m": md, "incl_deg": inc, "azim_deg": azi, "null_filled": False})


class TrajectoryTests(unittest.TestCase):
    def test_vertical_well(self):
        t = WellTrajectory("v", _survey([0, 1000, 2000], [0, 0, 0], [0, 0, 0]), rt_m=25.0, x0=10.0, y0=20.0)
        p = t.at_md([0, 500, 2000])
        np.testing.assert_allclose(p["tvd_rt_m"], [0, 500, 2000], atol=1e-9)
        np.testing.assert_allclose(p["tvdss_m"], [-25, 475, 1975], atol=1e-9)
        np.testing.assert_allclose(p["x"], 10.0)
        np.testing.assert_allclose(p["y"], 20.0)

    def test_straight_slant_east(self):
        md = np.arange(0, 1001, 100.0)
        t = WellTrajectory("s", _survey(md, np.full(md.size, 30.0), np.full(md.size, 90.0)), 0, 0, 0)
        p = t.at_md([0, 437.5, 1000])
        np.testing.assert_allclose(p["east_m"], np.array([0, 437.5, 1000]) * np.sin(np.radians(30)), atol=1e-6)
        np.testing.assert_allclose(p["tvd_rt_m"], np.array([0, 437.5, 1000]) * np.cos(np.radians(30)), atol=1e-6)
        np.testing.assert_allclose(p["north_m"], 0.0, atol=1e-6)

    def test_circular_build_is_exact(self):
        # constant build from 0 to 90 deg over L = 500 m towards north -> circle radius R = L / (pi/2)
        length = 500.0
        md = np.linspace(0, length, 11)
        inc = np.degrees(md / length * np.pi / 2)
        t = WellTrajectory("arc", _survey(md, inc, np.zeros(md.size)), 0, 0, 0)
        radius = length / (np.pi / 2)
        q = np.array([0, 123.4, 250, 377.7, 500])
        p = t.at_md(q)
        np.testing.assert_allclose(p["north_m"], radius * (1 - np.cos(q / radius)), atol=1e-6)
        np.testing.assert_allclose(p["tvd_rt_m"], radius * np.sin(q / radius), atol=1e-6)
        np.testing.assert_allclose(p["incl_deg"], np.degrees(q / radius), atol=1e-6)

    def test_azimuth_wrap_does_not_loop(self):
        t = WellTrajectory("w", _survey([0, 100, 200], [10, 10, 10], [359, 1, 359]), 0, 0, 0)
        p = t.at_md([200])
        self.assertGreater(p["north_m"].iloc[0], 30.0)
        self.assertLess(abs(p["east_m"].iloc[0]), 1.0)

    def test_extrapolation_below_survey_is_flagged(self):
        t = WellTrajectory("e", _survey([0, 1000], [0, 0], [0, 0]), 0, 0, 0)
        p = t.at_md([900, 1100])
        self.assertEqual(p["extrapolated_below_survey"].tolist(), [False, True])
        self.assertAlmostEqual(p["tvd_rt_m"].iloc[1], 1100.0)


class TimeDepthTests(unittest.TestCase):
    def test_interval_velocity_units(self):
        z = np.arange(0, 3001, 100.0)
        twt = 2000.0 * z / 2500.0                    # constant 2500 m/s
        tab = TableTZ("c", z, twt, "test")
        np.testing.assert_allclose(tab.vint(np.array([50.0, 1550.0, 2950.0])), 2500.0)
        poly = fit_anchored_polynomial(z, twt, degree=1)
        np.testing.assert_allclose(poly.vint(np.array([10.0, 2000.0])), 2500.0, rtol=1e-9)
        self.assertAlmostEqual(float(poly.twt(np.array([1000.0]))[0]), 800.0, places=6)

    def test_linear_velocity_law_recovered(self):
        v0, k = 1600.0, 0.6
        z = np.linspace(50, 3500, 200)
        twt = 2000.0 / k * np.log1p(k * z / v0)
        m = fit_linear_velocity(z, twt)
        self.assertAlmostEqual(m.v0, v0, delta=0.5)
        self.assertAlmostEqual(m.k, k, delta=1e-3)
        np.testing.assert_allclose(m.vint(np.array([1000.0])), v0 + k * 1000.0, rtol=1e-4)

    def test_non_physical_slope_separated_from_undefined(self):
        z = np.array([100.0, 200, 300, 400])
        v = np.array([2000.0, np.nan, np.nan, 2100])
        slope = np.array([1.0, -0.5, np.nan, 0.9])     # second sample: TWT decreases with depth
        q = velocity_qc(z, v, slope=slope)
        self.assertEqual(q["n_non_physical_slope"], 1)
        self.assertEqual(q["n_undefined_insufficient_control"], 1)
        self.assertFalse(q["monotonic_twt"])

    def test_reversed_table_is_flagged(self):
        tab = TableTZ("bad", [0, 100, 200, 300], [0, 100, 90, 250], "test")
        self.assertTrue(any("not_strictly_increasing" in f for f in tab.flags))

    def test_lowo_identical_wells_zero_error(self):
        z = np.linspace(0, 3000, 61)
        twt = 2000.0 / 0.5 * np.log1p(0.5 * z / 1700.0)
        lo = leave_one_well_out({"a": (z, twt), "b": (z, twt)},
                                fitter=lambda zz, tt, ww, fw: fit_linear_velocity(zz, tt, ww, fit_wells=fw))
        self.assertTrue((lo["rms_ms"] < 0.05).all())

    def test_extrapolation_turnover_detected(self):
        # cubic whose slope rises then the velocity turns over below 3000 m
        bad = PolynomialTZ("p", coeffs=[1500.0, 600.0, -150.0], scale=3000.0, z_support=(0, 3000), provenance="t")
        chk = physical_extrapolation_check(bad, 4500.0)
        self.assertFalse(chk["physically_valid"])
        good = fit_linear_velocity(np.linspace(50, 3000, 50), 2000 / 0.7 * np.log1p(0.7 * np.linspace(50, 3000, 50) / 1600))
        self.assertTrue(physical_extrapolation_check(good, 4500.0)["physically_valid"])


if __name__ == "__main__":
    unittest.main()
