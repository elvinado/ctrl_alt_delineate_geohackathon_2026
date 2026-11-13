"""SEG-Y geometry/decoding and well-file readers with known answers.  Software tests only."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from _fixtures import rotated_grid, write_deviation, write_las, write_segy, write_tdr

import numpy as np

from geogg_fresh.io_wells import (FileIndex, classify_curve, curve_statistics, parse_deviation,
                                  parse_tdr_table, read_las, read_well_header)
from geogg_fresh.segy import SegyFile, SurveyGeometry, ibm2ieee, ieee2ibm


class SegyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        ils_axis, xls_axis = list(range(100, 121, 2)), list(range(200, 216))
        self.drop = {(120, 200), (120, 201), (100, 215)}
        self.ils, self.xls, self.xs, self.ys = rotated_grid(ils_axis, xls_axis, drop=self.drop)
        rng = np.random.default_rng(0)
        self.data = rng.normal(size=(self.ils.size, 50)).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ibm_known_values(self):
        words = np.array([0x42640000, 0xC276A000, 0x00000000], dtype=np.uint32)
        np.testing.assert_allclose(ibm2ieee(words), [100.0, -118.625, 0.0])
        np.testing.assert_array_equal(ieee2ibm(np.array([100.0, -118.625, 0.0])), words)
        vals = np.random.default_rng(1).normal(scale=1e3, size=1000)
        np.testing.assert_allclose(ibm2ieee(ieee2ibm(vals)), vals, rtol=1e-6)

    def _check(self, fmt):
        path = self.tmp / f"t{fmt}.sgy"
        write_segy(path, self.data, self.ils, self.xls, self.xs, self.ys, dt_us=4000, fmt=fmt, delay_ms=8)
        with SegyFile(path) as s:
            self.assertEqual((s.n_traces, s.ns, s.format_code), (self.ils.size, 50, fmt))
            g = s.geometry()
            self.assertEqual((g.il_min, g.il_max, g.il_step, g.xl_min, g.xl_max, g.xl_step),
                             (100, 120, 2, 200, 215, 1))
            self.assertEqual((g.t0_ms, g.dt_ms, g.ns), (8.0, 4.0, 50))
            self.assertEqual(g.n_live, self.ils.size)
            self.assertLess(g.affine_residual_max, 0.02)
            for (il, xl) in self.drop:
                self.assertEqual(g.trace_index(il, xl)[0], -1)
            k = 37
            idx = g.trace_index(self.ils[k], self.xls[k])[0]
            self.assertEqual(idx, k)
            tol = 0 if fmt == 5 else 1e-6
            np.testing.assert_allclose(s.read_trace(idx), self.data[k], rtol=tol, atol=tol)
            il_f, xl_f = g.xy_to_ilxl(self.xs[k], self.ys[k])
            self.assertAlmostEqual(float(il_f), self.ils[k], places=4)
            self.assertAlmostEqual(float(xl_f), self.xls[k], places=4)
            along, between = g.bin_spacing()
            self.assertAlmostEqual(along, 25.0, places=3)
            self.assertAlmostEqual(between, 50.0, places=3)       # IL step 2 x 25 m
            section, xls = s.read_inline(g, 120)
            self.assertTrue(np.isnan(section[0]).all())          # (120, 200) dropped
            # extraction along a path through trace k at a sample time and half-way between samples
            ext = s.extract_along_path(g, [self.xs[k], self.xs[k]], [self.ys[k], self.ys[k]], [8 + 4 * 10, 8 + 4 * 10 + 2])
            self.assertAlmostEqual(ext["values"][0], float(self.data[k, 10]), places=5)
            self.assertAlmostEqual(ext["values"][1], float((self.data[k, 10] + self.data[k, 11]) / 2), places=5)
            self.assertLess(float(ext["offset_xy"].max()), 1e-3)
            g.save_npz(self.tmp / "geom.npz")
        back = SurveyGeometry.load_npz(self.tmp / "geom.npz")
        self.assertEqual(back.trace_index(self.ils[k], self.xls[k])[0], k)
        os.remove(path)          # handle released (matters on Windows)

    def test_ieee_volume(self):
        self._check(5)

    def test_ibm_volume(self):
        self._check(1)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_deviation_header_order_and_nulls(self):
        p = self.tmp / "w_Deviation.txt"
        write_deviation(p, "w", [0, 100, 200, 300], [0, 45, 45, -999.25], [0, 5, 10, -999.25])
        dev = parse_deviation(p, "w")
        self.assertEqual(dev.meta["column_order"], ["md", "azim", "incl"])
        self.assertEqual(dev.meta["n_null_rows"], 1)
        self.assertEqual(dev.stations["incl_deg"].tolist(), [0, 5, 10, 10])   # held from previous station
        self.assertTrue(dev.stations["null_filled"].iloc[-1])

    def test_deviation_without_names_uses_flagged_default(self):
        p = self.tmp / "x.txt"
        p.write_text("0 0 0\n100 5 45\n")
        dev = parse_deviation(p, "x")
        self.assertIn("column_order_assumed", dev.meta["flags"])

    def test_tdr_table_provenance_and_pattern(self):
        p = self.tmp / "w_CS.txt"
        z = np.arange(0, 1001, 100.0)
        write_tdr(p, "w", z + 25, 0.8 * z, z)
        t = parse_tdr_table(p, "w")
        self.assertEqual(t.meta["generated_by"], "test_fixture")
        self.assertEqual(list(t.data.columns), ["md_kb_m", "twt_ms", "tvdss_m"])
        self.assertTrue(t.meta["sampling_pattern"].startswith("regular"))
        self.assertAlmostEqual(t.meta["implied_datum_elevation_m"], 25.0)

    def test_builtin_las_reader_and_families(self):
        p = self.tmp / "w.las"
        depth = np.arange(1000, 1010, 0.5)
        vsh = np.linspace(0, 1, depth.size)
        vsh[3] = np.nan
        write_las(p, "w", depth, {"GR": ("API", np.linspace(20, 120, depth.size)), "VSH": ("V/V", vsh),
                                  "PS_GIP": ("", np.r_[np.zeros(10), np.ones(10)])})
        las = read_las(p, "w")
        self.assertEqual(las.depth_unit, "m")
        self.assertTrue(np.isnan(las.data["VSH"].iloc[3]))
        self.assertEqual([classify_curve(c["mnemonic"]) for c in las.curves[1:]],
                         ["GR", "VSH", "HC_INDICATOR_GAS"])
        stats = curve_statistics(las.data["PS_GIP"].to_numpy(), "HC_INDICATOR_GAS")
        self.assertAlmostEqual(stats["fraction_positive"], 0.5)
        pct = curve_statistics(np.linspace(0, 100, 50), "PHIT")
        self.assertIn("PERCENT", pct["notes"])

    def test_header_flags_and_file_index_priority(self):
        p = self.tmp / "h.csv"
        p.write_text("No.,Well Name,RT,TD,X,Y,Checkshots,Las\n1,A-1,22.5,3000,1,2,X,/\n")
        df, _ = read_well_header(p)
        self.assertEqual(df.loc[0, "well"], "a-1")
        self.assertEqual((df.loc[0, "flag_checkshot"], df.loc[0, "flag_las"]), ("not_available", "available"))
        (self.tmp / "dev" / "Dev2 for Pet").mkdir(parents=True)
        (self.tmp / "dev" / "a-1_Deviation.txt").write_text("x")
        (self.tmp / "dev" / "Dev2 for Pet" / "a-1_Deviation.txt").write_text("y")
        hits = FileIndex([self.tmp / "dev"]).find("a-1", ["{well}_Deviation.txt"])
        self.assertEqual(hits[0].parent.name, "dev")
        self.assertEqual(len(hits), 2)

    def test_header_row_below_title_rows_and_clear_error(self):
        import pandas as pd

        from geogg_fresh.io_wells import WellDataError

        p = self.tmp / "h.xlsx"
        with pd.ExcelWriter(p) as w:
            pd.DataFrame([["notes"]]).to_excel(w, sheet_name="About", header=False, index=False)
            pd.DataFrame([["Well list - edited", None, None], [None, None, None], ["WELL_NAME", "RT", "TD"],
                          ["A-1", 22.5, 3000], ["B-2", 25.0, 3100]]).to_excel(w, sheet_name="Wells", header=False,
                                                                              index=False)
        df, meta = read_well_header(p)
        self.assertEqual((meta["sheet"], meta["header_row"]), ("Wells", 2))
        self.assertEqual(df["well"].tolist(), ["a-1", "b-2"])
        self.assertEqual(df["rt_m"].tolist(), [22.5, 25.0])
        q = self.tmp / "bad.xlsx"
        pd.DataFrame([["Name", "RT"], ["A-1", 1]]).to_excel(q, header=False, index=False)
        with self.assertRaises(WellDataError) as cm:
            read_well_header(q)
        self.assertIn("first non-empty row ['Name', 'RT']", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
