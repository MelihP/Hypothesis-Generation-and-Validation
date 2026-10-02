import io
import unittest
import pandas as pd
from openpyxl import load_workbook
from agents.statistics import categorical_test, wilson_interval, descriptive_statistics
from agents.results import export_csv, export_excel, build_result, require_evidence


class TestStatistics(unittest.TestCase):
    def test_wilson_known_interval_and_invalid_denominator(self):
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, .40383153, places=6)
        self.assertAlmostEqual(high, .59616847, places=6)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)

    def test_chi_square_and_effect_are_computed(self):
        test = categorical_test([[30, 10], [10, 30]], independent=True)
        self.assertLess(test["p_value"], .001)
        self.assertGreater(test["cramers_v_corrected"], .4)
        self.assertAlmostEqual(test["proportion_difference"], .5)
        self.assertLess(test["difference_ci95"][0], .5)
        self.assertGreater(test["difference_ci95"][1], .5)

    def test_sparse_tables_and_independence(self):
        self.assertEqual(categorical_test([[1, 0], [0, 1]], independent=True)["method"], "Fisher exact testi")
        self.assertEqual(categorical_test([[1, 0, 1], [0, 1, 0]], independent=True)["status"], "insufficient")
        with self.assertRaises(ValueError):
            categorical_test([[10, 10], [10, 10]])
        with self.assertRaises(ValueError):
            categorical_test([[1.5, 2], [3, 4]], independent=True)

    def test_descriptives_exclude_ids(self):
        frame = pd.DataFrame({"user_id": [1, 2, 3], "volume": [2, 4, 6]})
        summary = descriptive_statistics(frame, [{"name": "user_id", "type": "number", "role": "identifier", "unit": "değer"},
                                                 {"name": "volume", "type": "number", "role": "metric", "unit": "adet"}])
        self.assertEqual(summary.iloc[0]["Sütun"], "volume")
        self.assertEqual(summary.iloc[0]["Ortalama"], 4)
        self.assertEqual(summary.iloc[0]["Standart sapma"], 2)

    def test_zero_and_missing_counts_do_not_support_synthesis(self):
        result = build_result("count", {"table": "t", "aggregates": [{"op": "count", "as": "n"}]}, "", [], ["n"], [(0,)])
        self.assertEqual(result["status"], "empty")
        with self.assertRaises(ValueError):
            require_evidence([result])

    def test_exports_round_trip_and_escape_formulas(self):
        frame = pd.DataFrame({"=label": ["=1+1", "Türkçe"], "n": [2, 4]})
        csv = export_csv(frame)
        self.assertIn("'=1+1", csv.decode("utf-8-sig"))
        book = load_workbook(io.BytesIO(export_excel(frame)))
        self.assertEqual(book.active["A1"].value, "'=label")
        self.assertEqual(book.active["A2"].value, "'=1+1")
        self.assertEqual(book.active["B3"].value, 4)
