"""Data integrity and validation-leakage checks for the V60 pipeline."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducers import v60_consistent_streams as model


class FundingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.reg = pd.DataFrame({
            "abitur_kod": [100, 100, 101, 102, 103, 103, 104],
            "zayav_type_kod": [1, 2, 1, 1, 1, 1, 1],
            "reg_ts": ["2019-07-01 10:00:00"]*6 + ["2020-07-01 10:00:00"],
        })
        self.ks = pd.DataFrame({
            "abitur_kod": [100, 100, 101, 101, 103, 104],
            "zayav_type_kod": [1, 2, 1, 1, 1, 1],
            "reg_ts": ["2019-07-01 11:00:00"]*5 + ["2020-07-01 11:00:00"],
            "ob_osnova_name": ["Бюджет", "Договор", "Договор", "Бюджет", "Гослиния", "Договор"],
        })

    def test_degree_keys_duplicates_and_missing_rows(self):
        counts, audit = model.funding_counts(self.reg, self.ks, 2020)
        # 100's two degrees stay distinct; 101's two options do not duplicate
        # its registration; both 103 registrations are preserved; 102 is missing.
        self.assertEqual(counts.iloc[0].to_dict(), {"budget": 4, "contract": 2})
        self.assertEqual(audit["training_registrations"], 6)
        self.assertEqual(audit["unmatched_registrations"], 1)

    def test_competition_row_order_cannot_change_stream(self):
        expected, _ = model.funding_counts(self.reg, self.ks, 2020)
        actual, _ = model.funding_counts(self.reg, self.ks.sample(frac=1, random_state=7), 2020)
        pd.testing.assert_frame_equal(actual, expected)

    def test_target_and_future_records_cannot_change_partition(self):
        expected, audit = model.funding_counts(self.reg, self.ks, 2020)
        r, k = self.reg.copy(), self.ks.copy()
        r.loc[r.reg_ts.str.startswith("2020"), "abitur_kod"] = 999
        k.loc[k.reg_ts.str.startswith("2020"), "ob_osnova_name"] = "Бюджет"
        actual, changed_audit = model.funding_counts(r, k, 2020)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(changed_audit, audit)


class EndToEndInputsTests(unittest.TestCase):
    def test_removing_all_validation_labels_preserves_training_and_test_features(self):
        # Integration check with the real files: even the held-out year is
        # completely absent from both sources in the second construction.
        year = 2023
        original, _ = model.make_grid(year, model.nominal_ends(year))
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for source in (model.base.REG_FILE, model.base.KS_FILE):
                data = pd.read_csv(source)
                data = data[pd.to_datetime(data.reg_ts).dt.year < year]
                target = Path(directory) / source.name
                data.to_csv(target, index=False)
                paths.append(target)
            scrubbed, _ = model.make_grid(year, model.nominal_ends(year), *paths)
        pd.testing.assert_frame_equal(original[original.year < year], scrubbed[scrubbed.year < year])
        pd.testing.assert_frame_equal(
            original.loc[original.year == year, model.base.FEATURE_COLUMNS],
            scrubbed.loc[scrubbed.year == year, model.base.FEATURE_COLUMNS])
        tr = original[original.year < year]
        self.assertEqual(model.base.determine_closed_days(tr, original[original.year == year]),
                         model.base.determine_closed_days(tr, scrubbed[scrubbed.year == year]))


if __name__ == "__main__":
    unittest.main()
