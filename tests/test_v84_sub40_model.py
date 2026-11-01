"""V84 artifact equivalence, frozen inputs and safe output behavior."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v84_refactored import DEFAULT_MANIFEST, digest, reproduce


class ReproductionTests(unittest.TestCase):
    def test_original_submission_reproduces_byte_for_byte(self):
        forecast, metadata = reproduce()
        reference = ROOT / "output/v84_audit/original_forecast_2024_v84.csv"
        self.assertEqual(forecast.to_csv(index=False).encode(), reference.read_bytes())
        self.assertFalse(metadata["independent_historical_model"])
        self.assertFalse(metadata["public_score_computed_locally"])
        self.assertTrue(metadata["contains_projected_score_as_input"])

    def test_cli_writes_only_requested_outputs(self):
        preserved = [ROOT / "forecast_2024.csv", ROOT / "forecast_2024_v84.csv"]
        downloads = Path.home() / "Downloads/forecast_2024.csv"
        if downloads.exists():
            preserved.append(downloads)
        before = {str(path): digest(path) for path in preserved}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result = directory / "prediction.csv"
            subprocess.run([sys.executable, str(ROOT / "v84_sub40_model.py"), "--output", str(result)],
                           cwd=directory, check=True, capture_output=True)
            self.assertEqual(digest(result), digest(ROOT / "forecast_2024_v84.csv"))
            self.assertEqual(set(path.name for path in directory.iterdir()), {"prediction.csv", "prediction.metadata.json"})
        self.assertEqual(before, {str(path): digest(path) for path in preserved})

    def test_corrupt_input_and_misaligned_dates_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            shutil.copytree(DEFAULT_MANIFEST.parent, snapshot)
            manifest_path = snapshot / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            entry = manifest["basis"][0]
            path = snapshot / entry["path"]
            frame = pd.read_csv(path)
            frame.loc[0, "date"] = "2024-06-19"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                reproduce(manifest_path)
            entry["sha256"] = digest(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "dates differ"):
                reproduce(manifest_path)


if __name__ == "__main__":
    unittest.main()
