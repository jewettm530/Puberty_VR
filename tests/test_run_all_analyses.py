"""Regression tests for the master pipeline's skip and safety behavior.

Run: python3 -m unittest discover -s tests -v
Tests mock the heavy analyses and never change participant data.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_all_analyses as runner  # noqa: E402


class RunnerTests(unittest.TestCase):
    def test_numbers_file_not_accepted_as_xlsx(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "wrong.xlsx"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("Index/Document.iwa", "numbers data")
            with patch.object(runner, "AGE_WORKBOOK", p):
                ok, reason = runner.check_age_workbook()
            self.assertFalse(ok)
            self.assertIn("Apple Numbers", reason)

    def test_real_excel_classifications_and_missing_subjects(self):
        import pandas as pd
        from multimodal import compare_minors_to_older as age_module
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            workbook = folder / "Real_Subject_Data_ID.xlsx"
            multimodal = folder / "multimodal.csv"
            pd.DataFrame({
                "Group flag": ["N/A Minor"] * 5 + ["Approved adult"],
                "Subj #": ["901", "902", "903", "904", "905", "906"],
                "Age": [16, 16, 15, 17, 17, 19],
            }).to_excel(workbook, index=False)
            pd.DataFrame({"subject_id": ["901", "902", "903", "904", "905", "906"]}).to_csv(multimodal, index=False)
            with patch.object(runner, "AGE_WORKBOOK", workbook), \
                 patch.object(age_module, "MULTIMODAL_CSV", multimodal):
                ok, reason = runner.check_age_workbook()
                self.assertTrue(ok, reason)
                pd.DataFrame({"subject_id": ["901", "902", "903", "904", "905", "906", "907"]}).to_csv(multimodal, index=False)
                ok, reason = runner.check_age_workbook()
            self.assertFalse(ok)
            self.assertIn("907", reason)

    def test_dry_run_does_not_execute_or_write(self):
        with tempfile.TemporaryDirectory() as folder:
            status = Path(folder) / "status.json"
            marker = Path(folder) / "NOT_CURRENT.txt"
            with patch.object(runner, "STATUS_PATH", status), \
                 patch.object(runner, "AGE_STALE_MARKER", marker), \
                 patch.object(runner.subprocess, "run") as subprocess_run, \
                 patch.object(sys, "argv", ["run_all_analyses.py", "--dry-run"]):
                exit_code = runner.main()
            self.assertEqual(exit_code, 0)
            subprocess_run.assert_not_called()
            self.assertFalse(status.exists())
            self.assertFalse(marker.exists())

    def test_skips_invalid_age_and_records_status(self):
        with tempfile.TemporaryDirectory() as folder:
            status = Path(folder) / "status.json"
            marker = Path(folder) / "NOT_CURRENT.txt"
            with patch.object(runner, "STATUS_PATH", status), \
                 patch.object(runner, "AGE_STALE_MARKER", marker), \
                 patch.object(runner, "check_age_workbook", return_value=(False, "invalid Numbers file")), \
                 patch.object(runner.subprocess, "run") as subprocess_run, \
                 patch.object(sys, "argv", ["run_all_analyses.py", "--skip-eeg"]):
                exit_code = runner.main()
            self.assertEqual(exit_code, 0)
            self.assertTrue(marker.exists())
            result = json.loads(status.read_text())
            self.assertEqual(result["age_comparison"], "skipped")
            self.assertEqual(result["result"], "completed_with_age_skipped")
            invoked = [call.args[0][1] for call in subprocess_run.call_args_list]
            self.assertNotIn(str(SCRIPTS / runner.AGE_STEPS[0]), invoked)
            self.assertNotIn(str(SCRIPTS / "eeg/process_all_eeg.py"), invoked)
            self.assertIn(str(SCRIPTS / "multimodal/build_multimodal_dataset.py"), invoked)


if __name__ == "__main__":
    unittest.main()
