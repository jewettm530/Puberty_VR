"""Integration/regression tests for the three-timepoint frontal log-bandpower exports.

Run: python3 -m unittest discover -s tests -v
"""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from PreprocessingScript_v1 import analyze
import eeg.process_all_eeg as batch


def synthetic_recording(path, *, amplitude=1.0, sampling_rate=256):
    n = 256 * 12
    t = np.arange(n) / sampling_rate
    # Distinct theta, alpha, beta components with nonzero power in each band.
    s = amplitude * (8*np.sin(2*np.pi*6*t) + 12*np.sin(2*np.pi*10*t) +
                     5*np.sin(2*np.pi*21*t))
    df = pd.DataFrame({'TimeStamp': 1000*t, 'RAW_AF7': s,
                       'RAW_AF8': 0.7*s + amplitude*np.sin(2*np.pi*26*t),
                       'AUX_1': np.sin(t)})
    if path.suffix == '.zip':
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('recording.csv', df.to_csv(index=False))
    else:
        df.to_csv(path, index=False)


class FrontalBandpowerTest(unittest.TestCase):
    def test_log10_of_mean_power_and_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f1, f2 = root/'one.csv', root/'two.csv'
            synthetic_recording(f1, amplitude=1)
            synthetic_recording(f2, amplitude=2)
            _, _, matrix1 = analyze(f1, show_figure=False, save_figure=False)
            _, _, matrix2 = analyze(f2, show_figure=False, save_figure=False)
            a = pd.read_csv(root/'one_FrontalLogPower.csv')
            b = pd.read_csv(root/'two_FrontalLogPower.csv')
            self.assertEqual(len(a), 6)
            self.assertEqual(set(a.channel), {'AF7','AF8'})
            self.assertEqual(set(a.band), {'theta','alpha','beta'})
            self.assertTrue(np.allclose(np.log10(a.power), a.log10_power))
            self.assertTrue(np.allclose(b.log10_power - a.log10_power, np.log10(4), atol=1e-5))
            self.assertTrue(np.allclose(matrix1, matrix2, atol=1e-5, rtol=1e-5))

    def test_phase_tables_zip_and_qc_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folders = root/'subjects'
            (folders/'sub001').mkdir(parents=True)
            (folders/'sub002').mkdir(parents=True)
            synthetic_recording(folders/'sub001'/'001_pre_vr_eeg.csv')
            synthetic_recording(folders/'sub001'/'001_post_vr_eeg.zip')
            synthetic_recording(folders/'sub001'/'001_recovery_eeg.csv')
            # Known rate mismatch must be excluded and not present in wide output.
            synthetic_recording(folders/'sub002'/'002_pre_vr_eeg.csv', sampling_rate=128)
            old_data, old_outputs = batch.EEG_ANALYSIS_DATA_DIR, batch.EEG_OUTPUTS_DIR
            try:
                batch.EEG_ANALYSIS_DATA_DIR = root/'analysis'
                batch.EEG_OUTPUTS_DIR = root/'outputs'
                self.assertEqual(batch.run_batch(folders, save_plots=False), 0)
            finally:
                batch.EEG_ANALYSIS_DATA_DIR, batch.EEG_OUTPUTS_DIR = old_data, old_outputs
            report = pd.read_csv(root/'analysis'/'eeg_batch_processing_report.csv', dtype={'subject_id': str})
            self.assertEqual(report.processing_status.tolist().count('processed'), 3)
            self.assertEqual(report.processing_status.tolist().count('skipped_qc'), 1)
            long = pd.read_csv(root/'analysis'/'eeg_frontal_log_bandpower_long.csv', dtype={'subject_id': str})
            wide = pd.read_csv(root/'analysis'/'eeg_frontal_log_bandpower_wide.csv', dtype={'subject_id': str})
            self.assertEqual(len(long), 18)
            self.assertEqual(set(long.timepoint), {'anticipation', 'acute_stress_response', 'recovery'})
            self.assertEqual(len(wide), 1)
            self.assertEqual(wide.subject_id.iloc[0], '001')
            self.assertEqual(wide.shape[1], 19)  # subject ID + 3 phases × 2 channels × 3 bands
            self.assertTrue(np.isfinite(wide.iloc[:, 1:].to_numpy(dtype=float)).all())
            self.assertTrue(long.loc[long.timepoint.eq('acute_stress_response'), 'measurement_timing'].eq('post_vr').all())
            # Rerunning with missing recordings replaces, not reuses, the old table.
            (folders/'sub001'/'001_recovery_eeg.csv').unlink()
            (folders/'sub001'/'001_post_vr_eeg.zip').unlink()
            try:
                batch.EEG_ANALYSIS_DATA_DIR = root/'analysis'
                batch.EEG_OUTPUTS_DIR = root/'outputs'
                batch.run_batch(folders, save_plots=False)
            finally:
                batch.EEG_ANALYSIS_DATA_DIR, batch.EEG_OUTPUTS_DIR = old_data, old_outputs
            refreshed = pd.read_csv(root/'analysis'/'eeg_frontal_log_bandpower_wide.csv')
            self.assertTrue(pd.isna(refreshed.loc[0,'recovery_af7_theta_log10_power']))
            self.assertTrue(pd.isna(refreshed.loc[0,'acute_stress_response_af7_theta_log10_power']))


if __name__ == '__main__':
    unittest.main()
