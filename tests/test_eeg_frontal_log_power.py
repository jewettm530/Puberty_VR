"""Integration/regression tests for frontal log-bandpower exports.

Run: python3 -m unittest discover -s tests -v
"""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from eeg import EEG_Audit, preprocessing, process_all_eeg


def synthetic_recording(path, *, amplitude=1.0, sampling_rate=256):
    n = 256 * 12
    t = np.arange(n) / sampling_rate
    # Distinct theta, alpha, beta components with nonzero power in each band.
    s = amplitude * (8*np.sin(2*np.pi*6*t) + 12*np.sin(2*np.pi*10*t) +
                     5*np.sin(2*np.pi*21*t))
    timestamps = pd.Timestamp('2026-01-01') + pd.to_timedelta(t, unit='s')
    df = pd.DataFrame({'TimeStamp': timestamps, 'RAW_AF7': s,
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
            _, _, matrix1 = preprocessing.analyze(f1, show_figure=False, save_figure=False)
            _, _, matrix2 = preprocessing.analyze(f2, show_figure=False, save_figure=False)
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
            folders = root / 'data' / 'raw' / 'subject_data'
            valid = folders / 'sub001'
            low_rate = folders / 'sub002'
            valid.mkdir(parents=True)
            low_rate.mkdir()
            synthetic_recording(valid / '001_baseline_eeg.csv')
            synthetic_recording(valid / '001_pre_vr_eeg.csv')
            synthetic_recording(valid / '001_post_vr_eeg.zip')
            synthetic_recording(valid / '001_recovery_eeg.csv')
            synthetic_recording(low_rate / '002_pre_vr_eeg.csv', sampling_rate=128)

            scripts = root / 'scripts' / 'eeg'
            scripts.mkdir(parents=True)
            shutil.copyfile(preprocessing.__file__, scripts / 'preprocessing.py')
            audit_path = scripts / 'EEG_Audit.py'

            def refresh():
                with patch.object(EEG_Audit, '__file__', str(audit_path)), \
                     patch.object(EEG_Audit, 'PREPROCESSOR', None), \
                     patch.object(sys, 'argv', [str(audit_path), '--skip-review-figures']), \
                     patch('matplotlib.figure.Figure.savefig'):
                    EEG_Audit.main()
                self.assertEqual(process_all_eeg.collect(root), 0)

            refresh()
            audit = pd.read_csv(root / 'outputs/eeg/audit/eeg_recording_qc.csv')
            statuses = audit.set_index('recording')['preprocessing_status']
            self.assertEqual(statuses.eq('COMPLETED').sum(), 4)
            self.assertEqual(statuses.loc['002_pre_vr_eeg.csv'], 'SKIPPED_LOW_RATE')
            analysis = root / 'data/analysis_data/eeg'
            report = pd.read_csv(analysis / 'eeg_batch_processing_report.csv', dtype={'subject_id': str})
            self.assertEqual(report.summary_status.tolist(), ['COLLECTED'] * 4)
            long = pd.read_csv(analysis / 'eeg_frontal_log_bandpower_long.csv', dtype={'subject_id': str})
            wide_path = analysis / 'eeg_frontal_log_bandpower_wide.csv'
            wide = pd.read_csv(wide_path, dtype={'subject_id': str})
            self.assertEqual(len(long), 24)
            self.assertEqual(set(long.timepoint), {'baseline', 'anticipation', 'acute_stress_response', 'recovery'})
            self.assertEqual(wide.shape, (1, 25))  # subject ID + 4 phases × 2 channels × 3 bands
            self.assertEqual(wide.subject_id.tolist(), ['001'])
            self.assertTrue(np.isfinite(wide.iloc[:, 1:].to_numpy(dtype=float)).all())
            self.assertTrue(long.loc[long.timepoint.eq('acute_stress_response'),
                                     'measurement_timing'].eq('after_vr_task').all())

            # Earlier exports still exist, but the refreshed audit omits deleted inputs.
            (valid / '001_recovery_eeg.csv').unlink()
            (valid / '001_post_vr_eeg.zip').unlink()
            refresh()
            refreshed = pd.read_csv(wide_path)
            self.assertTrue(pd.isna(refreshed.loc[0, 'recovery_af7_theta_log10_power']))
            self.assertTrue(pd.isna(refreshed.loc[0, 'acute_stress_response_af7_theta_log10_power']))



if __name__ == '__main__':
    unittest.main()
