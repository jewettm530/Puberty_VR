"""Regression checks for the shared two-second eyes-open EEG pipeline.

Run from the project root:
    MPLBACKEND=Agg python3 -m unittest discover -s tests -v
"""
from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from scipy import signal

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS / "eeg"))
import preprocessing
import process_all_eeg


def recording(path, seconds=12, *, offset=725., spike=None, gap=False, low_rate=False):
    fs = 128 if low_rate else 256
    t = np.arange(int(seconds*fs))/fs
    theta = np.where(t < seconds/2, 50., 10.)
    x = theta*np.sin(2*np.pi*6*t) + 20*np.sin(2*np.pi*10*t) + 8*np.sin(2*np.pi*21*t)
    second = x.copy()
    if spike is not None:
        x[int(spike*fs)] = 350.
    if gap:
        t[t >= seconds/2] += 1.
    frame = pd.DataFrame({
        "TimeStamp":pd.Timestamp("2026-01-01") + pd.to_timedelta(t, unit="s"),
        "RAW_AF7":x+offset, "RAW_AF8":second+offset})
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("recording.csv",frame.to_csv(index=False))
    else:
        frame.to_csv(path,index=False)
    return frame


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/"scripts").mkdir()
        shutil.copytree(SCRIPTS/"eeg",self.root/"scripts/eeg")
        (self.root/"data/raw/subject_data").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def path(self, sid="039", phase="pre_vr", zipped=False):
        return self.root/f"data/raw/subject_data/sub{sid}/{sid}_{phase}_eeg.{ 'zip' if zipped else 'csv'}"

    def command(self, name, *args):
        env = {**os.environ, "MPLBACKEND":"Agg"}
        result = subprocess.run([sys.executable,str(self.root/f"scripts/eeg/{name}.py"),*args],
                                cwd=self.root,env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+"\n"+result.stderr)
        return result.stdout

    def test_fixed_welch_and_no_short_epoch(self):
        sample = np.sin(2*np.pi*10*np.arange(512)/256)
        frequencies, actual = preprocessing.compute_psd(sample)
        f, expected = signal.welch(sample,fs=256,window=signal.windows.hann(512,sym=False),
                                  nperseg=512,noverlap=256,nfft=1024,detrend=False,scaling="density")
        np.testing.assert_array_equal(frequencies,f)
        np.testing.assert_allclose(actual,expected)
        self.assertEqual(frequencies[1],.25)
        with self.assertRaises(preprocessing.IneligibleRecording):
            preprocessing.compute_psd(sample[:-1])
        for length in (2.,3.,4.):
            p=self.path("038")
            recording(p,length)
            result=preprocessing.process_recording(p)
            self.assertTrue((result["windows"].stop_index-result["windows"].window_index).eq(512).all())

    def test_eyes_open_range_inclusive_and_all_phases(self):
        for sid in ("038","039","052","053"):
            for phase in ("baseline","pre_vr","post_vr","recovery"):
                p=self.path(sid,phase)
                recording(p,8)
                result=preprocessing.process_recording(p)
                expected=4. if sid in ("039","052") else 0.
                self.assertAlmostEqual(result["metadata"]["analysis_start_s"],expected)
                self.assertGreaterEqual(result["windows"].start_s.min(),expected)
        recording(self.path("039"),3.)
        with self.assertRaises(preprocessing.IneligibleRecording):
            preprocessing.process_recording(self.path("039"))

    def test_amplitude_mask_and_signed_input(self):
        p=self.path()
        recording(p,12,spike=8.)
        result=preprocessing.process_recording(p)
        qc=result["windows"]
        self.assertTrue(qc[(qc.channel=="AF7") & (qc.start_s<=8.) & (qc.end_s>8.)].amplitude_rejected.all())
        self.assertFalse(qc.loc[qc.channel=="AF8","amplitude_rejected"].any())
        self.assertTrue(qc.loc[qc.amplitude_rejected,"theta"].isna().all())
        signed=self.path("038")
        recording(signed,4.,offset=0.)
        self.assertTrue(preprocessing.process_recording(signed,raw_offset_uv=0.)["windows"].hard_pass.all())

    def test_gaps_no_interpolation_and_selected_only_filtering(self):
        p=self.path("038")
        frame=recording(p,8,gap=True)
        frame.loc[300,"RAW_AF7"]=np.nan
        frame.to_csv(p,index=False)
        result=preprocessing.process_recording(p)
        qc=result["windows"]
        self.assertTrue(qc[(qc.channel=="AF7") & (qc.window_index<=300) & (qc.stop_index>300)].hard_pass.eq(False).all())
        self.assertFalse(((qc.start_s<4) & (qc.end_s>5)).any())
        self.assertTrue(np.isnan(result["filtered_uv"][300,0]))
        # No filtering of an eyes-closed artifact into the selected eyes-open half.
        p=self.path()
        recording(p,8,spike=3.99)
        cropped=preprocessing.process_recording(p)
        self.assertTrue(cropped["windows"].hard_pass.all())

    def test_pipeline_cli_npz_psd_manual_and_collection(self):
        p=self.path()
        recording(p,12,spike=8.)
        recording(self.path("052","post_vr",zipped=True),12)
        recording(self.path("020"),12,low_rate=True)
        self.command("EEG_Audit","--no-preprocessing-figures","--skip-review-figures")
        bundle=self.root/"data/processed/eeg/sub039/039_pre_vr_eeg_Preprocessed.npz"
        data=preprocessing.load_preprocessed(bundle)
        self.assertAlmostEqual(data["times"][0],6.)
        self.command("plot_eeg_noise_comparison")
        coverage=pd.read_csv(self.root/"outputs/eeg/noise_comparison/recording_coverage.csv")
        count=int(coverage.loc[coverage.recording.str.contains("039_pre"),"af7_psd_windows"].iloc[0])
        self.assertEqual(count,int(data["windows"].loc[data["windows"].channel=="AF7","hard_pass"].astype(str).str.lower().eq("true").sum()))
        folder=self.root/"outputs/eeg/bandpower/sub_039"
        self.assertTrue((folder/"039_pre_vr_eeg_Spectrogram.png").is_file())
        self.assertTrue((folder/"039_pre_vr_eeg_PSD.csv").is_file())
        self.assertTrue((folder/"039_pre_vr_eeg_PSD.png").is_file())
        rows=pd.read_csv(folder/"039_pre_vr_eeg_FrontalLogPower.csv")
        self.assertEqual(len(rows),6)
        psd=pd.read_csv(folder/"039_pre_vr_eeg_PSD.csv")
        f=psd[psd.channel=="AF8"]
        alpha=preprocessing.band_power(f.frequency_hz.to_numpy(),
                                       f.psd_uv2_per_hz.to_numpy()[:,None],8,13)[0]
        self.assertAlmostEqual(alpha,rows[(rows.channel=="AF8") & (rows.band=="alpha")].power.iloc[0])
        # Simulate a human mark exported from an already preprocessed recording.
        manual=self.root/"outputs/eeg/manual_annotations/sub039"
        manual.mkdir(parents=True)
        pd.DataFrame([{"preprocessing_id":data["metadata"]["preprocessing_id"],
                       "label":"BAD_muscle","affected_channels":"AF8",
                       "recording_start_s":10.1,"recording_end_s":10.2}]).to_csv(
                           manual/"039_pre_vr_eeg_AF8_processed_test_intervals.csv",index=False)
        self.command("EEG_Audit","--no-preprocessing-figures","--skip-review-figures")
        after=preprocessing.load_preprocessed(bundle)
        self.assertEqual(data["metadata"]["preprocessing_id"],after["metadata"]["preprocessing_id"])
        self.assertTrue(after["windows"].manual_rejected.astype(str).str.lower().eq("true").any())
        self.command("process_all_eeg")
        long=pd.read_csv(self.root/"data/analysis_data/eeg/eeg_frontal_log_bandpower_long.csv")
        self.assertEqual(len(long),12)
        self.assertTrue(long.pipeline_version.eq(preprocessing.VERSION).all())
        report=pd.read_csv(self.root/"outputs/eeg/audit/eeg_recording_qc.csv")
        self.assertEqual(report.loc[report.recording.str.startswith("020"),"status"].iloc[0],"SKIPPED_RATE_OR_LENGTH")

    def test_manual_reviewer_uses_bundle_and_exports_original_time(self):
        p=self.path()
        recording(p,12)
        self.command("EEG_Audit","--no-preprocessing-figures","--skip-review-figures")
        import mne
        import EEG_Manual_Review as reviewer
        def annotate(raw, **kwargs):
            # A mark at display second 1 maps to original second 7 after crop.
            raw.annotations.append(1.,.25,"BAD_blink")
        args=["review","--subject","039","--phase","pre_vr","--channel","AF7"]
        with patch.object(reviewer,"project_dir",return_value=self.root), \
             patch.object(sys,"argv",args), \
             patch.object(mne.io.BaseRaw,"plot",annotate):
            reviewer.main()
        tables=list((self.root/"outputs/eeg/manual_annotations/sub039").glob("*_intervals.csv"))
        self.assertEqual(len(tables),1)
        row=pd.read_csv(tables[0]).iloc[0]
        self.assertAlmostEqual(row.recording_start_s,7.,places=3)
        self.assertAlmostEqual(row.recording_end_s,7.25,places=3)
        self.assertEqual(row.affected_channels,"AF7")
        self.command("EEG_Audit","--no-preprocessing-figures","--skip-review-figures")
        result=preprocessing.load_preprocessed(self.root/"data/processed/eeg/sub039/039_pre_vr_eeg_Preprocessed.npz")
        self.assertTrue(result["windows"].manual_rejected.astype(str).str.lower().eq("true").any())

    def test_psd_png_matches_csv_and_combined_tiff(self):
        from matplotlib.figure import Figure
        p=self.path()
        recording(p,8,spike=6.)
        result=preprocessing.process_recording(p)
        folder=self.root/"figure_check"
        curves=[]
        original_save=Figure.savefig
        def capture(fig, path, *args, **kwargs):
            if str(path).endswith("_PSD.png"):
                axis = next(ax for ax in fig.axes if ax.get_title()=="Notch-filtered EEG PSD")
                curves.extend(line.get_ydata().copy() for line in axis.lines)
                titles=[ax.get_title() for ax in fig.axes[:8]]
                self.assertEqual(titles[:2],["EEG Power Spectral Density","Notch-filtered EEG PSD"])
            return original_save(fig,path,*args,**kwargs)
        with patch.object(Figure,"savefig",capture):
            preprocessing.export_outputs(result,p,folder,save_figure=True)
        table=pd.read_csv(folder/(p.stem+"_PSD.csv"))
        for i,channel in enumerate(preprocessing.CHANNELS):
            with np.errstate(divide="ignore",invalid="ignore"):
                expected=10*np.log10(table.loc[table.channel==channel,"psd_uv2_per_hz"])
            np.testing.assert_allclose(curves[i],expected,atol=1e-8)
        from PIL import Image
        for suffix,kind in (("_PSD.png","PNG"),("_Spectrogram.png","PNG"),(".tiff","TIFF")):
            with Image.open(folder/(p.stem+suffix)) as image:
                self.assertEqual(image.format,kind)
                image.verify()

    def test_original_four_channel_movement_layout(self):
        from matplotlib.figure import Figure
        p=self.path()
        frame=recording(p,12)
        # Include all channels and AUX in the same input order as the reference.
        t=np.arange(len(frame))/256
        frame.insert(1,"RAW_TP9",725+15*np.sin(2*np.pi*8*t))
        frame["RAW_TP10"]=725+12*np.sin(2*np.pi*12*t)
        for i in range(1,5):
            frame[f"AUX_{i}"]=np.sin(2*np.pi*i*t)
        frame.to_csv(p,index=False)
        result=preprocessing.process_recording(p)
        self.assertEqual(result["eeg_names"],["RAW_TP9","RAW_AF7","RAW_AF8","RAW_TP10"])
        self.assertEqual(set(result["windows"].channel),{"AF7","AF8"})
        fig=preprocessing.plot_diagnostics(result)
        titles=[ax.get_title() for ax in fig.axes[:8]]
        self.assertEqual(titles,["EEG Power Spectral Density","Notch-filtered EEG PSD",
                                 "Movement PSD","","RAW_TP9","RAW_AF7","RAW_AF8","RAW_TP10"])
        self.assertEqual(len(fig.axes[0].lines),4)
        self.assertEqual(len(fig.axes[1].lines),4)
        self.assertEqual(len(fig.axes[2].lines),4)
        self.assertEqual(len(fig.axes[3].lines),1)
        # Match the original movement magnitude formula on the selected half.
        aux=result["aux"]
        expected=np.sqrt(np.sum(signal.detrend(aux,axis=0,type="linear")**2,axis=1))
        np.testing.assert_allclose(fig.axes[3].lines[0].get_ydata(),expected)
        import matplotlib.pyplot as plt
        plt.close(fig)
        _,_,matrix=preprocessing.export_outputs(result,p,self.root/"full_figure",save_figure=True)
        self.assertEqual(matrix.shape,(6,4))

    def test_short_rerun_clears_aggregates_and_old_reviews_are_ignored(self):
        p=self.path()
        recording(p,8)
        self.command("EEG_Audit","--no-preprocessing-figures","--skip-review-figures")
        manual=self.root/"outputs/eeg/manual_annotations/sub039"
        manual.mkdir(parents=True)
        pd.DataFrame([{"label":"BAD_blink","recording_start_s":4.5,
                       "recording_end_s":5.}]).to_csv(
                           manual/"039_pre_vr_eeg_ALL_span1_intervals.csv",index=False)
        result=preprocessing.process_recording(p,manual_dir=manual)
        self.assertEqual(result["metadata"]["manual_bad_intervals"],0)
        self.assertEqual(len(result["metadata"]["ignored_annotation_files"]),1)
        # Now selected half is too short. Previous exports must not enter tables.
        recording(p,3)
        self.command("EEG_Audit","--no-preprocessing-figures","--skip-review-figures")
        self.command("process_all_eeg")
        self.assertTrue(pd.read_csv(self.root/"data/analysis_data/eeg/eeg_frontal_log_bandpower_wide.csv").empty)
        self.assertTrue(pd.read_csv(self.root/"outputs/eeg/audit/eeg_final_candidate_band_powers.csv").empty)

    def test_all_rejected_and_stale_exports(self):
        p=self.path()
        frame=recording(p,8)
        frame["RAW_AF7"]=1000+np.sin(np.arange(len(frame)))
        frame["RAW_AF8"]=1000+np.sin(np.arange(len(frame)))
        frame.to_csv(p,index=False)
        self.command("EEG_Audit","--no-preprocessing-figures","--skip-review-figures")
        self.command("process_all_eeg")
        report=pd.read_csv(self.root/"data/analysis_data/eeg/eeg_batch_processing_report.csv")
        self.assertEqual(report.summary_status.iloc[0],"NO_ACCEPTED_WINDOWS")
        folder=self.root/"outputs/eeg/bandpower/sub_039/039_pre_vr_eeg_FrontalLogPower.csv"
        data=pd.read_csv(folder)
        data["pipeline_version"]="legacy"
        data.to_csv(folder,index=False)
        self.command("process_all_eeg")
        report=pd.read_csv(self.root/"data/analysis_data/eeg/eeg_batch_processing_report.csv")
        self.assertEqual(report.summary_status.iloc[0],"ERROR")
        long=pd.read_csv(self.root/"data/analysis_data/eeg/eeg_frontal_log_bandpower_long.csv")
        self.assertTrue(long.empty)


if __name__ == "__main__":
    unittest.main()
