# 2026-10-25 — AF7/AF8 frontal log-transformed EEG power

## Files changed

- `scripts/PreprocessingScript_v1.py`: in addition to existing band-power ratios,
  output absolute theta/alpha/beta mean band powers and base-10 log values for
  AF7/AF8 from the existing 256 Hz notched-EEG spectrogram. Skip plot creation
  entirely when `save_figure=False` and `show_figure=False`.
- `scripts/eeg/process_all_eeg.py`: add per-recording QC report columns and
  three-timepoint subject-by-band exports (long/wide). No extrapolation of missing
  recordings from stale outputs. Works for CSV and single-CSV ZIP files.
- `README.md`, `documentation/script_explanations.txt`,
  `documentation/output_explanations.txt`: document output formats, transform,
  recording-to-timepoint mapping, and QC limitations.
- `tests/test_eeg_frontal_log_power.py`: test signal scaling, preserved ratios,
  CSV/ZIP input, three-timepoint exports, known-rate-mismatch exclusion, and
  absence of stale data on refresh.

## How to run

`python3 scripts/run_all_analyses.py --no-eeg-plots`

The master script already includes EEG spectral processing by default, so no
changes to the stage order are necessary.

## Scientific interpretation

- `pre_vr` is used as anticipation (protocol timing should be verified).
- `post_vr` is named `acute_stress_response` because it is **post-VR**, not EEG
  acquired during the stress task. It cannot measure concurrent acute-stress EEG.
- `recovery` is mapped to recovery.
- Retains 256-Hz fixed-rate screening and the historical filter/interpolation
  method. No new artifact rejection is introduced.
- New data are unadjusted for motion and electrode artifacts, and should not
  be used for inference before reviewing these recordings.
