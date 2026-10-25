# Installation: AF7/AF8 theta, alpha, beta log-power update

Extract `Puberty_VR_frontal_log_power_update_2026-10-25.zip` into the **folder containing your existing `Puberty_VR` directory**, preserving paths. Replace the two matching Python files and merge the updated README, documentation, changelog and tests. Back up local modifications first. This patch does **not** contain or remove any subject data, generated figures, or earlier analyses.

## Files to update
- `Puberty_VR/scripts/PreprocessingScript_v1.py`
- `Puberty_VR/scripts/eeg/process_all_eeg.py`
- `Puberty_VR/README.md`
- `Puberty_VR/documentation/script_explanations.txt`
- `Puberty_VR/documentation/output_explanations.txt`
- `Puberty_VR/CHANGELOG_2026-10-25.md`
- `Puberty_VR/tests/test_eeg_frontal_log_power.py`

Your existing `scripts/run_all_analyses.py` **already runs** `scripts/eeg/process_all_eeg.py` by default. It does not need replacing for this enhancement (use the latest runner from the 2026-10-14 update, not the initial draft).

Run:

```bash
cd /Users/maddiemac/Puberty_VR
python3 -m unittest discover -s tests -v
python3 scripts/run_all_analyses.py --no-eeg-plots
```

Or run EEG only with `python3 scripts/eeg/process_all_eeg.py --no-plots`.

## Review outputs
- `data/analysis_data/eeg/eeg_frontal_log_bandpower_long.csv`
- `data/analysis_data/eeg/eeg_frontal_log_bandpower_wide.csv`
- `data/analysis_data/eeg/eeg_batch_processing_report.csv`

Post-VR is a **post-stressor response**, not simultaneous within-VR acute-stress EEG. EEG quality beyond sample rate has NOT been validated. Raw data remain untouched.
