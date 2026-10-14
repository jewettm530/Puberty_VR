# 2026-10-14 pipeline update

This is a **code + documentation patch** for the uploaded Puberty_VR project.
It deliberately excludes all raw subject data, derived participant data,
plots, presentation files, and the project's Git history. Extract this patch
into your **existing** project folder, retaining all other files.

## Files changed

- `scripts/run_all_analyses.py`: a safer one-command runner. Creates
  `outputs/pipeline_run_status.json` with stage outcomes and the EEG file-level
  QC summary. It specifically detects the Apple Numbers document mislabeled
  `.xlsx`, skips only the age comparison, and marks age outputs not current.
  Required-stage failures still stop the full pipeline.
- `scripts/eeg/process_all_eeg.py`: stores project-relative raw and result
  paths in the EEG batch QC report, for portable reruns.
- `scripts/setup/archive_legacy_outputs.py`: optional archive utility,
  dry-run by default; old EEG outputs **are not numerically interchangeable**.
- `tests/test_run_all_analyses.py`: three simple regression tests.
- `README.md`: new-subject workflow, commands, EEG QC, Numbers export,
  current and legacy outputs, and manual-age-lookup cautions.
- `PROJECT_AUDIT.md`: follow-up audit with unresolved items.
- `documentation/script_explanations.txt`, `output_explanations.txt`, and
  `OriginalR_vs_FinalPython.txt`: current scripts/results and metadata behavior.
- `.gitignore`: protect all generated data and output directories.

## Installation

1. Back up your original scripts and README if you have local changes.
2. Extract the ZIP into the folder *containing* your existing `Puberty_VR/`
   folder (e.g. `/Users/maddiemac/`) and merge the folders. You can also
   extract and copy the changed files manually.
3. Export the `Real_Subject_Data_ID` spreadsheet from **Numbers → File →
   Export To → Excel**. Place the real `.xlsx` file in
   `data/raw/subject_data/` after reviewing the age-group classifications,
   including `031` and `032`.
4. Run `python3 scripts/run_all_analyses.py --dry-run`.
5. Run `python3 scripts/run_all_analyses.py --no-eeg-plots` for a complete
   scientific-analysis refresh without large EEG TIFF plots, or omit the flag
   if you need those TIFFs regenerated.
6. Check `outputs/pipeline_run_status.json`, the EEG batch QC report, and
   the age comparison status; the run is only partially complete if age is
   skipped, even if its other stages succeed.

Tests: `python3 -m unittest discover -s tests -v`.

## Actual-dataset validation

The 2026-10-14 source ZIP was extracted to a separate working directory and
run through the entire required pipeline with `--no-eeg-plots`: all required
stages completed (exit code 0) and the age stage was skipped specifically
because the mislabeled `.xlsx` is a Numbers archive. The runner marked older
age comparisons `NOT_CURRENT`. A set of three isolated regression tests was
also run successfully. This testing does **not** establish scientific validity
of EEG ratios or age-comparison claims; EEG exclusions and raw signal artifacts
require review.

## Cleanup findings

- `outputs/eeg/batch/`: historical results, 56 files / ~278 MB in the source
  archive, not used by the current runner. Because their numerical outputs
  differ from counterparts in `outputs/eeg/bandpower/`, **archive, don't delete**.
- `data/processed/plearning/RL_parameter_results.csv`: older table not used by
  current scripts; optional archiver can preserve it.
- `rename_jsons.py`: useful single-purpose script, not run by master.
- `PreprocessingScript_v1.py`: required by the current EEG batch processor.

Nothing has been deleted from your project by this patch.
