# Puberty & VR project audit

Audit completed on the uploaded project archive. The corrected copy was tested from outside the project directory so successful runs did not depend on the current working directory.

## Executive summary

The original project had several failures that were more serious than ordinary cleanup:

1. Two setup scripts still used an old absolute Mac path and could not operate on the current folder structure.
2. `sub028` was nested inside `sub029`, so subject discovery was structurally wrong.
3. The JATOS converter labeled newer participants as `unknown`, causing different subjects to overwrite the same processed files while the script appeared to succeed.
4. The heart-rate summarizer read a calorie-percentage field or metadata value instead of the embedded `HR (bpm)` series. Original heart-rate and multimodal results were therefore invalid.
5. The multimodal merge attached one session's JATOS metadata to both sessions for a subject.
6. The EEG summarizer treated any readable CSV as valid EEG and counted a mislabeled Polar heart-rate file as available EEG.

Those issues have been corrected, and all derived outputs in this archive were regenerated.

## High-priority fixes completed

### Portable paths and folder structure

- Replaced hard-coded `/Users/maddiemac/Puberty_VR/results` references with shared paths in `scripts/project_paths.py`.
- All scripts now resolve the project root from their own file location and run successfully from another working directory.
- Moved `data/raw/subject_data/sub029/sub028/` to `data/raw/subject_data/sub028/`.
- Added project-relative path storage in metadata instead of embedding the temporary machine path.

### Filename handling

- Added `scripts/file_naming.py` as the canonical parser/generator for subject and pLearning filenames.
- Rewrote `standardize_file_names.py` as a dry-run-by-default command with `--apply`.
- Added case-safe renaming for macOS case-insensitive filesystems.
- Rewrote `rename_jsons.py` as a portable, focused alternative.
- Learning scripts now accept only exact `###_plearning_1.csv` and `###_plearning_2.csv` names, so stale files such as `unknown_plearning_1.csv` cannot become fake participants.

### JATOS conversion

- Correctly identifies subjects 027, 029, and 030 from legacy filenames and parent folders.
- Writes canonical, non-colliding outputs.
- Honors `--output-dir` and `--summary-path`.
- Recursively searches the project raw-data directory when run with no arguments.
- Rejects duplicate JSON files that map to the same subject/session instead of overwriting silently.
- Updates `subject_info_summary.csv` by `subjectId + plearning_num`, not worker ID alone.
- Removed stale `unknown` output rows/files.
- Corrected the impossible `mix_blocks` calculation.

Current behavioral result: **34 subject-session CSVs across 18 subjects**, with no duplicate subject/session rows and no `unknown` subjects.

### Heart-rate processing

The Polar files contain:

1. a workout metadata header,
2. one metadata row,
3. a second embedded header beginning with `Sample rate,Time,HR (bpm),...`,
4. the actual second-by-second samples.

The original script read the first table and selected an unrelated numeric column. Every summary effectively had one sample. The corrected script:

- detects and parses the embedded sample header,
- uses only `HR (bpm)`,
- rejects implausible/out-of-range values,
- records parsing format and errors,
- produces long and wide summaries using project-relative paths.

Regenerated result: **118 heart-rate recordings**, all using `HR (bpm)`, with **65–411 valid samples per file** rather than one.

### EEG processing

- Added support for Mind Monitor CSVs and ZIP archives containing exactly one CSV.
- Added schema validation requiring `TimeStamp` and recognizable EEG signal columns.
- Invalid-schema files are no longer counted as available EEG.

This correctly identified `sub007/007_pre_vr_eeg.csv` as a Polar heart-rate export, not EEG.

### Behavioral analysis consistency

- Added `scripts/learning_utils.py` so eight scripts share one exponential learning function instead of maintaining duplicate definitions.
- Standardized all good/bad learner comparisons to use `learning_rates.csv` as the single source of truth.
- Removed silent fallback classifications based on accuracy or win-stay behavior.
- Corrected Trial 1 versus Trial 18 pairing to join explicitly by subject before paired tests.
- Added small-sample and zero-variance handling to summary statistics.
- Replaced bare exception handlers in curve fitting with specific numerical exceptions.
- Updated win-stay/lose-switch calculations:
  - trial-wise values remain conditional on the preceding feedback;
  - overall subject values now pool all eligible transitions rather than averaging trial positions with unequal denominators;
  - all-NaN and small-n calculations no longer emit misleading runtime warnings.

### Multimodal merge

- JATOS metadata now merges on both `subject_id` and `session`.
- Subject-level heart-rate and EEG summaries use validated many-to-one merges.
- Duplicate IDs cause explicit errors.
- Final table has **34 rows and 221 columns**, one row per behavioral subject/session.
- Worker IDs match the corresponding subject and session on every row.

### Repository/documentation

- Added a real `.gitignore`; the README previously claimed one existed when it did not.
- Added `requirements.txt`.
- Rewrote the README with the current pipeline and folder structure.
- Updated script/output documentation and removed references to the nonexistent `individual_plots_with_classification.py` as an active script.
- Added `validate_raw_data.py` and `raw_data_quality_issues.csv` for repeatable checks.

## Raw-data issues requiring manual review

These were not automatically deleted or reassigned because the correct decision depends on study timing/protocol records.

| Subject | Issue | Evidence / action needed |
|---|---|---|
| 007 | `007_pre_vr_eeg.csv` is not EEG | It is byte-for-byte identical to `007_baseline_hr.csv` and contains Polar heart-rate columns. Locate the true pre-VR EEG file or mark it missing. |
| 007 | `007_vr_prep_hr.csv` and `007_vr_speech_hr.csv` are identical | Both contain the same recording starting at 16:45:14. Verify whether one phase label is wrong or one is an accidental duplicate. |
| 013 | Two distinct files map to pre-VR heart rate | `pre vr heart rate.CSV` starts at 15:45:02; `013_pre_vr_hr.csv` starts at 15:53:59. Determine which phase each recording represents before renaming. |
| 022 | Redundant unnamed HR export | `Hellen_Rivera_2026-04-24_16-40-08.CSV` is identical to `022_post_vr_hr.csv`. It can be removed after confirming the canonical copy and backup. |
| 028 | Incomplete subject folder | Only baseline HR and baseline EEG are present. The inventory correctly reports the remaining expected files as missing. |

The complete machine-readable list is in `data/metadata/raw_data_quality_issues.csv`.

### Protocol requirement review

The inventory currently marks separate `vr_math_eeg` and `vr_speech_eeg` files as required, but 18 of 19 raw subject folders lack each one. Confirm whether:

- EEG was supposed to be exported separately for math and speech,
- one continuous VR/TSST EEG recording should replace both entries, or
- those two file keys should be optional.

Until that is clarified, the missing-file report may overstate required EEG incompleteness.

### Privacy review

Consent PDFs and some filenames containing a participant/operator name are present under raw data. The new `.gitignore` prevents accidental Git commits, but inspect and remove/de-identify these before sharing the folder outside the approved research environment.

## Redundant data and code

### Safe-to-review data redundancy

- There are 29 raw pLearning CSV copies that are byte-for-byte identical to regenerated files in `data/processed/plearning/`.
- The JSON exports are the raw source; the processed folder is the canonical analysis input.
- The raw CSV copies can be archived or removed after confirming the JSON backups and workflow requirements.

### Remaining code refactor opportunities

The major duplicated learning function was centralized. Some broader refactoring remains optional:

- Nine older plotting/analysis scripts still execute at import time rather than placing work in `main()` functions.
- Several scripts repeat plotting, curve-fitting, and SEM-band code that could be consolidated into a plotting utility module.
- Figure colors and labels are hard-coded independently across scripts.
- A single pipeline runner or task tool could enforce dependencies and avoid manually running twelve behavioral scripts.

These do not prevent the current scripts from running, but refactoring them would make testing and reuse easier.

## Statistical cautions rather than code errors

- The good/bad learner median split is exploratory and sample-dependent. It should not be treated as a stable clinical category without prespecification and validation.
- `learning_slope_analysis.py` fits a Gaussian linear mixed model to binary trial-level correctness. For confirmatory inference, use a binomial mixed-effects model or GEE. The existing model can remain as a descriptive approximation if clearly labeled.
- Curve fits at parameter bounds should be interpreted cautiously, especially with the current small sample.

## Validation performed

The corrected project was compiled and run from `/tmp`, outside the project root.

Successfully tested:

- filename standardizer dry run, apply mode, and idempotent second dry run
- focused JSON renamer
- raw inventory and raw-data validator
- default recursive JATOS conversion
- custom JATOS output directory and summary path behavior
- heart-rate long/wide summaries
- EEG long/wide summaries and invalid-schema detection
- all 12 behavioral analysis/plotting scripts
- multimodal subject-session merge

Expected warnings only:

- subjects 014 and 027 have Session 1 but no Session 2 and are excluded from paired-session analyses.

No hard-coded personal filesystem paths, stale `../results` code paths, `unknown` participant outputs, tracebacks, or unexpected runtime warnings remain in the tested pipeline.
