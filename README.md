# Puberty & VR analysis project

Portable preprocessing and analysis code for the **Puberty & Virtual Reality** research project. The current pipeline organizes subject-level raw files, converts JATOS pLearning exports, summarizes heart-rate and EEG availability, runs behavioral learning analyses, and builds a subject-session multimodal table.

## Project structure

```text
Puberty_VR/
├── data/
│   ├── raw/subject_data/          # source files grouped as sub###/
│   ├── processed/plearning/       # trial-level behavioral CSVs
│   ├── analysis_data/             # reusable analysis tables
│   └── metadata/                  # inventories, summaries, validation reports
├── scripts/
│   ├── project_paths.py           # portable project paths
│   ├── file_naming.py             # canonical filename parsing
│   ├── learning_utils.py          # shared learning-curve function
│   ├── setup/
│   ├── heart_rate/
│   ├── eeg/
│   ├── learning_rates/
│   └── multimodal/
├── outputs/                       # generated figures, tables, and reports
├── documentation/
├── requirements.txt
└── README.md
```

Scripts resolve paths from their own location, so they can be launched from any working directory.

## Environment

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

## Recommended pipeline

Run these commands from the project root or replace paths with absolute paths.

### 1. Preview filename standardization

```bash
python3 scripts/setup/standardize_file_names.py
```

This is a dry run. Review `data/metadata/rename_log.csv`, especially any `target_exists`, `duplicate_target_in_plan`, or `subject_mismatch` rows.

Apply only after reviewing the log:

```bash
python3 scripts/setup/standardize_file_names.py --apply
```

`rename_jsons.py` is a narrower alternative that changes only pLearning JSON names. It is normally unnecessary when the full standardizer is used.

### 2. Inventory and validate raw files

```bash
python3 scripts/setup/raw_data_inventory.py
python3 scripts/setup/validate_raw_data.py
```

Important outputs:

- `subject_level_inventory.csv`: one row per subject
- `missing_raw_data_files.csv`: absent expected files, including whether each is required
- `unrecognized_raw_files.csv`: files outside the current naming dictionary
- `raw_data_quality_issues.csv`: schema mismatches, exact duplicate content, and filename conflicts

### 3. Convert JATOS JSON to trial-level CSV

```bash
python3 scripts/setup/process_jatos_python.py
```

With no arguments, the script recursively searches `data/raw/subject_data/`. It writes canonical files such as `027_plearning_1.csv` to `data/processed/plearning/` and updates `data/metadata/subject_info_summary.csv` by subject and session.

Custom inputs and output locations are supported:

```bash
python3 scripts/setup/process_jatos_python.py path/to/json_or_folder --recursive \
  --output-dir path/to/processed \
  --summary-path path/to/subject_info_summary.csv
```

### 4. Generate modality summaries

```bash
python3 scripts/heart_rate/summarize_heart_rate.py
python3 scripts/eeg/summarize_eeg.py
```

The heart-rate script handles Polar exports that contain a metadata row followed by an embedded `HR (bpm)` header. The EEG script accepts Mind Monitor CSV files or ZIP archives containing exactly one CSV and validates that expected EEG columns are present.

### 5. Run behavioral analyses

Run `calculate_learning_rates.py` first. Its Session 1 `learner_group` column is the single source of truth for all good/bad learner comparisons.

```bash
python3 scripts/learning_rates/calculate_learning_rates.py
python3 scripts/learning_rates/all_individuals_with_curve.py
python3 scripts/learning_rates/first_half_vs_second_half.py
python3 scripts/learning_rates/group_average_by_winner.py
python3 scripts/learning_rates/individual_first_vs_second_halves.py
python3 scripts/learning_rates/learning_rate_analysis.py
python3 scripts/learning_rates/learning_slope_analysis.py
python3 scripts/learning_rates/learning_summary_statistics.py
python3 scripts/learning_rates/plot_group_average_curve.py
python3 scripts/learning_rates/plot_individual_learning_curves.py
python3 scripts/learning_rates/winstay_loseswitch_analysis.py
python3 scripts/learning_rates/group_winstay_loseswitch_sessions.py
```

The final group win-stay/lose-switch script depends on the summary created by `winstay_loseswitch_analysis.py`.

### 6. Build the multimodal subject-session table

```bash
python3 scripts/multimodal/build_multimodal_dataset.py
```

Output:

```text
data/analysis_data/multimodal/multimodal_subject_session_summary.csv
```

The merge is one row per subject and pLearning session. Session-specific JATOS metadata is merged on both subject and session; subject-level physiological summaries are repeated across that subject's session rows.

### 7. Compare minor and older subjects

```bash
python3 scripts/multimodal/compare_minors_to_older.py
python3 scripts/multimodal/summarize_minor_comparison.py
```

The script reads minor status from the first column of
`data/raw/subject_data/Real_Subject_Data_ID.xlsx`. A trimmed, case-insensitive
value of `N/A Minor` defines the minor group. It requires at least five distinct
minor subjects in the multimodal dataset and writes:

- `data/analysis_data/minor_comparison/subject_age_groups.csv`
- `data/analysis_data/minor_comparison/minor_vs_older_joined.csv`
- `outputs/minor_comparison/minor_vs_older_numeric_comparisons.csv`
- `outputs/minor_comparison/minor_vs_older_data_availability.csv`
- `outputs/minor_comparison/minor_vs_older_analysis_report.txt`
- `outputs/minor_comparison/minor_vs_older_key_results.csv`
- `outputs/minor_comparison/minor_vs_older_major_findings.txt`
- `outputs/minor_comparison/figures/`

Repeated subject-level measures are tested once per person; measures that vary
within a person are compared separately by session. Continuous outcomes use
Welch tests with Hedges' g, binary outcomes use Fisher exact tests, and all
valid p-values receive Benjamini-Hochberg FDR correction. Treat these results
as exploratory because the minor cohort is small.

## Canonical filenames

Subject folders use `sub###`. Examples:

```text
sub027/
├── subject_027_plearning_1.json
├── 027_baseline_hr.csv
├── 027_baseline_eeg.zip
├── 027_pre_vr_hr.csv
├── 027_pre_vr_eeg.zip
└── ...
```

Processed behavioral files use:

```text
###_plearning_1.csv
###_plearning_2.csv
```

Only this exact processed pattern is accepted by learning scripts, preventing stale files such as `unknown_plearning_1.csv` from being treated as participants.

## Data protection

Raw files, derived subject data, metadata CSVs, and outputs are ignored by Git. Do not commit consent forms or files containing participant names. Before sharing the project, inspect:

```text
data/raw/subject_data/
data/metadata/unrecognized_raw_files.csv
data/metadata/raw_data_quality_issues.csv
```

## Statistical notes

- Good/bad learner groups use a Session 1 median split of the fitted exponential learning-rate parameter `k`. Treat this as an exploratory grouping unless it is prespecified and justified.
- `learning_slope_analysis.py` currently fits a Gaussian linear mixed model to a binary trial-level outcome. It is useful descriptively, but a binomial mixed-effects model or GEE is more appropriate for confirmatory inference on binary choices.
- Missing-session subjects remain in summaries where possible and are excluded only from paired analyses that require both sessions.

See `PROJECT_AUDIT.md` for the completed code/data audit and remaining manual-review items.
