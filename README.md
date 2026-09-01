# Puberty & Virtual Reality Research Analysis

<p align="center">
  <strong>Behavioral and multimodal analysis of reward learning and stress across development</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white">
  <img src="https://img.shields.io/badge/Behavioral%20Research-7950F2?style=flat-square">
</p>

## Overview

This repository contains the preprocessing and analysis pipeline for a research project examining **reward learning, stress responses, and developmental differences** using a virtual-reality stress paradigm.

The pipeline integrates:

* probabilistic reward-learning behavior;
* heart-rate recordings;
* EEG data;
* participant and session metadata;
* developmental group information.

The project emphasizes **reproducible behavioral analysis, multimodal integration, automated data validation, and privacy-conscious research-data management**.

---

## Research Questions

The project supports questions such as:

1. How do participants learn reward contingencies across repeated trials?
2. How much individual variability exists in learning?
3. Do reward-learning patterns differ across experimental sessions?
4. How are learning and physiological stress responses related?
5. Are there developmental differences between younger and older participants?

---

## Analysis Highlights

### 🎮 Reward Learning

Behavioral analyses include:

* group-average and individual learning curves;
* first-vs-second-half performance;
* session comparisons;
* learning-rate and learning-slope analyses;
* win-stay / lose-switch behavior;
* exploratory learner-group comparisons.

### ❤️ Heart Rate & 🧠 EEG

The pipeline parses and validates physiological files, summarizes data availability, and prepares measures for integration with behavioral data.

### 🔗 Multimodal Integration

A reusable participant-session table combines available:

* behavioral measures;
* JATOS/session metadata;
* heart-rate summaries;
* EEG summaries;
* developmental-group information.

This provides a common dataset for behavioral, developmental, and multimodal analyses.

---

## Repository Structure

```text
Puberty_VR/
├── data/
│   ├── raw/               # Local participant data — not committed
│   ├── processed/         # Processed behavioral data
│   ├── analysis_data/     # Analysis-ready datasets
│   └── metadata/          # Inventories and validation reports
│
├── scripts/
│   ├── setup/
│   ├── heart_rate/
│   ├── eeg/
│   ├── learning_rates/
│   └── multimodal/
│
├── outputs/               # Generated results and figures
├── documentation/
├── PROJECT_AUDIT.md
└── requirements.txt
```

---

## Quick Start

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

The primary workflow is:

```bash
# Validate and organize source data
python3 scripts/setup/raw_data_inventory.py
python3 scripts/setup/validate_raw_data.py

# Convert pLearning data
python3 scripts/setup/process_jatos_python.py

# Generate physiological summaries
python3 scripts/heart_rate/summarize_heart_rate.py
python3 scripts/eeg/summarize_eeg.py

# Behavioral learning analysis
python3 scripts/learning_rates/calculate_learning_rates.py

# Build integrated participant-session dataset
python3 scripts/multimodal/build_multimodal_dataset.py
```

Additional behavioral and statistical analyses are available under `scripts/learning_rates/` and `scripts/multimodal/`.

---

## Statistical Notes

* Learner groups currently use an exploratory Session 1 median split of an estimated learning parameter.
* Continuous developmental comparisons use Welch tests with Hedges' g.
* Binary comparisons use Fisher's exact tests.
* Valid p-values receive Benjamini-Hochberg FDR correction.
* Small developmental subgroups limit statistical power and should be interpreted cautiously.

---

## Data Protection

Participant-level research data are excluded from Git.

Do **not** commit:

* participant names or direct identifiers;
* consent forms;
* raw physiological recordings;
* protected subject-level datasets.

Detailed code and data-quality review notes are available in [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md).

---

## Limitations

* Not every participant has every physiological modality.
* Physiological data quality and availability vary across participants.
* Developmental subgroup sizes may be small.
* Several analyses are exploratory rather than confirmatory.

---

**Skills:** `Python` · `pandas` · `SciPy` · `Behavioral Modeling` · `Statistical Analysis` · `EEG` · `Heart Rate` · `Multimodal Data` · `Research Data Management`

---

<details>
<summary><h1>Technical Details</h1></summary>

### Filename Standardization

Subject-level files can be standardized before processing:

```bash
python3 scripts/setup/standardize_file_names.py
```

This performs a dry run and writes proposed changes to:

```text
data/metadata/rename_log.csv
```

After reviewing potential conflicts such as `target_exists`, `duplicate_target_in_plan`, and `subject_mismatch`, changes can be applied with:

```bash
python3 scripts/setup/standardize_file_names.py --apply
```

Canonical subject folders use:

```text
sub###
```

Processed pLearning files use:

```text
###_plearning_1.csv
###_plearning_2.csv
```

Restricting accepted filenames prevents stale or improperly named files from being treated as participants.

### Raw-Data Validation

```bash
python3 scripts/setup/raw_data_inventory.py
python3 scripts/setup/validate_raw_data.py
```

Important outputs include:

* `subject_level_inventory.csv`
* `missing_raw_data_files.csv`
* `unrecognized_raw_files.csv`
* `raw_data_quality_issues.csv`

These checks identify missing files, schema mismatches, duplicate content, and naming conflicts before analysis.

### JATOS Conversion

```bash
python3 scripts/setup/process_jatos_python.py
```

The script recursively searches `data/raw/subject_data/`, converts JATOS pLearning exports into standardized trial-level CSV files, and updates participant/session metadata.

### Behavioral Analysis

`calculate_learning_rates.py` should be run before analyses that depend on learner-group classifications.

Additional scripts support:

* individual learning curves;
* group-average curves;
* first-vs-second-half analyses;
* learning-rate analyses;
* learning-slope analyses;
* summary statistics;
* win-stay / lose-switch analyses.

### Multimodal Dataset

```bash
python3 scripts/multimodal/build_multimodal_dataset.py
```

The main output is:

```text
data/analysis_data/multimodal/
└── multimodal_subject_session_summary.csv
```

The dataset contains one row per participant × pLearning session. Session-specific metadata are joined by both participant and session, while participant-level physiological summaries can be associated with the relevant participant rows.

### Developmental Comparisons

```bash
python3 scripts/multimodal/compare_minors_to_older.py
python3 scripts/multimodal/summarize_minor_comparison.py
```

Repeated participant-level measures are tested once per person, while session-varying variables are compared separately by session.

### Additional Statistical Caveat

`learning_slope_analysis.py` currently uses a Gaussian linear mixed model for a binary trial-level outcome. This can be useful descriptively, but a binomial mixed-effects model or GEE would be preferable for confirmatory inference.

</details>
