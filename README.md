# Puberty_VR

Analysis pipeline for the **Puberty & Virtual Reality** summer research project. This repository is organized to support preprocessing, behavioral learning analyses, and future multimodal integration across pLearning task data, physiological recordings, and other study measures.

## Project overview

This project examines how puberty may influence reward learning and stress-related responses in a virtual reality context. Current work in this repository focuses on:

* standardizing and organizing subject-level raw files
* converting JATOS pLearning JSON exports into trial-level CSVs
* computing behavioral learning metrics across pLearning sessions
* generating reusable analysis tables and figures
* building a structure that can later incorporate EEG, heart rate, saliva, and multimodal merged data

## Repository structure

```text
Puberty_VR/
├── data/
│   ├── raw/
│   │   └── subject_data/
│   ├── processed/
│   │   └── plearning/
│   ├── analysis_data/
│   │   └── learning_rates/
│   └── metadata/
│
├── scripts/
│   ├── project_paths.py
│   ├── setup/
│   └── learning_rates/
│
├── outputs/
│   └── learning_rates/
│       ├── figures/
│       ├── tables/
│       └── reports/
│
├── documentation/
├── .gitignore
└── README.md
```

## Folder purpose

### `data/raw/subject_data/`

Contains subject-level raw files, including standardized behavioral, EEG, and heart-rate files. This folder is ignored by Git.

### `data/processed/plearning/`

Contains processed pLearning trial-level CSVs created from JATOS JSON files. These files are analysis-ready behavioral inputs and are ignored by Git.

### `data/analysis_data/learning_rates/`

Contains reusable analysis CSVs produced by learning-rate scripts, such as:

* `learning_rates.csv`
* `individual_proportions_session*.csv`
* `median_learning_rates_summary.csv`
* `winstay_loseswitch_summary.csv`
* `trial_level_data.csv`
* `learning_slopes.csv`

This folder is ignored by Git.

### `data/metadata/`

Contains project metadata and file-tracking tables, such as:

* `subject_info_summary.csv`
* `file_name_dictionary.csv`
* `subject_level_inventory.csv`
* `all_raw_data.csv`
* `missing_raw_data_files.csv`

This folder is ignored by Git.

### `scripts/setup/`

Contains preprocessing and organizational scripts, including:

* JSON renaming / standardization
* raw file renaming
* JATOS JSON → pLearning CSV conversion
* raw data inventory generation

### `scripts/learning_rates/`

Contains behavioral analysis scripts for:

* learning-rate estimation
* individual and group learning-curve plotting
* session 1 vs session 2 comparisons
* win-stay / lose-switch analyses
* slope analyses and related outputs

### `outputs/learning_rates/`

Contains generated figures, tables, and report files from learning-rate scripts. This folder is ignored by Git.

### `documentation/`

Contains project notes, script descriptions, and output explanations.

## Data handling

This repository is configured so that **raw data, processed subject data, analysis CSVs, metadata files, and generated outputs are not pushed to GitHub**. The `.gitignore` keeps the project code and structure version-controlled without exposing subject data or generated result files.

Tracked content includes:

* scripts
* documentation
* repository structure
* configuration files such as `.gitignore`

Ignored content includes:

* raw subject data
* processed pLearning CSVs
* analysis tables generated from subject data
* metadata files derived from subject data
* generated figures, tables, and reports

## Main setup / preprocessing scripts

### `scripts/setup/standardize_file_names.py`

Renames raw subject files into a standardized naming convention.

### `scripts/setup/rename_jsons.py`

Standardizes pLearning JSON filenames.

### `scripts/setup/process_jatos_python.py`

Converts JATOS pLearning JSON files into processed trial-level CSVs saved in `data/processed/plearning/`.

### `scripts/setup/raw_data_inventory.py`

Builds metadata tables describing what files exist for each subject and where they are located.

## Main learning-rate analysis scripts

Examples include:

* `calculate_learning_rates.py`
* `all_individuals_with_curve.py`
* `individual_plots_with_classification.py`
* `learning_rate_analysis.py`
* `winstay_loseswitch_analysis.py`
* `learning_slope_analysis.py`

These scripts use shared project paths defined in `scripts/project_paths.py`.

## Running scripts

Run scripts from the project root (`Puberty_VR/`) so relative imports behave consistently.

Example:

```bash
python3 scripts/setup/process_jatos_python.py data/raw/subject_data --recursive
python3 scripts/learning_rates/calculate_learning_rates.py
```

## Requirements

Core Python packages used in this project include:

* pandas
* numpy
* matplotlib
* scipy
* statsmodels

Install as needed in your environment.

## Notes

This repository is still evolving as the project expands beyond behavioral pLearning analyses to additional modalities such as ECG/heart rate, EEG, saliva measures, and multimodal merged datasets.
