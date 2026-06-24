#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from project_paths import (
    METADATA_DIR,
    LEARNING_ANALYSIS_DATA_DIR,
    HEART_RATE_ANALYSIS_DATA_DIR,
    EEG_ANALYSIS_DATA_DIR,
    MULTIMODAL_ANALYSIS_DATA_DIR,
)


def clean_subject_id(series):
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(3)


def read_if_exists(path, dtype=None):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, dtype=dtype)
    print(f"Warning: missing file, skipping: {path}")
    return None


def main():
    MULTIMODAL_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    learning_rates_path = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
    median_path = LEARNING_ANALYSIS_DATA_DIR / "median_learning_rates_summary.csv"
    winstay_path = LEARNING_ANALYSIS_DATA_DIR / "winstay_loseswitch_summary.csv"
    slopes_path = LEARNING_ANALYSIS_DATA_DIR / "learning_slopes.csv"

    hr_path = HEART_RATE_ANALYSIS_DATA_DIR / "heart_rate_summary_wide.csv"
    eeg_path = EEG_ANALYSIS_DATA_DIR / "eeg_file_summary_wide.csv"
    subject_info_path = METADATA_DIR / "subject_info_summary.csv"

    learning = read_if_exists(learning_rates_path, dtype={"subjectId": str})
    if learning is None:
        raise FileNotFoundError(
            f"Required file missing: {learning_rates_path}. Run calculate_learning_rates.py first."
        )

    learning["subject_id"] = clean_subject_id(learning["subjectId"])

    if "plearning_num" in learning.columns:
        learning["session"] = pd.to_numeric(learning["plearning_num"], errors="coerce").astype("Int64")
    else:
        learning["session"] = pd.NA

    multimodal = learning.copy()

    # Subject-level median learning summary
    median = read_if_exists(median_path)
    if median is not None:
        if "subject" in median.columns:
            median = median[median["subject"].astype(str).str.upper() != "AVERAGE"].copy()
            median["subject_id"] = clean_subject_id(median["subject"])
            median = median.drop(columns=["subject"], errors="ignore")
            multimodal = multimodal.merge(median, on="subject_id", how="left")

    # Subject-level win-stay / lose-switch summary
    winstay = read_if_exists(winstay_path)
    if winstay is not None:
        if "subject" in winstay.columns:
            winstay["subject_id"] = clean_subject_id(winstay["subject"])
            winstay = winstay.drop(columns=["subject"], errors="ignore")
            multimodal = multimodal.merge(winstay, on="subject_id", how="left")

    # Subject/session learning slopes
    slopes = read_if_exists(slopes_path)
    if slopes is not None:
        if "Subject" in slopes.columns:
            slopes["subject_id"] = clean_subject_id(slopes["Subject"])
            slopes = slopes.drop(columns=["Subject"], errors="ignore")
            multimodal = multimodal.merge(slopes, on="subject_id", how="left")

    # Heart-rate wide summary
    hr = read_if_exists(hr_path, dtype={"subject_id": str})
    if hr is not None:
        hr["subject_id"] = clean_subject_id(hr["subject_id"])
        multimodal = multimodal.merge(hr, on="subject_id", how="left")

    # EEG availability/file summary
    eeg = read_if_exists(eeg_path, dtype={"subject_id": str})
    if eeg is not None:
        eeg["subject_id"] = clean_subject_id(eeg["subject_id"])
        multimodal = multimodal.merge(eeg, on="subject_id", how="left")

    # Subject info summary
    subject_info = read_if_exists(subject_info_path)
    if subject_info is not None:
        if "subjectId" in subject_info.columns:
            subject_info["subject_id"] = clean_subject_id(subject_info["subjectId"])
        elif "subject_id" in subject_info.columns:
            subject_info["subject_id"] = clean_subject_id(subject_info["subject_id"])
        else:
            subject_info = None

        if subject_info is not None:
            subject_info = subject_info.drop_duplicates(subset=["subject_id"])
            multimodal = multimodal.merge(subject_info, on="subject_id", how="left", suffixes=("", "_subjectinfo"))

    output_path = MULTIMODAL_ANALYSIS_DATA_DIR / "multimodal_subject_session_summary.csv"
    multimodal.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(multimodal)}")
    print(f"Columns: {len(multimodal.columns)}")


if __name__ == "__main__":
    main()