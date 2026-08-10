#!/usr/bin/env python3
"""Merge behavioral, physiological, EEG-availability, and session metadata tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_paths import (  # noqa: E402
    EEG_ANALYSIS_DATA_DIR,
    HEART_RATE_ANALYSIS_DATA_DIR,
    LEARNING_ANALYSIS_DATA_DIR,
    METADATA_DIR,
    MULTIMODAL_ANALYSIS_DATA_DIR,
    TABLES_DIR,
)


def clean_subject_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"^(?:subject_)?(\d+)")[0].str.zfill(3)


def read_if_exists(path: str | Path, dtype: dict[str, type] | None = None) -> pd.DataFrame | None:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, dtype=dtype)
    print(f"Warning: missing file, skipping: {path}")
    return None


def merge_subject_level(base: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    if other["subject_id"].duplicated().any():
        duplicates = other.loc[other["subject_id"].duplicated(keep=False), "subject_id"].tolist()
        raise ValueError(f"Duplicate subject IDs in {label}: {duplicates}")
    return base.merge(other, on="subject_id", how="left", validate="many_to_one")


def main() -> None:
    MULTIMODAL_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    learning_rates_path = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
    median_path = TABLES_DIR / "learning_rate_analysis" / "median_learning_rates_summary.csv"
    winstay_path = TABLES_DIR / "winstay_loseswitch" / "winstay_loseswitch_summary.csv"
    slopes_path = LEARNING_ANALYSIS_DATA_DIR / "slope_analysis" / "learning_slopes.csv"
    hr_path = HEART_RATE_ANALYSIS_DATA_DIR / "heart_rate_summary_wide.csv"
    eeg_path = EEG_ANALYSIS_DATA_DIR / "eeg_file_summary_wide.csv"
    subject_info_path = METADATA_DIR / "subject_info_summary.csv"

    learning = read_if_exists(learning_rates_path, dtype={"subjectId": str})
    if learning is None:
        raise FileNotFoundError(
            f"Required file missing: {learning_rates_path}. Run calculate_learning_rates.py first."
        )

    learning["subject_id"] = clean_subject_id(learning["subjectId"])
    learning["session"] = pd.to_numeric(learning["plearning_num"], errors="coerce").astype("Int64")
    learning = learning.dropna(subset=["subject_id", "session"])
    if learning.duplicated(["subject_id", "session"]).any():
        raise ValueError("learning_rates.csv contains duplicate subject/session rows")
    multimodal = learning.copy()

    median = read_if_exists(median_path)
    if median is not None and "subject" in median.columns:
        median = median.copy()
        median["subject_id"] = clean_subject_id(median["subject"])
        median = median.dropna(subset=["subject_id"]).drop(columns=["subject"], errors="ignore")
        multimodal = merge_subject_level(multimodal, median, "median learning summary")

    winstay = read_if_exists(winstay_path)
    if winstay is not None and "subject" in winstay.columns:
        winstay = winstay.copy()
        winstay["subject_id"] = clean_subject_id(winstay["subject"])
        winstay = winstay.dropna(subset=["subject_id"]).drop(columns=["subject"], errors="ignore")
        multimodal = merge_subject_level(multimodal, winstay, "win-stay/lose-switch summary")

    slopes = read_if_exists(slopes_path)
    if slopes is not None and "Subject" in slopes.columns:
        slopes = slopes.copy()
        slopes["subject_id"] = clean_subject_id(slopes["Subject"])
        slopes = slopes.dropna(subset=["subject_id"]).drop(columns=["Subject"], errors="ignore")
        multimodal = merge_subject_level(multimodal, slopes, "learning slopes")

    hr = read_if_exists(hr_path, dtype={"subject_id": str})
    if hr is not None:
        hr = hr.copy()
        hr["subject_id"] = clean_subject_id(hr["subject_id"])
        hr = hr.dropna(subset=["subject_id"])
        multimodal = merge_subject_level(multimodal, hr, "heart-rate summary")

    eeg = read_if_exists(eeg_path, dtype={"subject_id": str})
    if eeg is not None:
        eeg = eeg.copy()
        eeg["subject_id"] = clean_subject_id(eeg["subject_id"])
        eeg = eeg.dropna(subset=["subject_id"])
        multimodal = merge_subject_level(multimodal, eeg, "EEG summary")

    # JATOS metadata is session-specific. The previous implementation dropped
    # duplicate subjects and then attached one session's metadata to both rows.
    subject_info = read_if_exists(subject_info_path, dtype={"subjectId": str, "plearning_num": str})
    if subject_info is not None and "subjectId" in subject_info.columns:
        subject_info = subject_info.copy()
        subject_info["subject_id"] = clean_subject_id(subject_info["subjectId"])
        if "plearning_num" in subject_info.columns:
            subject_info["session"] = pd.to_numeric(subject_info["plearning_num"], errors="coerce").astype("Int64")
        elif "worker_id" in subject_info.columns:
            extracted = subject_info["worker_id"].astype(str).str.extract(r"plearn(?:ing)?[_-]?([12])", expand=False)
            subject_info["session"] = pd.to_numeric(extracted, errors="coerce").astype("Int64")
        else:
            subject_info["session"] = pd.NA

        subject_info = subject_info.dropna(subset=["subject_id", "session"])
        subject_info = subject_info.drop_duplicates(["subject_id", "session"], keep="last")
        multimodal = multimodal.merge(
            subject_info,
            on=["subject_id", "session"],
            how="left",
            suffixes=("", "_subjectinfo"),
            validate="one_to_one",
        )

    output_path = MULTIMODAL_ANALYSIS_DATA_DIR / "multimodal_subject_session_summary.csv"
    multimodal = multimodal.sort_values(["subject_id", "session"]).reset_index(drop=True)
    multimodal.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(multimodal)}")
    print(f"Columns: {len(multimodal.columns)}")


if __name__ == "__main__":
    main()
