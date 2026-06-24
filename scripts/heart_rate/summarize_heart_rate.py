#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from project_paths import METADATA_DIR, HEART_RATE_ANALYSIS_DATA_DIR


def find_hr_column(df):
    possible = [
        "heart_rate", "heartrate", "heart rate", "hr", "HR",
        "bpm", "BPM", "value", "Value"
    ]

    for col in df.columns:
        if col in possible:
            return col

    for col in df.columns:
        lower = col.lower().replace(" ", "").replace("_", "")
        if lower in ["heartrate", "hr", "bpm"]:
            return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) > 0 and vals.between(30, 220).mean() > 0.70:
            return col

    return None


def summarize_file(row):
    file_path = Path(row["file_path"])

    base = {
        "subject_id": str(row["subject_id"]).zfill(3),
        "file_key": row["file_key"],
        "phase": row["phase"],
        "modality": row["modality"],
        "file_path": str(file_path),
        "file_exists": file_path.exists(),
        "hr_column": None,
        "n_samples": 0,
        "mean_hr": np.nan,
        "median_hr": np.nan,
        "min_hr": np.nan,
        "max_hr": np.nan,
        "sd_hr": np.nan,
    }

    if not file_path.exists():
        return base

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        base["read_error"] = str(e)
        return base

    hr_col = find_hr_column(df)
    base["hr_column"] = hr_col

    if hr_col is None:
        base["read_error"] = "No heart-rate column found"
        return base

    hr = pd.to_numeric(df[hr_col], errors="coerce").dropna()

    base.update({
        "n_samples": len(hr),
        "mean_hr": hr.mean(),
        "median_hr": hr.median(),
        "min_hr": hr.min(),
        "max_hr": hr.max(),
        "sd_hr": hr.std(),
        "read_error": "",
    })

    return base


def main():
    raw_index_path = METADATA_DIR / "all_raw_data.csv"

    if not raw_index_path.exists():
        raise FileNotFoundError(f"Missing {raw_index_path}. Run raw_data_inventory.py first.")

    raw = pd.read_csv(raw_index_path, dtype={"subject_id": str})
    hr_files = raw[raw["modality"] == "heart_rate"].copy()

    summaries = [summarize_file(row) for _, row in hr_files.iterrows()]
    long_df = pd.DataFrame(summaries)

    HEART_RATE_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    long_path = HEART_RATE_ANALYSIS_DATA_DIR / "heart_rate_summary_long.csv"
    wide_path = HEART_RATE_ANALYSIS_DATA_DIR / "heart_rate_summary_wide.csv"

    long_df.to_csv(long_path, index=False)

    value_cols = [
        "mean_hr", "median_hr", "min_hr", "max_hr", "sd_hr", "n_samples"
    ]

    wide_parts = []
    for value in value_cols:
        pivot = long_df.pivot_table(
            index="subject_id",
            columns="phase",
            values=value,
            aggfunc="first"
        )
        pivot.columns = [f"{phase}_{value}" for phase in pivot.columns]
        wide_parts.append(pivot)

    wide_df = pd.concat(wide_parts, axis=1).reset_index()

    # Derived HR reactivity variables
    if "vr_speech_mean_hr" in wide_df.columns and "pre_vr_mean_hr" in wide_df.columns:
        wide_df["speech_hr_reactivity"] = wide_df["vr_speech_mean_hr"] - wide_df["pre_vr_mean_hr"]

    if "vr_math_mean_hr" in wide_df.columns and "pre_vr_mean_hr" in wide_df.columns:
        wide_df["math_hr_reactivity"] = wide_df["vr_math_mean_hr"] - wide_df["pre_vr_mean_hr"]

    if "vr_prep_mean_hr" in wide_df.columns and "pre_vr_mean_hr" in wide_df.columns:
        wide_df["prep_hr_reactivity"] = wide_df["vr_prep_mean_hr"] - wide_df["pre_vr_mean_hr"]

    if "post_vr_mean_hr" in wide_df.columns and "pre_vr_mean_hr" in wide_df.columns:
        wide_df["post_minus_pre_hr"] = wide_df["post_vr_mean_hr"] - wide_df["pre_vr_mean_hr"]

    if "recovery_mean_hr" in wide_df.columns and "post_vr_mean_hr" in wide_df.columns:
        wide_df["recovery_minus_post_hr"] = wide_df["recovery_mean_hr"] - wide_df["post_vr_mean_hr"]

    wide_df.to_csv(wide_path, index=False)

    print(f"Saved: {long_path}")
    print(f"Saved: {wide_path}")


if __name__ == "__main__":
    main()