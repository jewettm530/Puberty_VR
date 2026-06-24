#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from project_paths import METADATA_DIR, EEG_ANALYSIS_DATA_DIR


def summarize_eeg_file(row):
    file_path = Path(row["file_path"])

    base = {
        "subject_id": str(row["subject_id"]).zfill(3),
        "file_key": row["file_key"],
        "phase": row["phase"],
        "modality": row["modality"],
        "file_path": str(file_path),
        "file_exists": file_path.exists(),
        "n_rows": 0,
        "n_columns": 0,
        "column_names": "",
        "read_error": "",
    }

    if not file_path.exists():
        return base

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        base["read_error"] = str(e)
        return base

    base.update({
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "column_names": "; ".join(df.columns.astype(str)),
    })

    return base


def main():
    raw_index_path = METADATA_DIR / "all_raw_data.csv"

    if not raw_index_path.exists():
        raise FileNotFoundError(f"Missing {raw_index_path}. Run raw_data_inventory.py first.")

    raw = pd.read_csv(raw_index_path, dtype={"subject_id": str})
    eeg_files = raw[raw["modality"] == "eeg"].copy()

    summaries = [summarize_eeg_file(row) for _, row in eeg_files.iterrows()]
    long_df = pd.DataFrame(summaries)

    EEG_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    long_path = EEG_ANALYSIS_DATA_DIR / "eeg_file_summary_long.csv"
    wide_path = EEG_ANALYSIS_DATA_DIR / "eeg_file_summary_wide.csv"

    long_df.to_csv(long_path, index=False)

    availability = long_df.copy()
    availability["available"] = availability["file_exists"].astype(int)

    wide_available = availability.pivot_table(
        index="subject_id",
        columns="phase",
        values="available",
        aggfunc="max",
        fill_value=0
    )

    wide_available.columns = [f"{phase}_eeg_available" for phase in wide_available.columns]

    wide_rows = long_df.pivot_table(
        index="subject_id",
        columns="phase",
        values="n_rows",
        aggfunc="first"
    )

    wide_rows.columns = [f"{phase}_eeg_n_rows" for phase in wide_rows.columns]

    wide_df = pd.concat([wide_available, wide_rows], axis=1).reset_index()
    wide_df.to_csv(wide_path, index=False)

    print(f"Saved: {long_path}")
    print(f"Saved: {wide_path}")


if __name__ == "__main__":
    main()