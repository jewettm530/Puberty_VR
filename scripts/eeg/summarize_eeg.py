#!/usr/bin/env python3
"""Summarize EEG CSVs or single-file ZIP archives from the raw-data index."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_paths import EEG_ANALYSIS_DATA_DIR, METADATA_DIR, resolve_project_path  # noqa: E402


def summarize_eeg_file(row: pd.Series) -> dict[str, object]:
    stored_path = Path(row["file_path"])
    file_path = resolve_project_path(stored_path)

    base: dict[str, object] = {
        "subject_id": str(row["subject_id"]).zfill(3),
        "file_key": row["file_key"],
        "phase": row["phase"],
        "modality": row["modality"],
        "file_path": str(stored_path),
        "storage_format": file_path.suffix.lower().lstrip("."),
        "file_exists": file_path.exists(),
        "n_rows": 0,
        "n_columns": 0,
        "column_names": "",
        "eeg_schema_valid": False,
        "read_error": "",
    }

    if not file_path.exists():
        base["read_error"] = "file_not_found"
        return base

    try:
        # pandas directly reads ZIP archives that contain exactly one CSV.
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as exc:
        base["read_error"] = str(exc)
        return base

    column_names = [str(column).strip() for column in df.columns]
    has_timestamp = "TimeStamp" in column_names
    has_eeg_signal = any(
        column.startswith(("Delta_", "Theta_", "Alpha_", "Beta_", "Gamma_", "RAW_"))
        for column in column_names
    )
    schema_valid = has_timestamp and has_eeg_signal

    base.update({
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "column_names": "; ".join(column_names),
        "eeg_schema_valid": schema_valid,
        "read_error": "" if schema_valid else "invalid_eeg_schema",
    })
    return base


def main() -> None:
    raw_index_path = METADATA_DIR / "all_raw_data.csv"
    if not raw_index_path.exists():
        raise FileNotFoundError(f"Missing {raw_index_path}. Run raw_data_inventory.py first.")

    raw = pd.read_csv(raw_index_path, dtype={"subject_id": str})
    eeg_files = raw[raw["modality"] == "eeg"].copy()
    long_df = pd.DataFrame([summarize_eeg_file(row) for _, row in eeg_files.iterrows()])

    EEG_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    long_path = EEG_ANALYSIS_DATA_DIR / "eeg_file_summary_long.csv"
    wide_path = EEG_ANALYSIS_DATA_DIR / "eeg_file_summary_wide.csv"
    long_df.to_csv(long_path, index=False)

    availability = long_df.copy()
    availability["available"] = (
        availability["file_exists"].astype(bool)
        & availability["eeg_schema_valid"].astype(bool)
        & availability["read_error"].fillna("").eq("")
    ).astype(int)

    wide_available = availability.pivot_table(
        index="subject_id", columns="phase", values="available", aggfunc="max", fill_value=0
    )
    wide_available.columns = [f"{phase}_eeg_available" for phase in wide_available.columns]

    wide_rows = long_df.pivot_table(index="subject_id", columns="phase", values="n_rows", aggfunc="first")
    wide_rows.columns = [f"{phase}_eeg_n_rows" for phase in wide_rows.columns]

    pd.concat([wide_available, wide_rows], axis=1).reset_index().to_csv(wide_path, index=False)
    print(f"Saved: {long_path}")
    print(f"Saved: {wide_path}")


if __name__ == "__main__":
    main()
