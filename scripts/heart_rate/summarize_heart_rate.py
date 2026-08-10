#!/usr/bin/env python3
"""Summarize heart-rate sample files listed by raw_data_inventory.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_paths import (  # noqa: E402
    HEART_RATE_ANALYSIS_DATA_DIR,
    METADATA_DIR,
    resolve_project_path,
)


def normalized_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def make_unique_headers(values: list[object]) -> list[str]:
    headers: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(values):
        text = str(value).strip() if pd.notna(value) else ""
        base = text or f"unnamed_{index}"
        count = counts.get(base, 0)
        counts[base] = count + 1
        headers.append(base if count == 0 else f"{base}_{count + 1}")
    return headers


def read_heart_rate_export(file_path: Path) -> tuple[pd.DataFrame, str]:
    """Read either a normal CSV or a Polar CSV with an embedded second header."""
    raw = pd.read_csv(file_path, header=None, dtype=str, low_memory=False)

    header_row_index: int | None = None
    for index, row in raw.head(20).iterrows():
        normalized = {normalized_header(value) for value in row.dropna().tolist()}
        if normalized & {"hrbpm", "heartrate", "heartratebpm", "bpm"}:
            header_row_index = int(index)
            break

    if header_row_index is not None:
        headers = make_unique_headers(raw.iloc[header_row_index].tolist())
        data = raw.iloc[header_row_index + 1 :].copy()
        data.columns = headers
        data = data.dropna(axis=1, how="all").dropna(axis=0, how="all")
        return data, "embedded_header"

    # Fall back to an ordinary one-header CSV.
    return pd.read_csv(file_path, low_memory=False), "standard_header"


def find_hr_column(df: pd.DataFrame) -> str | None:
    preferred = {
        "hrbpm",
        "heartrate",
        "heartratebpm",
        "heart_rate",
        "heartratevalue",
        "bpm",
    }
    for column in df.columns:
        if normalized_header(column) in preferred:
            return str(column)

    # Conservative numeric fallback: require many plausible sample values so a
    # one-row metadata field (calories, HR max, etc.) cannot be selected.
    candidates: list[tuple[int, float, str]] = []
    for column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        plausible = values[values.between(25, 250)]
        if len(plausible) >= 30 and len(plausible) / max(len(values), 1) >= 0.80:
            candidates.append((len(plausible), plausible.std(), str(column)))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def summarize_file(row: pd.Series) -> dict[str, object]:
    stored_path = Path(row["file_path"])
    file_path = resolve_project_path(stored_path)

    base: dict[str, object] = {
        "subject_id": str(row["subject_id"]).zfill(3),
        "file_key": row["file_key"],
        "phase": row["phase"],
        "modality": row["modality"],
        "file_path": str(stored_path),
        "file_exists": file_path.exists(),
        "source_format": "",
        "hr_column": "",
        "n_samples": 0,
        "n_invalid_or_out_of_range": 0,
        "mean_hr": np.nan,
        "median_hr": np.nan,
        "min_hr": np.nan,
        "max_hr": np.nan,
        "sd_hr": np.nan,
        "read_error": "",
    }

    if not file_path.exists():
        base["read_error"] = "file_not_found"
        return base

    try:
        df, source_format = read_heart_rate_export(file_path)
    except Exception as exc:
        base["read_error"] = str(exc)
        return base

    base["source_format"] = source_format
    hr_column = find_hr_column(df)
    base["hr_column"] = hr_column or ""
    if hr_column is None:
        base["read_error"] = "No heart-rate sample column found"
        return base

    numeric = pd.to_numeric(df[hr_column], errors="coerce")
    valid = numeric[numeric.between(25, 250)].dropna()
    base["n_invalid_or_out_of_range"] = int(numeric.notna().sum() - len(valid))

    if valid.empty:
        base["read_error"] = "Heart-rate column contains no valid samples"
        return base

    base.update({
        "n_samples": int(len(valid)),
        "mean_hr": float(valid.mean()),
        "median_hr": float(valid.median()),
        "min_hr": float(valid.min()),
        "max_hr": float(valid.max()),
        "sd_hr": float(valid.std()),
    })
    return base


def main() -> None:
    raw_index_path = METADATA_DIR / "all_raw_data.csv"
    if not raw_index_path.exists():
        raise FileNotFoundError(f"Missing {raw_index_path}. Run raw_data_inventory.py first.")

    raw = pd.read_csv(raw_index_path, dtype={"subject_id": str})
    hr_files = raw[raw["modality"] == "heart_rate"].copy()
    long_df = pd.DataFrame([summarize_file(row) for _, row in hr_files.iterrows()])

    HEART_RATE_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    long_path = HEART_RATE_ANALYSIS_DATA_DIR / "heart_rate_summary_long.csv"
    wide_path = HEART_RATE_ANALYSIS_DATA_DIR / "heart_rate_summary_wide.csv"
    long_df.to_csv(long_path, index=False)

    value_columns = ["mean_hr", "median_hr", "min_hr", "max_hr", "sd_hr", "n_samples"]
    wide_parts = []
    for value in value_columns:
        pivot = long_df.pivot_table(index="subject_id", columns="phase", values=value, aggfunc="first")
        pivot.columns = [f"{phase}_{value}" for phase in pivot.columns]
        wide_parts.append(pivot)

    wide_df = pd.concat(wide_parts, axis=1).reset_index() if wide_parts else pd.DataFrame()

    derived = {
        "speech_hr_reactivity": ("vr_speech_mean_hr", "pre_vr_mean_hr"),
        "math_hr_reactivity": ("vr_math_mean_hr", "pre_vr_mean_hr"),
        "prep_hr_reactivity": ("vr_prep_mean_hr", "pre_vr_mean_hr"),
        "post_minus_pre_hr": ("post_vr_mean_hr", "pre_vr_mean_hr"),
        "recovery_minus_post_hr": ("recovery_mean_hr", "post_vr_mean_hr"),
    }
    for output_column, (left, right) in derived.items():
        if left in wide_df.columns and right in wide_df.columns:
            wide_df[output_column] = wide_df[left] - wide_df[right]

    wide_df.to_csv(wide_path, index=False)
    print(f"Saved: {long_path}")
    print(f"Saved: {wide_path}")


if __name__ == "__main__":
    main()
