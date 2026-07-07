"""
calculate_learning_rates.py

This script processes each subject's pLearning behavioral CSV file from the
Puberty_VR study and computes an individual learning rate (k).

For each subject/session:
- Reads the CSV file from data/processed/plearning/.
- Requires columns: trialNumber and learned.
- Aggregates accuracy by trial position 1–18 across blocks.
- Fits an exponential learning curve:

      P(t) = a - b * exp(-k*t)

- Extracts k as the learning-rate parameter.
- Records total trials, mean accuracy, and fit status.

Learner classification:
- Good/bad learner classification is based on Session 1 learning_rate_k.
- The Session 1 median k is used as the cutoff.
- Subjects with Session 1 k >= median are classified as good learners.
- Subjects with Session 1 k < median are classified as bad learners.

Output:
- data/analysis_data/learning_rates/learning_rates.csv

Output columns include:
    subjectId
    plearning_num
    learning_rate_k
    n_trials
    mean_accuracy
    fit_status
    classification_learning_rate_k
    median_session1_learning_rate_k
    learner_group
    classification_basis

Dependencies: pandas, numpy, scipy.
"""
#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.append(str(Path(__file__).resolve().parents[1]))

from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR


def exp_learning(t, a, b, k):
    """
    Exponential learning curve:
    P(t) = a - b * exp(-k*t)

    k = learning rate.
    Higher k means faster learning.
    """
    return a - b * np.exp(-k * t)


def clean_learned_column(series):
    if series.dtype == bool:
        return series.astype(int)

    if series.dtype == object:
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
            })
        )

    return pd.to_numeric(series, errors="coerce")


def compute_learning_rate(df):
    df = df.copy()

    df["learned_numeric"] = clean_learned_column(df["learned"])
    df["trialNumber"] = pd.to_numeric(df["trialNumber"], errors="coerce")

    df = df.dropna(subset=["learned_numeric", "trialNumber"])

    if df.empty:
        return np.nan, np.nan, np.nan, "no_valid_trials"

    trial_accuracy = (
        df.groupby("trialNumber")["learned_numeric"]
        .mean()
        .reset_index()
        .sort_values("trialNumber")
    )

    x = trial_accuracy["trialNumber"].to_numpy(dtype=float)
    y = trial_accuracy["learned_numeric"].to_numpy(dtype=float)

    if len(x) < 4:
        return np.nan, np.nan, np.nan, "not_enough_trial_points"

    try:
        a0 = min(max(np.nanmax(y), 0.51), 1.0)
        b0 = max(a0 - np.nanmin(y), 0.01)
        k0 = 0.05

        bounds = (
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 2.0],
        )

        popt, _ = curve_fit(
            exp_learning,
            x,
            y,
            p0=[a0, b0, k0],
            bounds=bounds,
            maxfev=10000,
        )

        fit_a, fit_b, fit_k = popt
        return fit_a, fit_b, fit_k, "fit_success"

    except Exception as e:
        return np.nan, np.nan, np.nan, f"fit_failed: {e}"


def classify_learners(out_df):
    """
    Classify subjects using Session 1 learning_rate_k median split.
    Good learner = Session 1 k >= median Session 1 k.
    Bad learner = Session 1 k < median Session 1 k.

    If a subject's Session 1 k is NaN, fallback to Session 1 mean_accuracy.
    """
    out_df = out_df.copy()

    session1 = out_df[out_df["plearning_num"] == "1"].copy()

    valid_k = session1["learning_rate_k"].dropna()

    if valid_k.empty:
        out_df["learner_group"] = "unclassified"
        out_df["classification_basis"] = "no_valid_session1_k"
        return out_df

    median_k = valid_k.median()

    classification_rows = []

    for _, row in session1.iterrows():
        subject = row["subjectId"]
        k = row["learning_rate_k"]

        if pd.notna(k):
            group = "good" if k >= median_k else "bad"
            basis = "session1_learning_rate_k"
        else:
            group = "unclassified"
            basis = "missing_session1_learning_rate_k"

        classification_rows.append({
            "subjectId": subject,
            "classification_learning_rate_k": k,
            "median_session1_learning_rate_k": median_k,
            "learner_group": group,
            "classification_basis": basis,
        })

    classifications = pd.DataFrame(classification_rows)

    out_df = out_df.merge(classifications, on="subjectId", how="left")

    return out_df


def main():
    csv_files = sorted(PLEARNING_DIR.rglob("*_plearning_*.csv"))

    # Exclude EEG files if they exist in the same folder
    csv_files = [
        f for f in csv_files
        if "_plearning_1_eeg" not in f.name
        and "_plearning_2_eeg" not in f.name
        and "subject_info_summary" not in f.name
    ]

    results = []

    for csv_file in csv_files:
        parts = csv_file.stem.split("_")

        if len(parts) < 3:
            continue

        subject_id = parts[0].zfill(3)
        plearning_num = parts[2]

        df = pd.read_csv(csv_file)

        if "learned" not in df.columns or "trialNumber" not in df.columns:
            print(f"Skipping {csv_file}: missing 'learned' or 'trialNumber'")
            continue

        learned_numeric = clean_learned_column(df["learned"])

        fit_a, fit_b, k, fit_status = compute_learning_rate(df)

        results.append({
            "subjectId": subject_id,
            "plearning_num": str(plearning_num),
            "learning_rate_k": k,
            "fit_a": fit_a,
            "fit_b": fit_b,
            "n_trials": len(df),
            "mean_accuracy": learned_numeric.mean(),
            "fit_status": fit_status,
        })

    if not results:
        print("No valid CSV files found.")
        return

    out_df = pd.DataFrame(results)

    out_df = classify_learners(out_df)

    out_df = out_df.sort_values(["subjectId", "plearning_num"]).reset_index(drop=True)

    LEARNING_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"

    out_df.to_csv(output_path, index=False)

    print(f"Saved learning rates to {output_path}")
    print(out_df)


if __name__ == "__main__":
    main()