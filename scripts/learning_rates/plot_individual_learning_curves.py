#!/usr/bin/env python3
"""
plot_individual_learning_curves.py

Merged replacement for:
- individual_plots_with_classification.py
- individual_vs_average.py

For each subject in Session 1, this script:
- Computes observed proportion correct for trials 1–18.
- Reads learner_group directly from learning_rates.csv.
- Creates one individual learning curve plot per subject.
- Overlays the subject's observed data and exponential fit.
- Overlays the group average curve and SEM band.
- Saves a classification CSV with subject, trial, proportion_correct, learner_group, and learning_rate_k.

Inputs:
- data/processed/plearning/
- data/analysis_data/learning_rates/learning_rates.csv

Outputs:
- outputs/learning_rates/figures/individual_learning_curves/
- outputs/learning_rates/tables/individual_learning_curves/
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import sem
from scipy.interpolate import interp1d

sys.path.append(str(Path(__file__).resolve().parents[1]))

from project_paths import (
    PLEARNING_DIR,
    LEARNING_ANALYSIS_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
)


def exp_learning(t, a, b, k):
    return a - b * np.exp(-k * t)


def fit_exponential(x, y):
    y = np.array(y, dtype=float)
    valid = ~np.isnan(y)

    xv = x[valid]
    yv = y[valid]

    if len(xv) < 4:
        return None

    try:
        p0 = [np.max(yv), np.max(yv) - np.min(yv), 0.3]
        bounds = ([0.5, 0, 0], [1, 1, 2])

        popt, _ = curve_fit(
            exp_learning,
            xv,
            yv,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )

        return popt

    except Exception:
        return None


def load_learning_groups(session_num):
    learning_rates_path = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"

    if not learning_rates_path.exists():
        raise FileNotFoundError(
            f"Missing {learning_rates_path}. Run calculate_learning_rates.py first."
        )

    lr_df = pd.read_csv(learning_rates_path, dtype={"subjectId": str})
    lr_df["subjectId"] = lr_df["subjectId"].astype(str).str.zfill(3)
    lr_df["plearning_num"] = lr_df["plearning_num"].astype(str)

    lr_df = lr_df[lr_df["plearning_num"] == str(session_num)].copy()

    required_cols = {
        "subjectId",
        "learning_rate_k",
        "learner_group",
        "classification_basis",
    }

    missing_cols = required_cols - set(lr_df.columns)

    if missing_cols:
        raise ValueError(
            "learning_rates.csv is missing required columns. "
            "Rerun calculate_learning_rates.py. "
            f"Missing: {sorted(missing_cols)}"
        )

    lr_df["learner_group"] = lr_df["learner_group"].astype(str).str.lower()

    group_map = lr_df.set_index("subjectId")["learner_group"].to_dict()
    k_map = lr_df.set_index("subjectId")["learning_rate_k"].to_dict()
    basis_map = lr_df.set_index("subjectId")["classification_basis"].to_dict()

    return group_map, k_map, basis_map


def main():
    plearning_session = 1
    trials_per_block = 18

    output_dir = FIGURES_DIR / "individual_learning_curves"
    table_dir = TABLES_DIR / "individual_learning_curves"

    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    group_map, k_map, basis_map = load_learning_groups(plearning_session)

    x = np.arange(1, trials_per_block + 1)
    dense_x = np.linspace(1, trials_per_block, 200)

    all_subjects = []

    for csv_file in sorted(PLEARNING_DIR.rglob("*_plearning_*.csv")):
        if "_plearning_1_eeg" in csv_file.name or "_plearning_2_eeg" in csv_file.name:
            continue

        parts = csv_file.stem.split("_")

        if len(parts) < 3:
            continue

        subject_id = parts[0].zfill(3)

        try:
            session_num = int(parts[2])
        except Exception:
            continue

        if session_num != plearning_session:
            continue

        df = pd.read_csv(csv_file)

        if "learned" not in df.columns or "trialNumber" not in df.columns:
            print(f"Skipping {csv_file.name}: missing learned or trialNumber")
            continue

        df["learned"] = df["learned"].astype(int)
        df["trialNumber"] = pd.to_numeric(df["trialNumber"], errors="coerce")

        trial_props = [
            df[df["trialNumber"] == t]["learned"].mean()
            for t in range(1, trials_per_block + 1)
        ]

        all_subjects.append({
            "subject_id": subject_id,
            "proportions": trial_props,
            "learner_group": group_map.get(subject_id, "unknown"),
            "session1_learning_rate_k": k_map.get(subject_id, np.nan),
            "classification_basis": basis_map.get(subject_id, "unknown"),
        })

    if not all_subjects:
        raise ValueError("No valid pLearning subject files found.")

    # ---------- Save classification/trial CSV ----------
    classification_rows = []

    for subj in all_subjects:
        for trial_num, proportion in enumerate(subj["proportions"], start=1):
            classification_rows.append({
                "subject_id": subj["subject_id"],
                "session": plearning_session,
                "trial": trial_num,
                "proportion_correct": proportion,
                "learner_group": subj["learner_group"],
                "session1_learning_rate_k": subj["session1_learning_rate_k"],
                "classification_basis": subj["classification_basis"],
            })

    classification_df = pd.DataFrame(classification_rows)

    classification_df = classification_df.sort_values(
        ["subject_id", "session", "trial"]
    ).reset_index(drop=True)

    duplicates = classification_df.duplicated(
        subset=["subject_id", "session", "trial"],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate subject/session/trial rows found in "
            "individual_data_with_classification_sess1.csv:\n"
            + classification_df.loc[
                duplicates,
                ["subject_id", "session", "trial"]
            ].to_string(index=False)
        )

    expected_rows = classification_df["subject_id"].nunique() * trials_per_block

    if len(classification_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows in classification CSV, "
            f"found {len(classification_df)}."
        )

    classification_path = (
        table_dir / f"individual_data_with_classification_sess{plearning_session}.csv"
    )

    classification_df.to_csv(classification_path, index=False)

    print(f"Saved classification CSV: {classification_path}")

    # ---------- Group average ----------
    props_array = np.array([s["proportions"] for s in all_subjects], dtype=float)

    group_mean = np.nanmean(props_array, axis=0)
    group_sem = sem(props_array, axis=0, nan_policy="omit")

    group_fit = fit_exponential(x, group_mean)

    if group_fit is not None:
        group_smooth = exp_learning(dense_x, *group_fit)

        sem_valid = ~np.isnan(group_sem)

        if np.sum(sem_valid) >= 2:
            interp_sem = interp1d(
                x[sem_valid],
                group_sem[sem_valid],
                kind="linear",
                fill_value="extrapolate",
            )
            group_sem_smooth = interp_sem(dense_x)
        else:
            group_sem_smooth = np.full_like(dense_x, np.nanmean(group_sem))
    else:
        group_smooth = None
        group_sem_smooth = None

    # ---------- Individual plots ----------
    for subj in all_subjects:
        subject_id = subj["subject_id"]
        y = np.array(subj["proportions"], dtype=float)
        learner_group = subj["learner_group"]
        learning_rate_k = subj["session1_learning_rate_k"]
        classification_basis = subj["classification_basis"]

        valid = ~np.isnan(y)
        xv = x[valid]
        yv = y[valid]

        subject_fit = fit_exponential(x, y)

        plt.figure(figsize=(10, 6))

        # Subject observed values
        plt.scatter(
            xv,
            yv,
            color="blue",
            s=50,
            edgecolor="black",
            zorder=5,
            label=f"Subject {subject_id} observed",
        )

        # Subject fit
        if subject_fit is not None:
            subject_smooth = exp_learning(dense_x, *subject_fit)
            plt.plot(
                dense_x,
                subject_smooth,
                color="blue",
                linewidth=2,
                label=f"Subject {subject_id} exponential fit",
            )

            a, b, k = subject_fit
            fit_text = (
                f"Subject fit:\n"
                f"a = {a:.2f}\n"
                f"b = {b:.2f}\n"
                f"k = {k:.3f}"
            )
        else:
            fit_text = "Subject fit failed"

        # Group average observed
        plt.scatter(
            x,
            group_mean,
            color="gray",
            s=40,
            edgecolor="black",
            zorder=4,
            label="Group average observed",
        )

        # Group average fit
        if group_smooth is not None:
            plt.plot(
                dense_x,
                group_smooth,
                color="gray",
                linewidth=2,
                linestyle="--",
                label="Group average exponential fit",
            )

            plt.fill_between(
                dense_x,
                group_smooth - group_sem_smooth,
                group_smooth + group_sem_smooth,
                color="gray",
                alpha=0.2,
                label="Group SEM",
            )

        plt.axhline(0.5, color="gray", linestyle=":", label="Chance")

        plt.xlabel("Trial number")
        plt.ylabel("Proportion choosing set-winner")
        plt.ylim(0, 1)
        plt.xlim(0.9, trials_per_block + 0.1)
        plt.xticks(range(1, trials_per_block + 1))

        plt.title(
            f"Subject {subject_id} — {learner_group.capitalize()} learner\n"
            f"Individual learning curve vs group average"
        )

        if pd.notna(learning_rate_k):
            k_text = f"{learning_rate_k:.3f}"
        else:
            k_text = "NA"

        annotation = (
            f"Learner group: {learner_group}\n"
            f"Session 1 k: {k_text}\n"
            f"Basis: {classification_basis}\n\n"
            f"{fit_text}"
        )

        plt.annotate(
            annotation,
            xy=(0.04, 0.96),
            xycoords="axes fraction",
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = output_dir / f"subject_{subject_id}_session{plearning_session}_learning_curve.png"

        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved: {output_path}")

    print("\nDone.")
    print(f"Figures saved in: {output_dir}")
    print(f"CSV saved in: {classification_path}")


if __name__ == "__main__":
    main()