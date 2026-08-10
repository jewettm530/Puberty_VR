#!/usr/bin/env python3
"""
Processes all subjects' trial-level pLearning CSV files.

Outputs:
- individual_proportions_session1.csv
- individual_proportions_session2.csv
- trial1_trial18_raw_vs_fitted_diagnostics.csv
- all-individual learning curve figures

Important:
- Fitted diagnostics use fit_a, fit_b, and learning_rate_k from learning_rates.csv.
- Run calculate_learning_rates.py before this script.
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
from scipy.stats import sem, ttest_rel
from scipy.interpolate import interp1d

sys.path.append(str(Path(__file__).resolve().parents[1]))

from file_naming import parse_plearning_csv_name
from learning_utils import exp_learning
from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR


# ---------- MODEL ----------


def fit_subject_curve(x, y):
    y = np.array(y, dtype=float)
    valid = ~np.isnan(y)
    xv, yv = x[valid], y[valid]

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


def paired_stats(t1_vals, t18_vals):
    t1_vals = np.array(t1_vals, dtype=float)
    t18_vals = np.array(t18_vals, dtype=float)

    valid = ~np.isnan(t1_vals) & ~np.isnan(t18_vals)
    t1_vals = t1_vals[valid]
    t18_vals = t18_vals[valid]

    n = len(t1_vals)

    if n < 2:
        return {
            "n": n,
            "trial1_mean": np.nan,
            "trial1_sd": np.nan,
            "trial18_mean": np.nan,
            "trial18_sd": np.nan,
            "df": np.nan,
            "t": np.nan,
            "p": np.nan,
            "d": np.nan,
        }

    t_stat, p_val = ttest_rel(t18_vals, t1_vals)
    diff = t18_vals - t1_vals
    cohens_d = diff.mean() / diff.std(ddof=1)

    return {
        "n": n,
        "trial1_mean": t1_vals.mean(),
        "trial1_sd": t1_vals.std(ddof=1),
        "trial18_mean": t18_vals.mean(),
        "trial18_sd": t18_vals.std(ddof=1),
        "df": n - 1,
        "t": t_stat,
        "p": p_val,
        "d": cohens_d,
    }


def format_p(p):
    if pd.isna(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def manuscript_sentence(stats):
    if pd.isna(stats["df"]):
        return "Not enough paired observations to compute a manuscript-style statistic."

    return (
        f"Participants demonstrated successful acquisition of reward contingencies, "
        f"with optimal choices increasing from {stats['trial1_mean'] * 100:.1f}% "
        f"(SD = {stats['trial1_sd'] * 100:.1f}%) on the first trial to "
        f"{stats['trial18_mean'] * 100:.1f}% "
        f"(SD = {stats['trial18_sd'] * 100:.1f}%) on the final trial, "
        f"t({int(stats['df'])}) = {stats['t']:.2f}, "
        f"{format_p(stats['p'])}, "
        f"d = {stats['d']:.2f}."
    )


def print_stats(label, stats):
    print(f"\n{label}")
    print("-" * len(label))
    print(f"n: {stats['n']}")
    print(f"Trial 1 mean: {stats['trial1_mean']:.4f}")
    print(f"Trial 1 SD: {stats['trial1_sd']:.4f}")
    print(f"Trial 18 mean: {stats['trial18_mean']:.4f}")
    print(f"Trial 18 SD: {stats['trial18_sd']:.4f}")
    print(f"df: {stats['df']}")
    print(f"t: {stats['t']:.4f}")
    print(f"p: {stats['p']:.6f}")
    print(f"d: {stats['d']:.4f}")
    print(manuscript_sentence(stats))


# ---------- SETTINGS ----------
MAX_LEGEND_SUBJECTS = 20
ORIGINAL_GOOD_LEARNERS = ["005", "007", "012", "014", "019", "020", "022"]

RESULTS_DIR = PLEARNING_DIR
OUTPUT_DIR = FIGURES_DIR / "all_individuals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEARNING_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)

trials_per_block = 18
sessions = [1, 2]
x = np.arange(1, trials_per_block + 1)
dense_x = np.linspace(1, 18, 200)


# ---------- LOAD learning_rates.csv ----------
learning_rates_path = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"

if not learning_rates_path.exists():
    raise FileNotFoundError(
        f"Missing {learning_rates_path}. Run calculate_learning_rates.py first."
    )

lr_df = pd.read_csv(learning_rates_path, dtype={"subjectId": str})
lr_df["subjectId"] = lr_df["subjectId"].str.zfill(3)
lr_df["plearning_num"] = lr_df["plearning_num"].astype(str)

required_fit_cols = {
    "subjectId",
    "plearning_num",
    "learning_rate_k",
    "fit_a",
    "fit_b",
    "fit_status",
}
missing_cols = required_fit_cols - set(lr_df.columns)

if missing_cols:
    raise ValueError(
        "learning_rates.csv is missing fitted-curve columns. "
        "Run the updated calculate_learning_rates.py first. "
        f"Missing columns: {sorted(missing_cols)}"
    )

fit_lookup = lr_df.set_index(["subjectId", "plearning_num"])


# ---------- LOAD DATA FOR BOTH SESSIONS ----------
all_data_by_session = {sess: [] for sess in sessions}

for csv_file in RESULTS_DIR.rglob("*_plearning_*.csv"):
    parsed_name = parse_plearning_csv_name(csv_file)
    if parsed_name is None:
        continue
    subj_id, sess = parsed_name

    if sess not in sessions:
        continue

    df = pd.read_csv(csv_file)

    if "learned" not in df.columns or "trialNumber" not in df.columns:
        continue

    df["learned"] = pd.to_numeric(df["learned"], errors="coerce")
    df["trialNumber"] = pd.to_numeric(df["trialNumber"], errors="coerce")
    df = df.dropna(subset=["learned", "trialNumber"])

    trial_props = [
        df[df["trialNumber"] == t]["learned"].mean()
        for t in range(1, trials_per_block + 1)
    ]

    all_data_by_session[sess].append({
        "subject": subj_id,
        "session": sess,
        "proportions": trial_props,
    })


# ---------- PROCESS EACH SESSION ----------
diagnostic_rows = []

for sess in sessions:
    if not all_data_by_session[sess]:
        print(f"Warning: No data for session {sess}, skipping.")
        continue

    data = sorted(all_data_by_session[sess], key=lambda d: d["subject"])
    subjects = [d["subject"] for d in data]
    n_subj = len(subjects)

    # ---------- SAVE RAW PROPORTIONS CSV ----------
    rows = []

    for subj in data:
        for t, p in enumerate(subj["proportions"], start=1):
            rows.append({
                "subject": subj["subject"],
                "session": sess,
                "trial": t,
                "proportion_correct": p,
            })

    df_raw = pd.DataFrame(rows)
    raw_csv_path = LEARNING_ANALYSIS_DATA_DIR / f"individual_proportions_session{sess}.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Saved raw proportions: {raw_csv_path}")

    # ---------- DIAGNOSTICS USING SOURCE-OF-TRUTH FITS ----------
    for subj in data:
        y = np.array(subj["proportions"], dtype=float)

        raw_t1 = y[0]
        raw_t18 = y[-1]

        key = (subj["subject"], str(sess))

        if key in fit_lookup.index:
            fit_row = fit_lookup.loc[key]

            fit_a = fit_row["fit_a"]
            fit_b = fit_row["fit_b"]
            fit_k = fit_row["learning_rate_k"]
            fit_status = fit_row["fit_status"]

            if pd.notna(fit_a) and pd.notna(fit_b) and pd.notna(fit_k):
                fitted_t1 = exp_learning(1, fit_a, fit_b, fit_k)
                fitted_t18 = exp_learning(18, fit_a, fit_b, fit_k)
            else:
                fitted_t1 = np.nan
                fitted_t18 = np.nan
        else:
            fit_a = np.nan
            fit_b = np.nan
            fit_k = np.nan
            fit_status = "missing_from_learning_rates"
            fitted_t1 = np.nan
            fitted_t18 = np.nan

        diagnostic_rows.append({
            "subject": subj["subject"],
            "session": str(sess),
            "raw_trial1": raw_t1,
            "raw_trial18": raw_t18,
            "fitted_trial1": fitted_t1,
            "fitted_trial18": fitted_t18,
            "fit_a": fit_a,
            "fit_b": fit_b,
            "fit_k": fit_k,
            "fit_status": fit_status,
        })

    # ---------- PLOTTING COLORS ----------
    if n_subj <= 10:
        colormap = plt.cm.tab10
    else:
        colormap = plt.cm.tab20

    colors = [colormap(i % colormap.N) for i in range(n_subj)]
    subject_color = {
        subj["subject"]: colors[idx]
        for idx, subj in enumerate(data)
    }

    # ---------- GRAPH 1: Individual points + individual fits ----------
    plt.figure(figsize=(12, 6))

    for subj in data:
        y = np.array(subj["proportions"], dtype=float)
        color = subject_color[subj["subject"]]

        popt = fit_subject_curve(x, y)

        if popt is not None:
            y_smooth = exp_learning(dense_x, *popt)
            plt.plot(
                dense_x,
                y_smooth,
                linewidth=1,
                alpha=0.6,
                color=color,
                label="_nolegend_",
            )

        plt.plot(
            x,
            y,
            "o",
            markersize=3,
            alpha=0.7,
            color=color,
            label=subj["subject"] if n_subj <= MAX_LEGEND_SUBJECTS else "_nolegend_",
        )

    plt.axhline(0.5, color="gray", linestyle="--", label="Chance")
    plt.xlabel("Trial number")
    plt.ylabel("Proportion choosing set-winner")
    plt.ylim(0, 1)
    plt.xlim(0.9, 18.1)
    plt.xticks(range(1, trials_per_block + 1))
    plt.title(
        f"Session {sess} – All individuals "
        f"(points = observed, lines = individual exponential fits)"
    )

    if n_subj <= MAX_LEGEND_SUBJECTS:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        print(f"Session {sess}: Too many subjects ({n_subj}) to show legend; legend omitted.")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"all_individuals_with_curves_sess{sess}.png", dpi=150)
    plt.close()

    # ---------- GRAPH 2: All individual points + group average curve ----------
    props_array = np.array([d["proportions"] for d in data], dtype=float)
    group_mean = np.nanmean(props_array, axis=0)
    group_sem = sem(props_array, axis=0, nan_policy="omit")

    plt.figure(figsize=(12, 6))

    for subj in data:
        y = np.array(subj["proportions"], dtype=float)
        color = subject_color[subj["subject"]]

        plt.plot(
            x,
            y,
            "o",
            markersize=3,
            alpha=0.5,
            color=color,
            label=subj["subject"] if n_subj <= MAX_LEGEND_SUBJECTS else "_nolegend_",
        )

    valid = ~np.isnan(group_mean)
    xv, yv = x[valid], group_mean[valid]

    if len(xv) >= 4:
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
            y_smooth = exp_learning(dense_x, *popt)

            plt.plot(
                dense_x,
                y_smooth,
                color="blue",
                linewidth=2,
                label="Group average exponential fit",
            )

            sem_valid = ~np.isnan(group_sem)

            if np.sum(sem_valid) >= 2:
                interp_sem = interp1d(
                    x[sem_valid],
                    group_sem[sem_valid],
                    kind="linear",
                    fill_value="extrapolate",
                )
                sem_smooth = interp_sem(dense_x)

                plt.fill_between(
                    dense_x,
                    y_smooth - sem_smooth,
                    y_smooth + sem_smooth,
                    alpha=0.2,
                    color="blue",
                )

        except Exception as e:
            print(f"Group fit failed for session {sess}: {e}")
            plt.plot(xv, yv, "o-", color="blue", label="Group average observed")
    else:
        plt.plot(xv, yv, "o-", color="blue", label="Group average observed")

    plt.scatter(
        xv,
        yv,
        color="blue",
        s=40,
        edgecolor="black",
        zorder=5,
        label="_nolegend_",
    )

    plt.axhline(0.5, color="gray", linestyle="--", label="Chance")
    plt.xlabel("Trial number")
    plt.ylabel("Proportion choosing set-winner")
    plt.ylim(0, 1)
    plt.xlim(0.9, 18.1)
    plt.xticks(range(1, trials_per_block + 1))
    plt.title(f"Session {sess} – All individuals + group average curve")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"all_individuals_with_group_avg_sess{sess}.png", dpi=150)
    plt.close()


# ---------- SAVE DIAGNOSTIC FITTED VS RAW VALUES ----------
diagnostic_df = pd.DataFrame(diagnostic_rows)
diagnostic_df["subject"] = diagnostic_df["subject"].astype(str).str.zfill(3)
diagnostic_df["session"] = diagnostic_df["session"].astype(str)

diagnostic_path = LEARNING_ANALYSIS_DATA_DIR / "trial1_trial18_raw_vs_fitted_diagnostics.csv"
diagnostic_df.to_csv(diagnostic_path, index=False)
print(f"\nSaved diagnostic raw vs fitted values: {diagnostic_path}")


# ---------- VALIDATE DIAGNOSTIC K VALUES ----------
lr_validation = lr_df.copy()
lr_validation["subject"] = lr_validation["subjectId"].astype(str).str.zfill(3)
lr_validation["session"] = lr_validation["plearning_num"].astype(str)

validation = diagnostic_df.merge(
    lr_validation[["subject", "session", "learning_rate_k"]],
    on=["subject", "session"],
    how="left",
)

validation = validation.rename(columns={
    "learning_rate_k": "learning_rate_k_source"
})

k_mismatch = validation[
    ~np.isclose(
        validation["fit_k"].astype(float),
        validation["learning_rate_k_source"].astype(float),
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    )
]

if not k_mismatch.empty:
    raise ValueError(
        "fit_k mismatch between diagnostics and learning_rates.csv:\n"
        + k_mismatch[
            ["subject", "session", "fit_k", "learning_rate_k_source"]
        ].to_string(index=False)
    )


# ---------- PRINT RAW VS FITTED SUMMARY STATS ----------
print("\n" + "=" * 80)
print("RAW VS FITTED TRIAL 1/TRIAL 18 DIAGNOSTICS")
print("=" * 80)

current_good_learners = (
    lr_df[
        (lr_df["plearning_num"] == "1")
        & (lr_df["learner_group"].astype(str).str.lower() == "good")
    ]["subjectId"]
    .dropna()
    .astype(str)
    .str.zfill(3)
    .unique()
    .tolist()
)


def summarize_group(group_label, subject_list, session=1):
    session = str(session)

    subject_list = [
        str(s).zfill(3)
        for s in subject_list
    ]

    group_df = diagnostic_df[
        (diagnostic_df["session"] == session)
        & (diagnostic_df["subject"].isin(subject_list))
    ].copy()

    if group_df.empty:
        print(f"\n{group_label}: no matching subjects found for session {session}.")
        return

    group_df = group_df.sort_values("subject")

    print(f"\n{group_label} included:")
    print(group_df["subject"].tolist())

    raw_stats = paired_stats(group_df["raw_trial1"], group_df["raw_trial18"])
    fitted_stats = paired_stats(group_df["fitted_trial1"], group_df["fitted_trial18"])

    print_stats(f"{group_label} — RAW observed Trial 1 vs Trial 18", raw_stats)
    print_stats(f"{group_label} — FITTED exponential Trial 1 vs Trial 18", fitted_stats)


all_session1_subjects = (
    diagnostic_df[diagnostic_df["session"] == "1"]["subject"]
    .dropna()
    .astype(str)
    .str.zfill(3)
    .unique()
    .tolist()
)

summarize_group("Original good learners", ORIGINAL_GOOD_LEARNERS, session=1)
summarize_group("Current good learners", current_good_learners, session=1)
summarize_group("All Session 1 subjects", all_session1_subjects, session=1)

print(f"\nAll outputs saved in {OUTPUT_DIR}")