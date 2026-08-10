#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

sys.path.append(str(Path(__file__).resolve().parents[1]))

from project_paths import LEARNING_ANALYSIS_DATA_DIR, REPORTS_DIR


def format_p(p):
    if pd.isna(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def paired_stats(paired):
    paired = paired.dropna(subset=[1, 18]).copy()
    trial1 = paired[1].to_numpy(dtype=float)
    trial18 = paired[18].to_numpy(dtype=float)
    n = len(paired)

    if n >= 2:
        t_stat, p_val = ttest_rel(trial18, trial1)
        diff_sd = np.std(trial18 - trial1, ddof=1)
        cohens_d = np.mean(trial18 - trial1) / diff_sd if diff_sd > 0 else np.nan
        trial1_sd = np.std(trial1, ddof=1)
        trial18_sd = np.std(trial18, ddof=1)
    else:
        t_stat = p_val = cohens_d = np.nan
        trial1_sd = trial18_sd = np.nan

    return {
        "n": n,
        "df": n - 1 if n else 0,
        "trial1_mean": np.mean(trial1) if n else np.nan,
        "trial1_sd": trial1_sd,
        "trial18_mean": np.mean(trial18) if n else np.nan,
        "trial18_sd": trial18_sd,
        "t": t_stat,
        "p": p_val,
        "d": cohens_d,
    }


def manuscript_sentence(stats):
    if stats["n"] < 2:
        return "Insufficient paired observations to calculate a paired t-test."
    d_text = "NA" if pd.isna(stats["d"]) else f"{stats['d']:.2f}"
    return (
        "Participants demonstrated successful acquisition of reward contingencies, "
        f"with optimal choices increasing from {stats['trial1_mean'] * 100:.1f}% "
        f"(SD = {stats['trial1_sd'] * 100:.1f}%) on the first trial to "
        f"{stats['trial18_mean'] * 100:.1f}% "
        f"(SD = {stats['trial18_sd'] * 100:.1f}%) on the final trial, "
        f"t({stats['df']}) = {stats['t']:.2f}, {format_p(stats['p'])}, d = {d_text}."
    )


def fmt(value, decimals=4):
    return "NA" if pd.isna(value) else f"{value:.{decimals}f}"


def print_stats(label, stats):
    print("\n" + label)
    print("-" * len(label))
    print(f"n: {stats['n']}")
    print(f"Trial 1 mean: {fmt(stats['trial1_mean'])}")
    print(f"Trial 1 SD: {fmt(stats['trial1_sd'])}")
    print(f"Trial 18 mean: {fmt(stats['trial18_mean'])}")
    print(f"Trial 18 SD: {fmt(stats['trial18_sd'])}")
    print(f"df: {stats['df']}")
    print(f"t: {fmt(stats['t'])}")
    print(f"p: {fmt(stats['p'], 6)}")
    print(f"d: {fmt(stats['d'])}")
    print(manuscript_sentence(stats))


def main():
    props_path = LEARNING_ANALYSIS_DATA_DIR / "individual_proportions_session1.csv"
    groups_path = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"

    if not props_path.exists():
        raise FileNotFoundError(f"Missing {props_path}. Run all_individuals_with_curve.py first.")
    if not groups_path.exists():
        raise FileNotFoundError(f"Missing {groups_path}. Run calculate_learning_rates.py first.")

    props = pd.read_csv(props_path, dtype={"subject": str})
    groups = pd.read_csv(groups_path, dtype={"subjectId": str})
    props["subject"] = props["subject"].astype(str).str.zfill(3)
    props["trial"] = pd.to_numeric(props["trial"], errors="coerce")
    groups["subjectId"] = groups["subjectId"].astype(str).str.zfill(3)
    groups["plearning_num"] = pd.to_numeric(groups["plearning_num"], errors="coerce")

    session1_groups = groups[groups["plearning_num"] == 1].copy()
    learner_group = session1_groups["learner_group"].astype(str).str.lower()
    good_subjects = session1_groups.loc[learner_group == "good", "subjectId"].dropna().unique().tolist()
    bad_subjects = session1_groups.loc[learner_group == "bad", "subjectId"].dropna().unique().tolist()
    all_subjects = sorted(props["subject"].dropna().unique().tolist())

    summary_rows = []
    for label, subject_list in [
        ("Good learners", good_subjects),
        ("Bad learners", bad_subjects),
        ("All Session 1 subjects", all_subjects),
    ]:
        group_props = props[props["subject"].isin(subject_list)].copy()
        paired = (
            group_props[group_props["trial"].isin([1, 18])]
            .pivot_table(index="subject", columns="trial", values="proportion_correct", aggfunc="first")
            .reindex(columns=[1, 18])
        )
        stats = paired_stats(paired)
        print_stats(label, stats)
        summary_rows.append({"group": label, **stats, "sentence": manuscript_sentence(stats)})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "learning_summary_statistics.txt"
    with output_path.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(row["group"] + "\n")
            f.write("-" * len(row["group"]) + "\n")
            f.write(f"n: {row['n']}\n")
            f.write(f"Trial 1 mean: {fmt(row['trial1_mean'])}\n")
            f.write(f"Trial 1 SD: {fmt(row['trial1_sd'])}\n")
            f.write(f"Trial 18 mean: {fmt(row['trial18_mean'])}\n")
            f.write(f"Trial 18 SD: {fmt(row['trial18_sd'])}\n")
            f.write(f"df: {row['df']}\n")
            f.write(f"t: {fmt(row['t'])}\n")
            f.write(f"p: {fmt(row['p'], 6)}\n")
            f.write(f"d: {fmt(row['d'])}\n")
            f.write(row["sentence"] + "\n\n")
    print(f"\nSaved summary statistics to: {output_path}")


if __name__ == "__main__":
    main()
