#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

sys.path.append(str(Path(__file__).resolve().parents[1]))

from project_paths import LEARNING_ANALYSIS_DATA_DIR, REPORTS_DIR


def format_p(p):
    if pd.isna(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def paired_stats(trial1, trial18):
    trial1 = np.array(trial1, dtype=float)
    trial18 = np.array(trial18, dtype=float)

    valid = ~np.isnan(trial1) & ~np.isnan(trial18)
    trial1 = trial1[valid]
    trial18 = trial18[valid]

    t_stat, p_val = ttest_rel(trial18, trial1)

    diff = trial18 - trial1
    cohens_d = diff.mean() / diff.std(ddof=1)

    return {
        "n": len(trial1),
        "df": len(trial1) - 1,
        "trial1_mean": trial1.mean(),
        "trial1_sd": trial1.std(ddof=1),
        "trial18_mean": trial18.mean(),
        "trial18_sd": trial18.std(ddof=1),
        "t": t_stat,
        "p": p_val,
        "d": cohens_d,
    }


def manuscript_sentence(stats):
    return (
        f"Participants demonstrated successful acquisition of reward contingencies, "
        f"with optimal choices increasing from {stats['trial1_mean'] * 100:.1f}% "
        f"(SD = {stats['trial1_sd'] * 100:.1f}%) on the first trial to "
        f"{stats['trial18_mean'] * 100:.1f}% "
        f"(SD = {stats['trial18_sd'] * 100:.1f}%) on the final trial, "
        f"t({stats['df']}) = {stats['t']:.2f}, "
        f"{format_p(stats['p'])}, "
        f"d = {stats['d']:.2f}."
    )


def print_stats(label, stats):
    print("\n" + label)
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


def main():
    props_path = LEARNING_ANALYSIS_DATA_DIR / "individual_proportions_session1.csv"
    groups_path = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"

    if not props_path.exists():
        raise FileNotFoundError(
            f"Missing {props_path}. Run all_individuals_with_curve.py first."
        )

    if not groups_path.exists():
        raise FileNotFoundError(
            f"Missing {groups_path}. Run calculate_learning_rates.py first."
        )

    props = pd.read_csv(props_path, dtype={"subject": str})
    groups = pd.read_csv(groups_path, dtype={"subjectId": str})

    props["subject"] = props["subject"].astype(str).str.zfill(3)
    groups["subjectId"] = groups["subjectId"].astype(str).str.zfill(3)
    groups["plearning_num"] = groups["plearning_num"].astype(str)

    session1_groups = groups[groups["plearning_num"] == "1"].copy()

    good_subjects = (
        session1_groups[session1_groups["learner_group"].astype(str).str.lower() == "good"]
        ["subjectId"]
        .dropna()
        .unique()
        .tolist()
    )

    bad_subjects = (
        session1_groups[session1_groups["learner_group"].astype(str).str.lower() == "bad"]
        ["subjectId"]
        .dropna()
        .unique()
        .tolist()
    )

    all_subjects = sorted(props["subject"].unique().tolist())

    summary_rows = []

    for label, subject_list in [
        ("Good learners", good_subjects),
        ("Bad learners", bad_subjects),
        ("All Session 1 subjects", all_subjects),
    ]:
        group_props = props[props["subject"].isin(subject_list)].copy()

        trial1 = (
            group_props[group_props["trial"] == 1]
            .sort_values("subject")["proportion_correct"]
        )

        trial18 = (
            group_props[group_props["trial"] == 18]
            .sort_values("subject")["proportion_correct"]
        )

        stats = paired_stats(trial1, trial18)
        print_stats(label, stats)

        summary_rows.append({
            "group": label,
            **stats,
            "sentence": manuscript_sentence(stats),
        })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "learning_summary_statistics.txt"

    with open(output_path, "w") as f:
        for row in summary_rows:
            f.write(row["group"] + "\n")
            f.write("-" * len(row["group"]) + "\n")
            f.write(f"n: {row['n']}\n")
            f.write(f"Trial 1 mean: {row['trial1_mean']:.4f}\n")
            f.write(f"Trial 1 SD: {row['trial1_sd']:.4f}\n")
            f.write(f"Trial 18 mean: {row['trial18_mean']:.4f}\n")
            f.write(f"Trial 18 SD: {row['trial18_sd']:.4f}\n")
            f.write(f"df: {row['df']}\n")
            f.write(f"t: {row['t']:.4f}\n")
            f.write(f"p: {row['p']:.6f}\n")
            f.write(f"d: {row['d']:.4f}\n")
            f.write(row["sentence"] + "\n\n")

    print(f"\nSaved summary statistics to: {output_path}")


if __name__ == "__main__":
    main()