#!/usr/bin/env python3
"""Create a focused table, findings report, and figures for the age-group analysis."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_paths import (  # noqa: E402
    MINOR_COMPARISON_ANALYSIS_DATA_DIR,
    MINOR_COMPARISON_OUTPUTS_DIR,
)


JOINED_PATH = MINOR_COMPARISON_ANALYSIS_DATA_DIR / "minor_vs_older_joined.csv"
COMPARISONS_PATH = MINOR_COMPARISON_OUTPUTS_DIR / "minor_vs_older_numeric_comparisons.csv"
KEY_RESULTS_PATH = MINOR_COMPARISON_OUTPUTS_DIR / "minor_vs_older_key_results.csv"
STRATEGY_COUNTS_PATH = MINOR_COMPARISON_OUTPUTS_DIR / "minor_vs_older_strategy_use_counts.csv"
FINDINGS_PATH = MINOR_COMPARISON_OUTPUTS_DIR / "minor_vs_older_major_findings.txt"
FIGURES_DIR = MINOR_COMPARISON_OUTPUTS_DIR / "figures"

GROUP_ORDER = ["minor", "older_than_minor"]
GROUP_LABELS = {"minor": "Minors", "older_than_minor": "Older"}
GROUP_COLORS = {"minor": "#D95F59", "older_than_minor": "#3B6EA8"}
STRATEGY_THRESHOLD = 0.50

KEY_OUTCOMES = [
    ("Learning", "Learning rate k", "Session 1", "learning_rate_k", "session_1", "rate parameter"),
    ("Learning", "Learning rate k", "Session 2", "learning_rate_k", "session_2", "rate parameter"),
    ("Learning", "Mean accuracy", "Session 1", "mean_accuracy", "session_1", "proportion correct"),
    ("Learning", "Mean accuracy", "Session 2", "mean_accuracy", "session_2", "proportion correct"),
    ("Learning", "Area under learning curve", "Session 1", "auc_sess1", "subject_level", "AUC"),
    ("Learning", "Area under learning curve", "Session 2", "auc_sess2", "subject_level", "AUC"),
    ("Learning", "Median trial accuracy", "Session 1", "median_sess1", "subject_level", "proportion correct"),
    ("Learning", "Median trial accuracy", "Session 2", "median_sess2", "subject_level", "proportion correct"),
    ("Learning", "Linear learning slope", "Session 1", "slope_pre", "subject_level", "proportion per trial"),
    ("Learning", "Linear learning slope", "Session 2", "slope_post", "subject_level", "proportion per trial"),
    ("Learning", "Change in linear slope", "Session 2 minus Session 1", "delta_slope", "subject_level", "proportion per trial"),
    ("Strategy", "Win-stay probability", "Session 1", "overall_winstay_sess1", "subject_level", "conditional probability"),
    ("Strategy", "Lose-switch probability", "Session 1", "overall_loseswitch_sess1", "subject_level", "conditional probability"),
    ("Strategy", "Win-stay probability", "Session 2", "overall_winstay_sess2", "subject_level", "conditional probability"),
    ("Strategy", "Lose-switch probability", "Session 2", "overall_loseswitch_sess2", "subject_level", "conditional probability"),
    ("Heart rate", "Baseline mean heart rate", "Subject level", "baseline_mean_hr", "subject_level", "bpm"),
    ("Heart rate", "Math heart-rate reactivity", "Subject level", "math_hr_reactivity", "subject_level", "bpm change from pre-VR"),
    ("Heart rate", "Speech heart-rate reactivity", "Subject level", "speech_hr_reactivity", "subject_level", "bpm change from pre-VR"),
    ("Heart rate", "Preparation heart-rate reactivity", "Subject level", "prep_hr_reactivity", "subject_level", "bpm change from pre-VR"),
    ("Heart rate", "Post minus pre-VR heart rate", "Subject level", "post_minus_pre_hr", "subject_level", "bpm difference"),
]


def first_nonmissing(series: pd.Series) -> object:
    values = series.dropna()
    return values.iloc[0] if not values.empty else np.nan


def subject_level_data(joined: pd.DataFrame) -> pd.DataFrame:
    columns = [item[3] for item in KEY_OUTCOMES if item[4] == "subject_level"]
    return joined.sort_values("session").groupby("subject_id", as_index=False).agg(
        {**{column: first_nonmissing for column in sorted(set(columns))}, "age_group": "first"}
    )


def build_key_results(joined: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    subject_data = subject_level_data(joined)
    rows: list[dict[str, object]] = []
    for domain, outcome, session, variable, scope, units in KEY_OUTCOMES:
        match = comparisons.loc[
            comparisons["variable"].eq(variable) & comparisons["scope"].eq(scope)
        ]
        if len(match) != 1:
            raise ValueError(f"Expected one comparison for {scope}/{variable}; found {len(match)}")
        source = match.iloc[0]
        row = {
            "domain": domain,
            "outcome": outcome,
            "session": session,
            "source_variable": variable,
            "units": units,
            "n_minor": int(source["n_minor"]),
            "n_older": int(source["n_older"]),
            "minor_mean": source["minor_mean"],
            "minor_sd": source["minor_sd"],
            "older_mean": source["older_mean"],
            "older_sd": source["older_sd"],
            "minor_minus_older": source["minor_minus_older"],
            "hedges_g": source["effect_size"],
            "p_value": source["p_value"],
            "p_value_fdr_bh": source["p_value_fdr_bh"],
            "fdr_significant_0_05": bool(source["significant_fdr_0_05"]),
            "minor_at_or_above_0_50_n": np.nan,
            "minor_at_or_above_0_50_pct": np.nan,
            "older_at_or_above_0_50_n": np.nan,
            "older_at_or_above_0_50_pct": np.nan,
        }
        if domain == "Strategy":
            for group, prefix in [("minor", "minor"), ("older_than_minor", "older")]:
                values = pd.to_numeric(
                    subject_data.loc[subject_data["age_group"].eq(group), variable],
                    errors="coerce",
                ).dropna()
                count = int(values.ge(STRATEGY_THRESHOLD).sum())
                row[f"{prefix}_at_or_above_0_50_n"] = count
                row[f"{prefix}_at_or_above_0_50_pct"] = 100 * count / len(values) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_strategy_counts(key_results: pd.DataFrame) -> pd.DataFrame:
    strategy = key_results.loc[key_results["domain"].eq("Strategy")].copy()
    rows = []
    for _, row in strategy.iterrows():
        for group, prefix in [("Minors", "minor"), ("Older", "older")]:
            rows.append({
                "strategy": row["outcome"],
                "session": row["session"],
                "group": group,
                "n_with_valid_score": int(row[f"n_{prefix}"]),
                "mean_conditional_probability": row[f"{prefix}_mean"],
                "sd_conditional_probability": row[f"{prefix}_sd"],
                "n_at_or_above_0_50": int(row[f"{prefix}_at_or_above_0_50_n"]),
                "percent_at_or_above_0_50": row[f"{prefix}_at_or_above_0_50_pct"],
            })
    return pd.DataFrame(rows)


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def p_text(value: float) -> str:
    if pd.isna(value):
        return "p = NA"
    return "p < .001" if value < 0.001 else f"p = {value:.3f}".replace("0.", ".")


def result_row(key_results: pd.DataFrame, variable: str, session: str) -> pd.Series:
    match = key_results.loc[
        key_results["source_variable"].eq(variable) & key_results["session"].eq(session)
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one key result for {variable}/{session}")
    return match.iloc[0]


def write_findings(key_results: pd.DataFrame, strategy_counts: pd.DataFrame, path: Path) -> None:
    lr1 = result_row(key_results, "learning_rate_k", "Session 1")
    lr2 = result_row(key_results, "learning_rate_k", "Session 2")
    acc1 = result_row(key_results, "mean_accuracy", "Session 1")
    acc2 = result_row(key_results, "mean_accuracy", "Session 2")
    slope1 = result_row(key_results, "slope_pre", "Session 1")
    slope2 = result_row(key_results, "slope_post", "Session 2")

    lines = [
        "MAJOR FINDINGS: MINORS COMPARED WITH OLDER SUBJECTS",
        "==================================================",
        "",
        "Sample",
        "------",
        "The comparison includes 6 minor subjects and 12 older subjects. Session 2",
        "has 5 minors and 11 older subjects because subjects 027 and 014 are missing",
        "Session 2, respectively. Results are exploratory because the groups are small.",
        "",
        "Learning performance",
        "--------------------",
        (
            f"Session 1 average learning-rate k was {fmt(lr1.minor_mean)} for minors and "
            f"{fmt(lr1.older_mean)} for older subjects (minor-minus-older difference "
            f"{fmt(lr1.minor_minus_older)}, Hedges' g = {fmt(lr1.hedges_g)}, "
            f"{p_text(lr1.p_value)})."
        ),
        (
            f"Session 2 average learning-rate k was {fmt(lr2.minor_mean)} for minors and "
            f"{fmt(lr2.older_mean)} for older subjects (difference "
            f"{fmt(lr2.minor_minus_older)}, Hedges' g = {fmt(lr2.hedges_g)}, "
            f"{p_text(lr2.p_value)}). Lower k indicates a slower fitted approach to the "
            "learning asymptote."
        ),
        (
            f"Despite lower fitted k values, minors had higher mean accuracy: Session 1 "
            f"{fmt(acc1.minor_mean)} versus {fmt(acc1.older_mean)} (difference "
            f"{fmt(acc1.minor_minus_older)}, {p_text(acc1.p_value)}), and Session 2 "
            f"{fmt(acc2.minor_mean)} versus {fmt(acc2.older_mean)} (difference "
            f"{fmt(acc2.minor_minus_older)}, {p_text(acc2.p_value)}). This is possible "
            "because fitted learning speed and overall accuracy measure different features."
        ),
        (
            f"Average linear slope was {fmt(slope1.minor_mean, 4)} for minors versus "
            f"{fmt(slope1.older_mean, 4)} for older subjects in Session 1, and "
            f"{fmt(slope2.minor_mean, 4)} versus {fmt(slope2.older_mean, 4)} in Session 2."
        ),
        "",
        "Win-stay and lose-switch strategy use",
        "-------------------------------------",
        (
            "Strategy-use counts below define use as a conditional probability of at "
            "least 0.50; the mean probabilities are also reported so the threshold does "
            "not hide variation."
        ),
    ]
    for _, row in strategy_counts.iterrows():
        lines.append(
            f"{row['session']} {row['strategy'].lower()}, {row['group'].lower()}: mean "
            f"{fmt(row['mean_conditional_probability'])}; "
            f"{int(row['n_at_or_above_0_50'])}/{int(row['n_with_valid_score'])} "
            f"({row['percent_at_or_above_0_50']:.1f}%) at or above 0.50."
        )

    lines.extend([
        "",
        "Heart-rate reactivity",
        "---------------------",
    ])
    for variable, label in [
        ("math_hr_reactivity", "Math reactivity"),
        ("speech_hr_reactivity", "Speech reactivity"),
        ("prep_hr_reactivity", "Preparation reactivity"),
    ]:
        row = result_row(key_results, variable, "Subject level")
        lines.append(
            f"{label}: minors {fmt(row.minor_mean)} bpm, older subjects "
            f"{fmt(row.older_mean)} bpm; difference {fmt(row.minor_minus_older)} bpm, "
            f"{p_text(row.p_value)}."
        )

    smallest = key_results.sort_values("p_value").iloc[0]
    lines.extend([
        "",
        "Statistical conclusion",
        "----------------------",
        "None of the full set of 160 outcome comparisons was significant after",
        "Benjamini-Hochberg false-discovery-rate correction. Among the focused key",
        f"results, the smallest unadjusted p-value was for {smallest.outcome.lower()} "
        f"({smallest.session}; {p_text(smallest.p_value)}, FDR-adjusted "
        f"p = {fmt(smallest.p_value_fdr_bh)}). These data show descriptive group",
        "differences, not reliable evidence of age-group effects. Larger groups are",
        "needed for stable effect estimates and confirmatory inference.",
        "",
        "Reading the figures",
        "-------------------",
        "Bars show group means, error bars show standard errors, and dots show individual",
        "subjects. The standardized-effect figure shows Hedges' g as minor minus older;",
        "positive values favor higher values among minors. It intentionally does not show",
        "confidence intervals, so it should be used as a descriptive overview only.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_mean_sem_points(ax: plt.Axes, values: dict[tuple[int, str], np.ndarray], sessions: list[str], ylabel: str, title: str, reference: float | None = None) -> None:
    width = 0.34
    offsets = {"minor": -width / 2, "older_than_minor": width / 2}
    for group in GROUP_ORDER:
        means = []
        sems = []
        positions = []
        for index, _ in enumerate(sessions):
            array = values[(index, group)]
            means.append(np.mean(array))
            sems.append(np.std(array, ddof=1) / np.sqrt(len(array)) if len(array) > 1 else 0)
            position = index + offsets[group]
            positions.append(position)
            jitter = np.linspace(-0.055, 0.055, len(array)) if len(array) > 1 else np.array([0.0])
            ax.scatter(
                position + jitter,
                array,
                s=28,
                color=GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                zorder=3,
            )
        ax.bar(
            positions,
            means,
            width=width * 0.82,
            color=GROUP_COLORS[group],
            alpha=0.30,
            edgecolor=GROUP_COLORS[group],
            linewidth=1.3,
            label=GROUP_LABELS[group],
            zorder=1,
        )
        ax.errorbar(positions, means, yerr=sems, fmt="none", color="#222222", capsize=4, linewidth=1.2, zorder=4)
    if reference is not None:
        ax.axhline(reference, color="#777777", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xticks(range(len(sessions)), sessions)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)


def session_values(joined: pd.DataFrame, variable: str) -> dict[tuple[int, str], np.ndarray]:
    values = {}
    for session_index, session in enumerate([1, 2]):
        for group in GROUP_ORDER:
            values[(session_index, group)] = pd.to_numeric(
                joined.loc[joined["session"].eq(session) & joined["age_group"].eq(group), variable],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
    return values


def subject_session_values(subject_data: pd.DataFrame, variables: list[str]) -> dict[tuple[int, str], np.ndarray]:
    values = {}
    for index, variable in enumerate(variables):
        for group in GROUP_ORDER:
            values[(index, group)] = pd.to_numeric(
                subject_data.loc[subject_data["age_group"].eq(group), variable], errors="coerce"
            ).dropna().to_numpy(dtype=float)
    return values


def save_figures(joined: pd.DataFrame, key_results: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    subject_data = subject_level_data(joined)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    add_mean_sem_points(axes[0], session_values(joined, "learning_rate_k"), ["Session 1", "Session 2"], "Fitted learning-rate k", "Learning speed")
    add_mean_sem_points(axes[1], session_values(joined, "mean_accuracy"), ["Session 1", "Session 2"], "Mean proportion correct", "Overall accuracy", reference=0.5)
    add_mean_sem_points(axes[2], subject_session_values(subject_data, ["auc_sess1", "auc_sess2"]), ["Session 1", "Session 2"], "Area under learning curve", "Learning-curve AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.suptitle("Learning outcomes: minors compared with older subjects", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(FIGURES_DIR / "learning_outcomes_by_age_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    add_mean_sem_points(axes[0], subject_session_values(subject_data, ["overall_winstay_sess1", "overall_winstay_sess2"]), ["Session 1", "Session 2"], "P(stay | previous gain)", "Win-stay", reference=0.5)
    add_mean_sem_points(axes[1], subject_session_values(subject_data, ["overall_loseswitch_sess1", "overall_loseswitch_sess2"]), ["Session 1", "Session 2"], "P(switch | previous loss)", "Lose-switch", reference=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.suptitle("Feedback-based strategy use by age group", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(FIGURES_DIR / "strategy_use_by_age_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    add_mean_sem_points(
        ax,
        subject_session_values(subject_data, ["prep_hr_reactivity", "math_hr_reactivity", "speech_hr_reactivity"]),
        ["Preparation", "Math", "Speech"],
        "Heart-rate change from pre-VR (bpm)",
        "Heart-rate reactivity",
        reference=0,
    )
    ax.legend(frameon=False)
    fig.savefig(FIGURES_DIR / "heart_rate_reactivity_by_age_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    effect_rows = key_results.loc[
        key_results["source_variable"].isin([
            "learning_rate_k", "mean_accuracy", "auc_sess1", "auc_sess2",
            "slope_pre", "slope_post", "overall_winstay_sess1",
            "overall_winstay_sess2", "overall_loseswitch_sess1",
            "overall_loseswitch_sess2", "math_hr_reactivity",
            "speech_hr_reactivity", "prep_hr_reactivity",
        ])
    ].copy()
    effect_rows["label"] = effect_rows["outcome"] + " — " + effect_rows["session"]
    effect_rows = effect_rows.sort_values("hedges_g")
    fig_height = max(5.5, 0.45 * len(effect_rows))
    fig, ax = plt.subplots(figsize=(10, fig_height), constrained_layout=True)
    y = np.arange(len(effect_rows))
    ax.scatter(effect_rows["hedges_g"], y, s=65, color="#5B4B8A", zorder=3)
    ax.axvline(0, color="#555555", linewidth=1.2)
    ax.set_yticks(y, effect_rows["label"])
    ax.set_xlabel("Hedges' g (minor minus older)")
    ax.set_title("Exploratory standardized group differences\n(points only; no confidence intervals)", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    for yi, (_, row) in enumerate(effect_rows.iterrows()):
        ax.annotate(f"p={row.p_value:.3f}", (row.hedges_g, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    fig.savefig(FIGURES_DIR / "key_outcome_effect_sizes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not JOINED_PATH.exists() or not COMPARISONS_PATH.exists():
        raise FileNotFoundError(
            "Missing minor-comparison inputs. Run compare_minors_to_older.py first."
        )
    joined = pd.read_csv(JOINED_PATH, dtype={"subject_id": str})
    comparisons = pd.read_csv(COMPARISONS_PATH)
    key_results = build_key_results(joined, comparisons)
    strategy_counts = build_strategy_counts(key_results)

    MINOR_COMPARISON_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    key_results.to_csv(KEY_RESULTS_PATH, index=False)
    strategy_counts.to_csv(STRATEGY_COUNTS_PATH, index=False)
    write_findings(key_results, strategy_counts, FINDINGS_PATH)
    save_figures(joined, key_results)

    print(f"Saved table: {KEY_RESULTS_PATH}")
    print(f"Saved strategy counts: {STRATEGY_COUNTS_PATH}")
    print(f"Saved report: {FINDINGS_PATH}")
    print(f"Saved figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
