#!/usr/bin/env python3
"""
Create updated CPLearn group and individual learning-curve figures.

Professor-requested outputs
---------------------------
Figure A:
    Group-level proportion of choosing the set-winner across trials 1-18,
    plus Trial 1 vs Trial 18 paired t-test and descriptive statistics.

Figure B:
    Individual learning curves across trials 1-18 for every participant.

The script also creates an adolescent-only version (ages 14-19) so that
the full sample and younger-subject sample can be compared directly.

IMPORTANT:
This script uses behavioral learning curves (proportion choosing the
set-winner), NOT computational alpha_plus / alpha_minus parameters.

Recommended unit of analysis:
    Session 1 only, so each participant contributes one learning curve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# Project paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_paths import (  # noqa: E402
    LEARNING_ANALYSIS_DATA_DIR,
    METADATA_DIR,
    MINOR_COMPARISON_OUTPUTS_DIR,
)


# =============================================================================
# Configuration
# =============================================================================

# Existing outputs from the learning-rate pipeline.
PROPORTION_FILE = (
    LEARNING_ANALYSIS_DATA_DIR
    / "individual_proportions_session1.csv"
)

LEARNING_RATE_FILE = (
    LEARNING_ANALYSIS_DATA_DIR
    / "learning_rates.csv"
)

AGE_FILE = (
    METADATA_DIR
    / "rl_subject_ages.csv"
)

# New output folder.
OUT_DIR = (
    MINOR_COMPARISON_OUTPUTS_DIR
    / "individual_learning_curves"
)

FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# Session to use for one curve per participant.
SESSION = 1

# Make subject labels appear at the right edge of Figure B.
# False is cleaner and more similar to your professor's example.
LABEL_INDIVIDUAL_CURVES = False


# =============================================================================
# Utilities
# =============================================================================

def normalize_subject(series: pd.Series) -> pd.Series:
    """Normalize current-study IDs to zero-padded strings such as 005."""
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(3)
    )


def format_p(p: float) -> str:
    """Format p-values for manuscript/slide reporting."""
    if pd.isna(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def quadratic_smooth(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a descriptive quadratic curve.

    This is used only to make the plot visually similar to the supplied
    professor figure. It is NOT the inferential learning-rate model.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if weights is not None:
        weights = np.asarray(weights, dtype=float)[good]

    if len(x) < 3 or len(np.unique(x)) < 3:
        return x, y

    if weights is None:
        coef = np.polyfit(x, y, deg=2)
    else:
        coef = np.polyfit(
            x,
            y,
            deg=2,
            w=np.sqrt(weights),
        )

    x_smooth = np.linspace(1, 18, 250)
    y_smooth = np.polyval(coef, x_smooth)

    # A probability cannot be below 0 or above 1.
    y_smooth = np.clip(y_smooth, 0, 1)

    return x_smooth, y_smooth


# =============================================================================
# Load trial-by-trial proportion data
# =============================================================================

def load_proportion_data() -> pd.DataFrame:
    """Load Session-1 subject × trial set-winner proportions."""

    if not PROPORTION_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{PROPORTION_FILE}\n\n"
            "Rerun the learning-rate/proportion analysis first so newly "
            "added participants are included."
        )

    data = pd.read_csv(PROPORTION_FILE)

    required = {
        "subject",
        "trial",
        "proportion_correct",
    }

    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError(
            f"{PROPORTION_FILE.name} is missing columns: {missing}"
        )

    data["subject_id"] = normalize_subject(data["subject"])
    data["trial"] = pd.to_numeric(
        data["trial"],
        errors="coerce",
    )
    data["proportion_correct"] = pd.to_numeric(
        data["proportion_correct"],
        errors="coerce",
    )

    # Restrict to the 18 experimental trial positions.
    data = data.loc[
        data["trial"].between(1, 18)
    ].copy()

    # If the file contains a session column, make absolutely sure this
    # analysis uses Session 1.
    if "session" in data.columns:
        data["session"] = pd.to_numeric(
            data["session"],
            errors="coerce",
        )

        data = data.loc[
            data["session"] == SESSION
        ].copy()

    # Collapse accidental duplicate subject × trial rows safely.
    data = (
        data.groupby(
            ["subject_id", "trial"],
            as_index=False,
        )
        .agg(
            proportion_correct=(
                "proportion_correct",
                "mean",
            )
        )
    )

    return data


# =============================================================================
# Load ages
# =============================================================================

def load_ages() -> pd.DataFrame:
    """Load participant ages from the same lookup used by the RL script."""

    if not AGE_FILE.exists():
        print(
            f"WARNING: {AGE_FILE} does not exist. "
            "Age-restricted analyses will not be available."
        )

        return pd.DataFrame(
            columns=["subject_id", "age"]
        )

    ages = pd.read_csv(
        AGE_FILE,
        dtype={"subject": str},
    )

    required = {"subject", "age"}

    missing = sorted(required.difference(ages.columns))

    if missing:
        raise ValueError(
            f"{AGE_FILE.name} is missing columns: {missing}"
        )

    ages["subject_id"] = normalize_subject(
        ages["subject"]
    )

    ages["age"] = pd.to_numeric(
        ages["age"],
        errors="coerce",
    )

    ages = (
        ages[
            ["subject_id", "age"]
        ]
        .drop_duplicates("subject_id")
    )

    return ages


# =============================================================================
# Load existing scalar learning-rate k
# =============================================================================

def load_learning_rates() -> pd.DataFrame:
    """
    Load the existing exponential learning-rate k values.

    k is kept in the output table as an additional quantitative measure of
    individual differences, but Figure B itself shows the observed
    trial-by-trial proportions.
    """

    if not LEARNING_RATE_FILE.exists():
        return pd.DataFrame(
            columns=[
                "subject_id",
                "learning_rate_k",
                "mean_accuracy",
                "learner_group",
            ]
        )

    rates = pd.read_csv(LEARNING_RATE_FILE)

    required = {
        "subjectId",
        "plearning_num",
        "learning_rate_k",
    }

    missing = sorted(required.difference(rates.columns))

    if missing:
        print(
            f"WARNING: {LEARNING_RATE_FILE.name} "
            f"is missing {missing}. "
            "Continuing without k."
        )

        return pd.DataFrame(
            columns=[
                "subject_id",
                "learning_rate_k",
                "mean_accuracy",
                "learner_group",
            ]
        )

    rates["subject_id"] = normalize_subject(
        rates["subjectId"]
    )

    rates["plearning_num"] = pd.to_numeric(
        rates["plearning_num"],
        errors="coerce",
    )

    rates = rates.loc[
        rates["plearning_num"] == SESSION
    ].copy()

    keep = [
        "subject_id",
        "learning_rate_k",
    ]

    for optional in [
        "mean_accuracy",
        "learner_group",
    ]:
        if optional in rates.columns:
            keep.append(optional)

    return (
        rates[keep]
        .drop_duplicates("subject_id")
    )


# =============================================================================
# Subject-level summary
# =============================================================================

def make_subject_summary(
    data: pd.DataFrame,
    learning_rates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one row per participant.

    Includes:
        Trial 1 proportion
        Trial 18 proportion
        Trial 18 - Trial 1 gain
        descriptive linear slope across all 18 trials
        existing exponential learning-rate k
    """

    first_last = (
        data.loc[
            data["trial"].isin([1, 18])
        ]
        .pivot(
            index="subject_id",
            columns="trial",
            values="proportion_correct",
        )
        .reset_index()
    )

    first_last.columns.name = None

    first_last = first_last.rename(
        columns={
            1: "trial1",
            18: "trial18",
        }
    )

    if "trial1" not in first_last.columns:
        first_last["trial1"] = np.nan

    if "trial18" not in first_last.columns:
        first_last["trial18"] = np.nan

    first_last["trial_gain"] = (
        first_last["trial18"]
        - first_last["trial1"]
    )

    slope_rows = []

    for subject_id, subject_data in data.groupby(
        "subject_id"
    ):
        subject_data = subject_data.dropna(
            subset=[
                "trial",
                "proportion_correct",
            ]
        )

        if len(subject_data) >= 2:
            slope = np.polyfit(
                subject_data["trial"],
                subject_data["proportion_correct"],
                deg=1,
            )[0]
        else:
            slope = np.nan

        slope_rows.append(
            {
                "subject_id": subject_id,
                "descriptive_linear_slope": slope,
                "n_trial_positions": (
                    subject_data["trial"].nunique()
                ),
            }
        )

    slopes = pd.DataFrame(slope_rows)

    summary = first_last.merge(
        slopes,
        on="subject_id",
        how="outer",
        validate="one_to_one",
    )

    summary = summary.merge(
        learning_rates,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )

    return summary.sort_values(
        "subject_id"
    ).reset_index(drop=True)


# =============================================================================
# Trial 1 vs Trial 18 statistics
# =============================================================================

def first_last_statistics(
    summary: pd.DataFrame,
    sample_name: str,
) -> dict[str, object]:

    paired = summary.dropna(
        subset=["trial1", "trial18"]
    ).copy()

    n = len(paired)

    if n < 2:
        return {
            "sample": sample_name,
            "n": n,
        }

    first = paired["trial1"].to_numpy(
        dtype=float
    )

    last = paired["trial18"].to_numpy(
        dtype=float
    )

    difference = last - first

    # Direction is explicitly Trial 18 - Trial 1.
    t_result = stats.ttest_rel(
        last,
        first,
        nan_policy="omit",
    )

    t_value = float(t_result.statistic)
    p_value = float(t_result.pvalue)

    mean_first = float(np.mean(first))
    sd_first = float(np.std(first, ddof=1))

    mean_last = float(np.mean(last))
    sd_last = float(np.std(last, ddof=1))

    mean_change = float(
        np.mean(difference)
    )

    sd_change = float(
        np.std(difference, ddof=1)
    )

    # Paired-samples Cohen's dz.
    if sd_change > 0:
        cohen_dz = mean_change / sd_change
    else:
        cohen_dz = np.nan

    # 95% CI for the mean paired difference.
    difference_sem = stats.sem(
        difference,
        nan_policy="omit",
    )

    if np.isfinite(difference_sem):
        ci_low, ci_high = stats.t.interval(
            0.95,
            df=n - 1,
            loc=mean_change,
            scale=difference_sem,
        )
    else:
        ci_low = np.nan
        ci_high = np.nan

    print("\n" + "=" * 72)
    print(sample_name.upper())
    print("=" * 72)

    print(f"Subjects with Trial 1 and Trial 18: n = {n}")

    print(
        f"Trial 1:  M = {mean_first:.3f}, "
        f"SD = {sd_first:.3f}"
    )

    print(
        f"Trial 18: M = {mean_last:.3f}, "
        f"SD = {sd_last:.3f}"
    )

    print(
        f"Mean Trial 1 -> Trial 18 increase: "
        f"{mean_change:.3f}"
    )

    print(
        f"Paired t-test: "
        f"t({n - 1}) = {t_value:.3f}, "
        f"{format_p(p_value)}"
    )

    print(
        f"Cohen's dz = {cohen_dz:.3f}"
    )

    print(
        f"95% CI for change = "
        f"[{ci_low:.3f}, {ci_high:.3f}]"
    )

    print(
        "\nProfessor-ready wording:\n"
        f'Among {n} participants, the mean proportion of '
        f'choosing the set-winner increased from '
        f'{mean_first:.2f} on Trial 1 to '
        f'{mean_last:.2f} on Trial 18 '
        f'[paired t({n - 1}) = {t_value:.2f}, '
        f'{format_p(p_value)}, '
        f"Cohen's dz = {cohen_dz:.2f}]."
    )

    return {
        "sample": sample_name,
        "n": n,
        "trial1_mean": mean_first,
        "trial1_sd": sd_first,
        "trial18_mean": mean_last,
        "trial18_sd": sd_last,
        "mean_change": mean_change,
        "t": t_value,
        "df": n - 1,
        "p": p_value,
        "cohen_dz": cohen_dz,
        "change_ci95_low": ci_low,
        "change_ci95_high": ci_high,
    }


# =============================================================================
# Group-level trial summary
# =============================================================================

def make_group_curve(
    data: pd.DataFrame,
) -> pd.DataFrame:

    group = (
        data.groupby(
            "trial",
            as_index=False,
        )
        .agg(
            mean_proportion=(
                "proportion_correct",
                "mean",
            ),
            sd=(
                "proportion_correct",
                "std",
            ),
            n=(
                "subject_id",
                "nunique",
            ),
        )
    )

    group["sem"] = (
        group["sd"]
        / np.sqrt(group["n"])
    )

    # Trial-specific 95% CI across participants.
    group["ci95"] = np.where(
        group["n"] > 1,
        stats.t.ppf(
            0.975,
            group["n"] - 1,
        )
        * group["sem"],
        np.nan,
    )

    group["ci_low"] = np.clip(
        group["mean_proportion"]
        - group["ci95"],
        0,
        1,
    )

    group["ci_high"] = np.clip(
        group["mean_proportion"]
        + group["ci95"],
        0,
        1,
    )

    return group


# =============================================================================
# Figure A
# =============================================================================

def plot_group_curve(
    data: pd.DataFrame,
    sample_name: str,
) -> pd.DataFrame:

    group = make_group_curve(data)

    n_subjects = data[
        "subject_id"
    ].nunique()

    fig, ax = plt.subplots(
        figsize=(7.0, 5.5)
    )

    # Confidence band.
    ax.fill_between(
        group["trial"],
        group["ci_low"],
        group["ci_high"],
        alpha=0.20,
        label="95% CI",
    )

    # Observed group means.
    ax.scatter(
        group["trial"],
        group["mean_proportion"],
        s=35,
        zorder=3,
        label="Observed group mean",
    )

    # Smooth quadratic line, visually similar to the reference figure.
    x_smooth, y_smooth = quadratic_smooth(
        group["trial"].to_numpy(),
        group["mean_proportion"].to_numpy(),
        weights=group["n"].to_numpy(),
    )

    ax.plot(
        x_smooth,
        y_smooth,
        linewidth=2.5,
        label="Smoothed group curve",
    )

    ax.axhline(
        0.50,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(0, 1.05)

    ax.set_xticks(
        [1, 5, 10, 15, 18]
    )

    ax.set_xlabel(
        "Trial Number",
        fontsize=12,
    )

    ax.set_ylabel(
        "Proportion choosing set-winner",
        fontsize=12,
    )

    ax.set_title(
        f"Group Reward Learning Curve\n"
        f"{sample_name.replace('_', ' ')}",
        fontsize=13,
    )

    ax.text(
        0.97,
        0.06,
        f"n = {n_subjects}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    fig.tight_layout()

    fig.savefig(
        FIG_DIR
        / f"figureA_group_curve_{sample_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_DIR
        / f"figureA_group_curve_{sample_name}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    return group


# =============================================================================
# Figure B
# =============================================================================

def plot_individual_curves(
    data: pd.DataFrame,
    sample_name: str,
) -> None:

    subjects = sorted(
        data["subject_id"].dropna().unique()
    )

    n_subjects = len(subjects)

    group = make_group_curve(data)

    fig, ax = plt.subplots(
        figsize=(9.2, 6.2)
    )

    # Put group uncertainty behind the individual curves,
    # similar to the gray background region in the reference.
    ax.fill_between(
        group["trial"],
        group["ci_low"],
        group["ci_high"],
        alpha=0.12,
        zorder=0,
    )

    # One distinct color per participant.
    colors = plt.cm.hsv(
        np.linspace(
            0,
            0.95,
            max(n_subjects, 1),
        )
    )

    for color, subject_id in zip(
        colors,
        subjects,
    ):
        subject_data = (
            data.loc[
                data["subject_id"]
                == subject_id
            ]
            .sort_values("trial")
        )

        x = subject_data[
            "trial"
        ].to_numpy(dtype=float)

        y = subject_data[
            "proportion_correct"
        ].to_numpy(dtype=float)

        # Raw subject × trial proportions.
        ax.scatter(
            x,
            y,
            s=15,
            alpha=0.70,
            color=color,
            zorder=3,
        )

        # Descriptive smooth for that participant.
        x_smooth, y_smooth = quadratic_smooth(
            x,
            y,
        )

        ax.plot(
            x_smooth,
            y_smooth,
            linewidth=1.3,
            alpha=0.75,
            color=color,
            zorder=2,
        )

        if (
            LABEL_INDIVIDUAL_CURVES
            and len(y_smooth) > 0
        ):
            ax.text(
                18.1,
                y_smooth[-1],
                subject_id,
                fontsize=6,
                color=color,
                alpha=0.8,
            )

    ax.axhline(
        0.50,
        linestyle="--",
        linewidth=0.9,
        alpha=0.4,
    )

    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(0, 1.05)

    ax.set_xticks(
        [1, 5, 10, 15, 18]
    )

    ax.set_xlabel(
        "Trial Number",
        fontsize=12,
    )

    ax.set_ylabel(
        "Proportion choosing set-winner",
        fontsize=12,
    )

    ax.set_title(
        "Individual Reward Learning Curves\n"
        f"{sample_name.replace('_', ' ')} "
        f"(n = {n_subjects})",
        fontsize=13,
    )

    # No giant legend: with many participants it obscures the figure.
    fig.tight_layout()

    fig.savefig(
        FIG_DIR
        / f"figureB_individual_curves_{sample_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_DIR
        / f"figureB_individual_curves_{sample_name}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# Run one sample
# =============================================================================

def analyze_sample(
    full_data: pd.DataFrame,
    learning_rates: pd.DataFrame,
    sample_name: str,
    age_min: float | None = None,
    age_max: float | None = None,
) -> dict[str, object]:

    data = full_data.copy()

    if age_min is not None:
        data = data.loc[
            data["age"] >= age_min
        ].copy()

    if age_max is not None:
        data = data.loc[
            data["age"] <= age_max
        ].copy()

    if data.empty:
        print(
            f"\nNo participants available for "
            f"{sample_name}."
        )
        return {
            "sample": sample_name,
            "n": 0,
        }

    n_subjects = data[
        "subject_id"
    ].nunique()

    print(
        f"\nRunning {sample_name}: "
        f"{n_subjects} participants"
    )

    summary = make_subject_summary(
        data,
        learning_rates,
    )

    # Add age to participant summary.
    subject_ages = (
        data[
            ["subject_id", "age"]
        ]
        .drop_duplicates("subject_id")
    )

    summary = summary.merge(
        subject_ages,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )

    stats_row = first_last_statistics(
        summary,
        sample_name,
    )

    group = plot_group_curve(
        data,
        sample_name,
    )

    plot_individual_curves(
        data,
        sample_name,
    )

    # Save the exact data underlying each figure.
    data.to_csv(
        TABLE_DIR
        / f"{sample_name}_individual_trial_proportions.csv",
        index=False,
    )

    summary.to_csv(
        TABLE_DIR
        / f"{sample_name}_subject_learning_summary.csv",
        index=False,
    )

    group.to_csv(
        TABLE_DIR
        / f"{sample_name}_group_trial_summary.csv",
        index=False,
    )

    return stats_row


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    proportions = load_proportion_data()
    ages = load_ages()
    learning_rates = load_learning_rates()

    proportions = proportions.merge(
        ages,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    print("\nParticipants found:")
    print(
        proportions[
            ["subject_id", "age"]
        ]
        .drop_duplicates()
        .sort_values("subject_id")
        .to_string(index=False)
    )

    missing_age_subjects = (
        proportions.loc[
            proportions["age"].isna(),
            "subject_id",
        ]
        .drop_duplicates()
        .tolist()
    )

    if missing_age_subjects:
        print(
            "\nParticipants with missing age "
            "(included in all-subject figures, "
            "excluded from age-filtered figures):"
        )
        print(", ".join(missing_age_subjects))

    stats_rows = []

    # -----------------------------------------------------------------
    # Professor Figure A/B: all available Session-1 participants
    # -----------------------------------------------------------------

    stats_rows.append(
        analyze_sample(
            full_data=proportions,
            learning_rates=learning_rates,
            sample_name="all_subjects",
        )
    )

    # -----------------------------------------------------------------
    # Younger/adolescent version requested by professor: ages 14-19
    # -----------------------------------------------------------------

    stats_rows.append(
        analyze_sample(
            full_data=proportions,
            learning_rates=learning_rates,
            sample_name="age14_19",
            age_min=14,
            age_max=19,
        )
    )

    # -----------------------------------------------------------------
    # Optional: specifically under age 15.
    #
    # Uncomment this if your professor wants a separate figure containing
    # only participants younger than 15.
    # -----------------------------------------------------------------

    # stats_rows.append(
    #     analyze_sample(
    #         full_data=proportions,
    #         learning_rates=learning_rates,
    #         sample_name="under15",
    #         age_max=14.999,
    #     )
    # )

    pd.DataFrame(stats_rows).to_csv(
        TABLE_DIR
        / "first_vs_last_trial_statistics.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)

    print(
        f"\nFigures saved to:\n{FIG_DIR}"
    )

    print(
        f"\nTables saved to:\n{TABLE_DIR}"
    )


if __name__ == "__main__":
    main()