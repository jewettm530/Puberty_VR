#!/usr/bin/env python3
"""Relate physiological stress reactivity to pre/post probabilistic learning change.

Primary scientific question
---------------------------
Do participants who show greater heart-rate reactivity during the VR stressor
also show a larger change in learning rate and/or reinforcement strategy
from Session 1 (pre-stressor) to Session 2 (post-stressor)?

Primary statistics
------------------
* Spearman rank correlations (robust to skew/outliers; appropriate for the
  current small exploratory sample).
* Benjamini-Hochberg FDR correction across the primary correlation family.
* Pearson correlations are saved as a secondary/sensitivity analysis.

Learning outcomes
-----------------
* Exponential learning-rate k
* Linear learning slope
* Mean accuracy
* Trial 1 -> Trial 18 accuracy gain

Heart-rate stress indices
-------------------------
* Preparation reactivity = prep mean HR - pre-VR mean HR
* Speech reactivity      = speech mean HR - pre-VR mean HR
* Math reactivity        = math mean HR - pre-VR mean HR
* Mean stress reactivity = mean of available prep/speech/math reactivity
* Maximum stress reactivity = max of available prep/speech/math reactivity

The script also performs secondary correlations with post-stress learning and
recovery measures, but these are kept separate from the primary FDR family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Make shared project paths importable no matter where the script is launched.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_paths import (  # noqa: E402
    HEART_RATE_ANALYSIS_DATA_DIR,
    LEARNING_ANALYSIS_DATA_DIR,
    MULTIMODAL_ANALYSIS_DATA_DIR,
    MULTIMODAL_OUTPUTS_DIR,
    TABLES_DIR,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_N = 6  # Do not test correlations with fewer than this many complete cases.

OUT_DIR = MULTIMODAL_OUTPUTS_DIR / "heart_rate_learning_correlations"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LEARNING_RATE_FILE = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
SLOPE_FILE = LEARNING_ANALYSIS_DATA_DIR / "slope_analysis" / "learning_slopes.csv"
SESSION1_PROP_FILE = LEARNING_ANALYSIS_DATA_DIR / "individual_proportions_session1.csv"
SESSION2_PROP_FILE = LEARNING_ANALYSIS_DATA_DIR / "individual_proportions_session2.csv"
HR_FILE = HEART_RATE_ANALYSIS_DATA_DIR / "heart_rate_summary_wide.csv"
WINSTAY_FILE = TABLES_DIR / "winstay_loseswitch" / "winstay_loseswitch_summary.csv"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def normalize_subject(series: pd.Series) -> pd.Series:
    """Return subject IDs as zero-padded three-character strings."""
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(3)
    )


def bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR-adjusted p-values, preserving missing values."""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return result

    order = np.argsort(valid.to_numpy())
    sorted_p = valid.to_numpy()[order]
    m = len(sorted_p)

    adjusted = sorted_p * m / np.arange(1, m + 1)
    # Enforce monotonicity from largest rank to smallest.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    valid_indices = valid.index.to_numpy()
    result.loc[valid_indices[order]] = adjusted
    return result


def safe_corr(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    """Compute Spearman and Pearson correlations using pairwise complete rows."""
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)

    output: dict[str, float | int] = {
        "n": n,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
    }

    if n < MIN_N or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return output

    rho, sp = stats.spearmanr(pair["x"], pair["y"])
    r, pp = stats.pearsonr(pair["x"], pair["y"])
    output.update(
        {
            "spearman_rho": float(rho),
            "spearman_p": float(sp),
            "pearson_r": float(r),
            "pearson_p": float(pp),
        }
    )
    return output


def correlation_strength(value: float) -> str:
    """Simple magnitude label for an exploratory correlation."""
    if pd.isna(value):
        return "not estimated"
    a = abs(value)
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "moderate"
    return "large"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            "Run the relevant learning-rate/heart-rate summary scripts first."
        )


# ---------------------------------------------------------------------------
# Build subject-level learning table
# ---------------------------------------------------------------------------
def build_learning_table() -> pd.DataFrame:
    for path in [LEARNING_RATE_FILE, SLOPE_FILE, SESSION1_PROP_FILE, SESSION2_PROP_FILE]:
        require_file(path)

    lr = pd.read_csv(LEARNING_RATE_FILE)
    lr["subject_id"] = normalize_subject(lr["subjectId"])
    lr["plearning_num"] = pd.to_numeric(lr["plearning_num"], errors="coerce")

    # One row per subject, with pre (Session 1) and post (Session 2) measures.
    keep = ["subject_id", "plearning_num", "learning_rate_k", "mean_accuracy", "learner_group"]
    lr = lr[keep].copy()

    pre = lr[lr["plearning_num"] == 1].rename(
        columns={
            "learning_rate_k": "k_pre",
            "mean_accuracy": "accuracy_pre",
            "learner_group": "learner_group",
        }
    )[["subject_id", "k_pre", "accuracy_pre", "learner_group"]]

    post = lr[lr["plearning_num"] == 2].rename(
        columns={
            "learning_rate_k": "k_post",
            "mean_accuracy": "accuracy_post",
        }
    )[["subject_id", "k_post", "accuracy_post"]]

    learning = pre.merge(post, on="subject_id", how="outer", validate="one_to_one")
    learning["delta_k"] = learning["k_post"] - learning["k_pre"]
    learning["delta_accuracy"] = learning["accuracy_post"] - learning["accuracy_pre"]

    slopes = pd.read_csv(SLOPE_FILE)
    slopes["subject_id"] = normalize_subject(slopes["Subject"])
    slope_cols = ["subject_id", "slope_pre", "slope_post", "delta_slope"]
    learning = learning.merge(slopes[slope_cols], on="subject_id", how="outer", validate="one_to_one")

    # Trial 1 -> Trial 18 gain gives a simple non-parametric description of
    # acquisition that complements exponential k and the fitted linear slope.
    prop_frames = []
    for path in [SESSION1_PROP_FILE, SESSION2_PROP_FILE]:
        df = pd.read_csv(path)
        df["subject_id"] = normalize_subject(df["subject"])
        df["session"] = pd.to_numeric(df["session"], errors="coerce")
        df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
        df["proportion_correct"] = pd.to_numeric(df["proportion_correct"], errors="coerce")
        prop_frames.append(df)
    prop = pd.concat(prop_frames, ignore_index=True)

    first_last = (
        prop[prop["trial"].isin([1, 18])]
        .pivot_table(
            index=["subject_id", "session"],
            columns="trial",
            values="proportion_correct",
            aggfunc="mean",
        )
        .reset_index()
    )
    first_last.columns.name = None
    first_last = first_last.rename(columns={1: "trial1", 18: "trial18"})
    first_last["trial_gain"] = first_last["trial18"] - first_last["trial1"]

    gain_pre = first_last[first_last["session"] == 1][["subject_id", "trial_gain"]].rename(
        columns={"trial_gain": "trial_gain_pre"}
    )
    gain_post = first_last[first_last["session"] == 2][["subject_id", "trial_gain"]].rename(
        columns={"trial_gain": "trial_gain_post"}
    )
    learning = learning.merge(gain_pre, on="subject_id", how="outer", validate="one_to_one")
    learning = learning.merge(gain_post, on="subject_id", how="outer", validate="one_to_one")
    learning["delta_trial_gain"] = learning["trial_gain_post"] - learning["trial_gain_pre"]

    return learning.sort_values("subject_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build subject-level HR table
# ---------------------------------------------------------------------------
def build_hr_table() -> pd.DataFrame:
    require_file(HR_FILE)
    hr = pd.read_csv(HR_FILE)
    hr["subject_id"] = normalize_subject(hr["subject_id"])

    needed = [
        "subject_id",
        "pre_vr_mean_hr",
        "vr_prep_mean_hr",
        "vr_speech_mean_hr",
        "vr_math_mean_hr",
        "post_vr_mean_hr",
        "recovery_mean_hr",
        "prep_hr_reactivity",
        "speech_hr_reactivity",
        "math_hr_reactivity",
        "post_minus_pre_hr",
        "recovery_minus_post_hr",
    ]
    available = [c for c in needed if c in hr.columns]
    hr = hr[available].copy()

    reactivity_cols = [
        c for c in ["prep_hr_reactivity", "speech_hr_reactivity", "math_hr_reactivity"]
        if c in hr.columns
    ]
    hr["mean_stress_reactivity"] = hr[reactivity_cols].mean(axis=1, skipna=True)
    hr["max_stress_reactivity"] = hr[reactivity_cols].max(axis=1, skipna=True)

    # Avoid creating an apparent value when every stress phase is missing.
    no_reactivity = hr[reactivity_cols].isna().all(axis=1)
    hr.loc[no_reactivity, ["mean_stress_reactivity", "max_stress_reactivity"]] = np.nan

    return hr.sort_values("subject_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build subject-level win-stay / lose-switch strategy table
# ---------------------------------------------------------------------------
def build_strategy_table() -> pd.DataFrame:
    """Return one row per subject with pre/post conditional strategy measures.

    Win-stay and lose-switch are conditional probabilities, not mutually
    exclusive strategy choices.  We therefore analyze them separately.
    A descriptive strategy-bias score (win-stay - lose-switch) is also
    provided, but should not be interpreted as a forced choice between them.
    """
    require_file(WINSTAY_FILE)
    ws = pd.read_csv(WINSTAY_FILE, dtype={"subject": str})
    ws["subject_id"] = normalize_subject(ws["subject"])

    required = [
        "subject_id",
        "overall_winstay_sess1",
        "overall_winstay_sess2",
        "overall_loseswitch_sess1",
        "overall_loseswitch_sess2",
    ]
    missing = [c for c in required if c not in ws.columns]
    if missing:
        raise ValueError(
            f"{WINSTAY_FILE} is missing required columns: {missing}\n"
            "Run scripts/learning_rates/winstay_loseswitch_analysis.py first."
        )

    out = ws[required].copy()
    out = out.rename(
        columns={
            "overall_winstay_sess1": "winstay_pre",
            "overall_winstay_sess2": "winstay_post",
            "overall_loseswitch_sess1": "loseswitch_pre",
            "overall_loseswitch_sess2": "loseswitch_post",
        }
    )

    out["delta_winstay"] = out["winstay_post"] - out["winstay_pre"]
    out["delta_loseswitch"] = out["loseswitch_post"] - out["loseswitch_pre"]

    # Descriptive relative tendency only. Win-stay and lose-switch occur after
    # different feedback types, so this is not a mutually exclusive choice.
    out["strategy_bias_pre"] = out["winstay_pre"] - out["loseswitch_pre"]
    out["strategy_bias_post"] = out["winstay_post"] - out["loseswitch_post"]
    out["delta_strategy_bias"] = out["strategy_bias_post"] - out["strategy_bias_pre"]

    return out.sort_values("subject_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Correlation analyses
# ---------------------------------------------------------------------------
def run_correlation_family(
    data: pd.DataFrame,
    hr_vars: list[str],
    learning_vars: list[str],
    family_name: str,
    apply_fdr: bool = True,
) -> pd.DataFrame:
    rows = []
    for hr_var in hr_vars:
        if hr_var not in data.columns:
            continue
        for learning_var in learning_vars:
            if learning_var not in data.columns:
                continue
            stats_out = safe_corr(data[hr_var], data[learning_var])
            rows.append(
                {
                    "family": family_name,
                    "hr_variable": hr_var,
                    "learning_variable": learning_var,
                    **stats_out,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    if apply_fdr:
        result["spearman_p_fdr"] = bh_fdr(result["spearman_p"])
        result["fdr_significant_0_05"] = result["spearman_p_fdr"] < 0.05
    else:
        result["spearman_p_fdr"] = np.nan
        result["fdr_significant_0_05"] = False

    result["spearman_strength"] = result["spearman_rho"].apply(correlation_strength)
    result["direction"] = np.select(
        [result["spearman_rho"] > 0, result["spearman_rho"] < 0],
        ["positive", "negative"],
        default="none",
    )
    return result.sort_values(
        ["spearman_p_fdr" if apply_fdr else "spearman_p", "spearman_p"],
        na_position="last",
    ).reset_index(drop=True)


def make_scatterplots(data: pd.DataFrame, primary: pd.DataFrame) -> None:
    """Save one scatterplot for every primary HR-learning pair."""
    for _, row in primary.iterrows():
        x_name = row["hr_variable"]
        y_name = row["learning_variable"]
        pair = data[["subject_id", x_name, y_name]].dropna()
        if len(pair) < MIN_N:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.scatter(pair[x_name], pair[y_name], s=55, alpha=0.8)

        # A linear line is only a visual guide; Spearman rho is the primary test.
        if pair[x_name].nunique() > 1:
            slope, intercept = np.polyfit(pair[x_name], pair[y_name], 1)
            x_line = np.linspace(pair[x_name].min(), pair[x_name].max(), 100)
            ax.plot(x_line, intercept + slope * x_line, linewidth=1.5)

        for _, point in pair.iterrows():
            ax.annotate(
                point["subject_id"],
                (point[x_name], point[y_name]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                alpha=0.75,
            )

        rho = row["spearman_rho"]
        p = row["spearman_p"]
        pfdr = row["spearman_p_fdr"]
        ax.set_title(
            f"{x_name} vs {y_name}\n"
            f"Spearman rho={rho:.3f}, p={p:.3f}, FDR p={pfdr:.3f}, n={int(row['n'])}"
        )
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.axhline(0, linewidth=0.8, alpha=0.5)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{x_name}__vs__{y_name}.png", dpi=180)
        plt.close(fig)


def write_report(
    data: pd.DataFrame,
    learning_rate_results: pd.DataFrame,
    strategy_results: pd.DataFrame,
    exploratory_results: pd.DataFrame,
) -> None:
    report_path = OUT_DIR / "heart_rate_learning_correlation_report.txt"

    complete_prepost = data[["k_pre", "k_post"]].dropna().shape[0]
    lr_sig = learning_rate_results[learning_rate_results["fdr_significant_0_05"]]
    st_sig = strategy_results[strategy_results["fdr_significant_0_05"]]

    with report_path.open("w", encoding="utf-8") as f:
        f.write("HEART RATE × LEARNING / STRATEGY ANALYSIS\n")
        f.write("=" * 56 + "\n\n")
        f.write("Primary questions:\n")
        f.write("1) Does greater VR-stressor HR reactivity predict a change in learning rate?\n")
        f.write("2) Does greater VR-stressor HR reactivity predict a change in win-stay or lose-switch behavior?\n\n")
        f.write(f"Subjects in merged dataset: {data['subject_id'].nunique()}\n")
        f.write(f"Subjects with both learning sessions: {complete_prepost}\n\n")

        def section(title: str, frame: pd.DataFrame) -> None:
            f.write(title + "\n")
            f.write("-" * len(title) + "\n")
            estimable = frame.dropna(subset=["spearman_rho"]).sort_values("spearman_p")
            if estimable.empty:
                f.write("No estimable correlations.\n\n")
                return
            for _, row in estimable.iterrows():
                f.write(
                    f"{row['hr_variable']} vs {row['learning_variable']}: "
                    f"rho={row['spearman_rho']:.3f}, n={int(row['n'])}, "
                    f"p={row['spearman_p']:.4f}, FDR p={row['spearman_p_fdr']:.4f} "
                    f"({row['spearman_strength']} {row['direction']})\n"
                )
            f.write("\n")

        section("PRIMARY A: HR REACTIVITY VS LEARNING-RATE CHANGE", learning_rate_results)
        f.write(
            "delta_k and delta_slope are Post - Pre. Positive values mean faster/steeper learning after stress; "
            "negative values mean slower/flatter learning after stress.\n\n"
        )
        section("PRIMARY B: HR REACTIVITY VS STRATEGY CHANGE", strategy_results)
        f.write(
            "delta_winstay and delta_loseswitch are Post - Pre. Positive delta_winstay means a participant became "
            "more likely to stay after a gain; positive delta_loseswitch means they became more likely to switch after a loss.\n"
            "delta_strategy_bias is descriptive only because win-stay and lose-switch are conditioned on different feedback events.\n\n"
        )

        f.write(f"Learning-rate results surviving BH-FDR: {len(lr_sig)}\n")
        f.write(f"Strategy results surviving BH-FDR: {len(st_sig)}\n\n")

        f.write("EXPLORATORY PERFORMANCE / POST-STRESS LEVELS\n")
        f.write("--------------------------------------------\n")
        ex = exploratory_results.dropna(subset=["spearman_rho"]).sort_values("spearman_p").head(20)
        for _, row in ex.iterrows():
            f.write(
                f"{row['hr_variable']} vs {row['learning_variable']}: "
                f"rho={row['spearman_rho']:.3f}, n={int(row['n'])}, p={row['spearman_p']:.4f}\n"
            )

        f.write("\nINTERPRETATION CAUTIONS\n")
        f.write("-----------------------\n")
        f.write("* These are exploratory observational associations; correlation does not show HR caused the behavioral change.\n")
        f.write("* The current sample is small, so effect estimates can be unstable.\n")
        f.write("* Spearman rho is primary because learning-rate k and change scores can be skewed.\n")
        f.write("* BH-FDR is applied separately to the learning-rate and strategy hypothesis families.\n")
        f.write("* Win-stay and lose-switch are conditional behaviors after different feedback types, not mutually exclusive choices.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    learning = build_learning_table()
    strategy = build_strategy_table()
    hr = build_hr_table()

    behavioral = learning.merge(strategy, on="subject_id", how="outer", validate="one_to_one")
    merged = behavioral.merge(hr, on="subject_id", how="outer", validate="one_to_one")

    merged_path = MULTIMODAL_ANALYSIS_DATA_DIR / "heart_rate_learning_subject_summary.csv"
    merged.to_csv(merged_path, index=False)

    primary_hr = [
        "prep_hr_reactivity",
        "speech_hr_reactivity",
        "math_hr_reactivity",
        "mean_stress_reactivity",
        "max_stress_reactivity",
    ]

    # PRIMARY FAMILY A: actual learning-rate measures.
    learning_rate_results = run_correlation_family(
        merged,
        primary_hr,
        ["delta_k", "delta_slope"],
        family_name="primary_hr_vs_learning_rate_change",
        apply_fdr=True,
    )

    # PRIMARY FAMILY B: change in conditional reinforcement strategies.
    strategy_results = run_correlation_family(
        merged,
        primary_hr,
        ["delta_winstay", "delta_loseswitch", "delta_strategy_bias"],
        family_name="primary_hr_vs_strategy_change",
        apply_fdr=True,
    )

    # Exploratory outcomes: broader performance changes and absolute post-stress
    # learning/strategy levels. Kept outside the two primary FDR families.
    exploratory_hr = primary_hr + ["post_minus_pre_hr", "recovery_minus_post_hr"]
    exploratory_outcomes = [
        "delta_accuracy",
        "delta_trial_gain",
        "k_post",
        "slope_post",
        "accuracy_post",
        "trial_gain_post",
        "winstay_post",
        "loseswitch_post",
        "strategy_bias_post",
    ]
    exploratory_results = run_correlation_family(
        merged,
        exploratory_hr,
        exploratory_outcomes,
        family_name="exploratory_hr_vs_performance_and_post_levels",
        apply_fdr=False,
    )

    lr_path = OUT_DIR / "primary_learning_rate_correlations.csv"
    strategy_path = OUT_DIR / "primary_strategy_correlations.csv"
    exploratory_path = OUT_DIR / "exploratory_correlations.csv"
    learning_rate_results.to_csv(lr_path, index=False)
    strategy_results.to_csv(strategy_path, index=False)
    exploratory_results.to_csv(exploratory_path, index=False)

    # Figures for both primary hypothesis families.
    make_scatterplots(merged, learning_rate_results)
    make_scatterplots(merged, strategy_results)
    write_report(merged, learning_rate_results, strategy_results, exploratory_results)

    print("Heart rate × learning/strategy analysis complete.")
    print(f"Merged subject table: {merged_path}")
    print(f"Primary learning-rate correlations: {lr_path}")
    print(f"Primary strategy correlations: {strategy_path}")
    print(f"Exploratory correlations: {exploratory_path}")
    print(f"Report: {OUT_DIR / 'heart_rate_learning_correlation_report.txt'}")
    print(f"Figures: {FIG_DIR}")

    for label, frame in [
        ("learning-rate", learning_rate_results),
        ("strategy", strategy_results),
    ]:
        estimable = frame.dropna(subset=["spearman_rho"])
        if not estimable.empty:
            print(f"\nTop {label} associations by unadjusted p-value:")
            cols = [
                "hr_variable",
                "learning_variable",
                "n",
                "spearman_rho",
                "spearman_p",
                "spearman_p_fdr",
            ]
            print(estimable.sort_values("spearman_p")[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()