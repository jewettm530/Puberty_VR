#!/usr/bin/env python3
"""Compare HR reactivity with changes in linear slope and exponential k."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import (  # noqa: E402
    HEART_RATE_ANALYSIS_DATA_DIR,
    MULTIMODAL_ANALYSIS_DATA_DIR,
    MULTIMODAL_OUTPUTS_DIR,
)

OUT = MULTIMODAL_OUTPUTS_DIR / "heart_rate_learning_correlations"
HR_LABEL = "Mean speech and arithmetic HR − pre-VR HR (bpm)"
OUTCOMES = {
    "delta_slope": ("Change in linear learning slope", "Post − pre slope (percentage points per trial)"),
    "delta_k": ("Change in exponential learning rate", "Post − pre k (per trial)"),
}


def load_measures():
    learning = pd.read_csv(
        MULTIMODAL_ANALYSIS_DATA_DIR / "heart_rate_learning_subject_summary.csv",
        usecols=["subject_id", *OUTCOMES],
        dtype={"subject_id": "string"},
    )
    hr_columns = ["pre_vr_mean_hr", "vr_prep_mean_hr", "vr_speech_mean_hr", "vr_math_mean_hr"]
    hr = pd.read_csv(
        HEART_RATE_ANALYSIS_DATA_DIR / "heart_rate_summary_wide.csv",
        usecols=["subject_id", *hr_columns],
        dtype={"subject_id": "string"},
    )
    for frame in (learning, hr):
        frame["subject_id"] = frame["subject_id"].str.strip().str.zfill(3)
        if frame["subject_id"].isna().any():
            raise ValueError("Input summaries contain a missing subject ID.")

    data = learning.merge(hr, on="subject_id", validate="one_to_one")
    numeric = [*OUTCOMES, *hr_columns]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data[numeric] = data[numeric].replace([np.inf, -np.inf], np.nan)
    data["mean_speech_math_minus_pre_hr"] = data[
        ["vr_speech_mean_hr", "vr_math_mean_hr"]
    ].mean(axis=1, skipna=False) - data["pre_vr_mean_hr"]
    for stage in ("prep", "speech", "math"):
        data[f"{stage}_minus_pre_hr"] = (
            data[f"vr_{stage}_mean_hr"] - data["pre_vr_mean_hr"]
        )
    return data.sort_values("subject_id")


def summarize(pair):
    if len(pair) < 3 or pair.nunique().min() < 2:
        return np.nan, np.nan, np.nan, np.nan
    rho, p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    indices = np.random.default_rng(122).integers(len(pair), size=(3000, len(pair)))
    ranks = stats.rankdata(pair.to_numpy()[indices], axis=1)
    ranks -= ranks.mean(axis=1, keepdims=True)
    numerator = (ranks[:, :, 0] * ranks[:, :, 1]).sum(axis=1)
    denominator = np.linalg.norm(ranks, axis=1).prod(axis=1)
    valid = denominator > 0
    low, high = np.quantile(numerator[valid] / denominator[valid], [.025, .975])
    return rho, low, high, p


def main():
    data = load_measures()
    main_hr = "mean_speech_math_minus_pre_hr"
    # Use the same people when directly comparing the two learning metrics.
    matched = data[[main_hr, *OUTCOMES]].dropna()
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")

    for ax, (outcome, (title, ylabel)) in zip(axes, OUTCOMES.items()):
        pair = matched[[main_hr, outcome]]
        rho, low, high, p = summarize(pair)
        records.append(("poster", main_hr, outcome, len(pair), rho, low, high, p))
        scale = 100 if outcome == "delta_slope" else 1
        ax.scatter(pair[main_hr], scale * pair[outcome], s=41, c="#12766F",
                   edgecolors="white", linewidths=.6, zorder=3)
        for line in (ax.axhline, ax.axvline):
            line(0, color="#80939A", linewidth=.8)
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        subtitle = f"n = {len(pair)}"
        if np.isfinite(rho):
            subtitle += f"   ρ = {rho:+.2f}   95% CI [{low:+.2f}, {high:+.2f}]"
        else:
            subtitle += "   Spearman correlation unavailable"
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=11)
        ax.set_xlabel(HR_LABEL)
        ax.set_ylabel(ylabel)

    # Stage-specific correlations are exploratory and use stage-specific availability.
    for stage in ("prep", "speech", "math"):
        hr = f"{stage}_minus_pre_hr"
        phase_matched = data[[hr, *OUTCOMES]].dropna()
        for outcome in OUTCOMES:
            pair = phase_matched[[hr, outcome]]
            records.append(("exploratory", hr, outcome, len(pair), *summarize(pair)))

    table = pd.DataFrame(
        records, columns=["analysis", "hr_measure", "learning_measure", "n",
                          "spearman_rho", "ci_low", "ci_high", "p_unadjusted"]
    )
    table["p_holm_within_analysis"] = np.nan
    for _, group in table.groupby("analysis"):
        ordered = group["p_unadjusted"].dropna().sort_values()
        factors = np.arange(len(ordered), 0, -1)
        table.loc[ordered.index, "p_holm_within_analysis"] = (
            (ordered * factors).cummax().clip(upper=1)
        )

    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    table_path = OUT / "hr_vs_learning_rate_changes.csv"
    figure_path = figures / "hr_vs_delta_slope_and_delta_k.png"
    table.to_csv(table_path, index=False)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Saved {table_path}\nSaved {figure_path}")


if __name__ == "__main__":
    main()
