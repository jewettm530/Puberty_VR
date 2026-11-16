"""Generate the HR–theta and HR–learning figures for the VR stress poster."""

import numpy as np
import pandas as pd
from scipy import stats

# This module configures the non-interactive Matplotlib backend.
from multimodal_associations import MIN_N, OUT, build, sid
from learning_rates.reward_value_alpha import prepare_behavior_data
from project_paths import EEG_OUTPUTS_DIR, PLEARNING_DIR

import matplotlib.pyplot as plt


def frontal_theta_change():
    """Calculate post-minus-pre log10 theta from paired AF7/AF8 windows."""
    audit = EEG_OUTPUTS_DIR / "audit" / "eeg_window_qc_and_power.csv"
    keys = ["subject_id", "phase", "recording", "window_index"]
    columns = ["subject", *keys[1:], "channel", "theta", "paired_pass"]
    windows = pd.read_csv(audit, usecols=columns)
    windows["subject_id"] = sid(windows["subject"])
    windows = windows.loc[
        windows["phase"].isin(["pre_vr", "post_vr"])
        & windows["channel"].isin(["AF7", "AF8"])
    ].dropna(subset=keys).copy()

    recordings = windows[keys[:3]].drop_duplicates()
    if recordings.duplicated(keys[:2]).any():
        raise ValueError("More than one EEG recording for a subject and phase.")

    total = windows.groupby(keys[:2])["window_index"].nunique()
    windows["theta"] = pd.to_numeric(windows["theta"], errors="coerce")
    accepted = windows.loc[
        windows["paired_pass"].astype(str).str.lower().eq("true")
        & np.isfinite(windows["theta"])
        & windows["theta"].gt(0)
    ]
    paired = accepted.pivot(index=keys, columns="channel", values="theta")
    paired = paired.reindex(columns=["AF7", "AF8"]).dropna()
    by_phase = paired.mean(axis=1).groupby(level=keys[:2])
    count = by_phase.size().reindex(total.index, fill_value=0)
    eligible = count.ge(5) & count.div(total).ge(0.50)
    phase_power = np.log10(by_phase.mean().reindex(total.index).where(eligible))
    phase_power = phase_power.unstack().reindex(columns=["pre_vr", "post_vr"])
    return (phase_power["post_vr"] - phase_power["pre_vr"]).rename("theta_change")


def learning_gain_change():
    """Compare late-minus-early optimal-choice gain across sessions."""
    rows = []
    for path in sorted(PLEARNING_DIR.glob("*_plearning_*.csv")):
        subject, _, session = path.stem.partition("_plearning_")
        if not subject.isdecimal() or session not in {"1", "2"}:
            continue
        trials = prepare_behavior_data(path)
        optimal = trials["chosen"].eq(trials["winner"]).where(
            trials["winner"].isin(["face", "house"])
        )
        early = optimal[trials["trialNumber"].between(1, 6)].mean()
        late = optimal[trials["trialNumber"].between(13, 18)].mean()
        rows.append((subject.zfill(3), session, 100 * (late - early)))

    gains = pd.DataFrame(rows, columns=["subject_id", "session", "gain"])
    gains = gains.pivot(index="subject_id", columns="session", values="gain")
    gains = gains.reindex(columns=["1", "2"])
    return (gains["2"] - gains["1"]).rename("gain_change")


def plot_panel(ax, hr, outcome, title, ylabel):
    pair = pd.concat([hr.rename("hr"), outcome.rename("outcome")], axis=1)
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    label = f"n = {len(pair)}"
    if len(pair) >= MIN_N and pair.nunique().min() > 1:
        rho = stats.spearmanr(pair["hr"], pair["outcome"]).statistic
        draws = np.random.default_rng(122).integers(
            len(pair), size=(3000, len(pair))
        )
        ranks = stats.rankdata(pair.to_numpy()[draws], axis=1)
        ranks -= ranks.mean(axis=1, keepdims=True)
        numerator = (ranks[:, :, 0] * ranks[:, :, 1]).sum(axis=1)
        denominator = np.linalg.norm(ranks, axis=1).prod(axis=1)
        valid = denominator > 0
        label += f"   ρ = {rho:+.2f}"
        if valid.any():
            low, high = np.quantile(
                numerator[valid] / denominator[valid], [0.025, 0.975]
            )
            label += f"   95% CI [{low:+.2f}, {high:+.2f}]"
    else:
        label += "   Insufficient data to estimate ρ"

    ax.scatter(pair["hr"], pair["outcome"], color="#167A78", s=42)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title(f"{title}\n{label}", loc="left", fontsize=11)
    ax.set_xlabel("Mean speech and arithmetic HR − pre-VR HR (bpm)")
    ax.set_ylabel(ylabel)


def main():
    data = build().set_index("subject_id")
    hr = data[["vr_speech_mean_hr", "vr_math_mean_hr"]].mean(
        axis=1, skipna=False
    ) - data["pre_vr_mean_hr"]
    theta = frontal_theta_change()
    gain = learning_gain_change()
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), layout="constrained")
    plot_panel(
        axes[0], hr, theta.abs(), "Magnitude of frontal theta change",
        "|Post − pre theta| (log₁₀ power)",
    )
    plot_panel(
        axes[1], hr, theta, "Direction of frontal theta change",
        "Post − pre theta (log₁₀ power)",
    )
    fig.savefig(folder / "hr_reactivity_vs_frontal_theta.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.3), layout="constrained")
    plot_panel(
        ax, hr, gain, "Change in optimal-choice learning gain",
        "Post − pre learning gain (percentage points)",
    )
    fig.savefig(folder / "hr_reactivity_vs_learning_gain.png", dpi=300)
    plt.close(fig)
    print(f"Saved both poster figures to {folder}")


if __name__ == "__main__":
    main()