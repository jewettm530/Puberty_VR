#!/usr/bin/env python3
"""Subject-level HR / frontal EEG / learning association analyses.

EEG phases: baseline, anticipation=preVR, acute_stress_response=postVR,
recovery=recovery. No EEG is recorded during VR prep, speech, or math.
Plearn 1 precedes preVR; Plearn 2 follows postVR.


Run after the HR, EEG, learning, and existing HR-learning summary pipelines:
    python3 scripts/multimodal/multimodal_associations.py

All analyses are exploratory and observational. Each subject contributes one row.
Missing measurements remain missing; no zero-filling or imputation is performed.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import (  # noqa: E402
    EEG_ANALYSIS_DATA_DIR, HEART_RATE_ANALYSIS_DATA_DIR,
    MULTIMODAL_ANALYSIS_DATA_DIR, MULTIMODAL_OUTPUTS_DIR,
)

OUT = MULTIMODAL_OUTPUTS_DIR / "associations"
DATA_FILE = MULTIMODAL_ANALYSIS_DATA_DIR / "multimodal_associations_dataset.csv"
MIN_N = 6
MIN_MODEL_N = 12
BANDS = ("theta", "alpha", "beta")
SENSORS = ("frontal", "af7", "af8")


def sid(series):
    return series.astype("string").str.extract(r"(\d+)", expand=False).str.zfill(3)


def read_unique(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    df = pd.read_csv(path)
    if "subject_id" not in df:
        raise ValueError(f"Missing subject_id in {path}")
    df["subject_id"] = sid(df["subject_id"])
    df = df.dropna(subset=["subject_id"])
    if df.subject_id.duplicated().any():
        raise ValueError(f"Duplicate subjects in {path}")
    return df


def build():
    hrlearn = read_unique(MULTIMODAL_ANALYSIS_DATA_DIR / "heart_rate_learning_subject_summary.csv")
    eeg = read_unique(EEG_ANALYSIS_DATA_DIR / "eeg_frontal_log_bandpower_wide.csv")
    hr = read_unique(HEART_RATE_ANALYSIS_DATA_DIR / "heart_rate_summary_wide.csv")
    # Prefer freshly rebuilt HR values over stale copies in the HR-learning table.
    hr_fields = [c for c in hr.columns if c != "subject_id"]
    stale = [c for c in hr_fields if c in hrlearn.columns]
    merged = hrlearn.drop(columns=stale).merge(hr, on="subject_id", how="outer", validate="one_to_one")
    merged = merged.merge(eeg, on="subject_id", how="outer", validate="one_to_one", indicator="eeg_join")
    merged["has_hr"] = merged.get("pre_vr_mean_hr", pd.Series(np.nan, index=merged.index)).notna()
    merged["has_learning_change"] = merged["delta_k"].notna()
    merged["has_eeg"] = merged["eeg_join"].ne("left_only")
    merged = merged.drop(columns="eeg_join")
    # EEG frontal mean: average of valid AF7/AF8 channels, retaining single-channel
    # values only in channel-specific columns; frontal requires both channels.
    for phase in ("baseline", "anticipation", "acute_stress_response", "recovery"):
        for band in BANDS:
            for channel in ("af7", "af8"):
                key = f"{phase}_{channel}_{band}_log10_power"
                if key not in merged:
                    merged[key] = np.nan
                merged[key] = pd.to_numeric(merged[key], errors="coerce")
            keys = [f"{phase}_{ch}_{band}_log10_power" for ch in ("af7", "af8")]
            merged[f"{phase}_frontal_{band}"] = merged[keys].mean(axis=1, skipna=False)
        for ch in SENSORS:
            def power(band):
                return merged[f"{phase}_{ch}_{band}_log10_power"] if ch != "frontal" else merged[f"{phase}_frontal_{band}"]
            # Difference of log powers = log10(beta/alpha); no exponentiation.
            merged[f"{phase}_{ch}_log_beta_alpha_ratio"] = power("beta") - power("alpha")
    # These source labels refer to measurement order, not VR-task EEG:
    # anticipation = preVR; acute_stress_response = postVR.
    for ch in SENSORS:
        for measure in (*BANDS, "log_beta_alpha_ratio"):
            def v(phase):
                if measure in BANDS:
                    return (merged[f"{phase}_{ch}_{measure}_log10_power"] if ch != "frontal"
                            else merged[f"{phase}_frontal_{measure}"])
                return merged[f"{phase}_{ch}_{measure}"]
            merged[f"eeg_pre_minus_baseline_{ch}_{measure}"] = v("anticipation") - v("baseline")
            merged[f"eeg_post_minus_pre_{ch}_{measure}"] = v("acute_stress_response") - v("anticipation")
            merged[f"eeg_recovery_minus_post_{ch}_{measure}"] = v("recovery") - v("acute_stress_response")
    for c in ("pre_vr_mean_hr", "vr_prep_mean_hr", "vr_speech_mean_hr", "vr_math_mean_hr", "post_vr_mean_hr", "recovery_mean_hr"):
        if c not in merged:
            merged[c] = np.nan
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged["hr_prep_vs_pre"] = merged.vr_prep_mean_hr - merged.pre_vr_mean_hr
    merged["hr_speech_vs_pre"] = merged.vr_speech_mean_hr - merged.pre_vr_mean_hr
    merged["hr_math_vs_pre"] = merged.vr_math_mean_hr - merged.pre_vr_mean_hr
    tasks = merged[["vr_prep_mean_hr", "vr_speech_mean_hr", "vr_math_mean_hr"]]
    # Require all three stages for a comparable combined VR index.
    merged["hr_mean_vr_vs_pre"] = tasks.mean(axis=1, skipna=False) - merged.pre_vr_mean_hr
    merged["hr_max_vr_vs_pre"] = tasks.max(axis=1, skipna=False) - merged.pre_vr_mean_hr
    merged["hr_post_minus_pre"] = merged.post_vr_mean_hr - merged.pre_vr_mean_hr
    merged["hr_recovery_minus_post"] = merged.recovery_mean_hr - merged.post_vr_mean_hr
    merged["hr_post_minus_mean_vr"] = merged.post_vr_mean_hr - tasks.mean(axis=1, skipna=False)
    return merged.sort_values("subject_id").reset_index(drop=True)


def bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    result = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return result
    order = ok[np.argsort(p[ok])]
    adj = np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1]
    result[order] = np.clip(adj, 0, 1)
    return result


def correlate(df, x, y, family, priority, records):
    if x not in df or y not in df:
        return
    pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    row = dict(family=family, priority=priority, predictor=x, outcome=y, n=n,
               spearman_rho=np.nan, spearman_p=np.nan, pearson_r=np.nan, pearson_p=np.nan)
    if n >= MIN_N and pair[x].nunique() > 1 and pair[y].nunique() > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, p = stats.spearmanr(pair[x], pair[y]); rp, pp = stats.pearsonr(pair[x], pair[y])
        row.update(spearman_rho=float(r), spearman_p=float(p), pearson_r=float(rp), pearson_p=float(pp))
    records.append(row)


def correlations(df):
    rows = []
    # Primary: phase-matched HR and EEG preVR -> postVR changes.
    pairs = [
        ("hr_post_minus_pre", "eeg_post_minus_pre_frontal_alpha", "hr_eeg", "primary"),
        ("hr_post_minus_pre", "eeg_post_minus_pre_frontal_beta", "hr_eeg", "primary"),
        ("hr_post_minus_pre", "eeg_post_minus_pre_frontal_theta", "hr_eeg", "secondary"),
        ("hr_recovery_minus_post", "eeg_recovery_minus_post_frontal_alpha", "hr_eeg", "secondary"),
        ("hr_mean_vr_vs_pre", "delta_k", "hr_learning", "primary"),
        ("hr_mean_vr_vs_pre", "delta_accuracy", "hr_learning", "primary"),
        ("hr_max_vr_vs_pre", "delta_k", "hr_learning", "secondary"),
        ("hr_prep_vs_pre", "delta_k", "hr_learning", "secondary"),
        ("hr_speech_vs_pre", "delta_k", "hr_learning", "secondary"),
        ("hr_math_vs_pre", "delta_k", "hr_learning", "secondary"),
        ("eeg_post_minus_pre_frontal_alpha", "delta_k", "eeg_learning", "primary"),
        ("eeg_post_minus_pre_frontal_beta", "delta_k", "eeg_learning", "primary"),
        ("eeg_post_minus_pre_frontal_log_beta_alpha_ratio", "delta_k", "eeg_learning", "primary"),
        ("eeg_post_minus_pre_frontal_theta", "delta_k", "eeg_learning", "secondary"),
        ("eeg_post_minus_pre_frontal_alpha", "delta_accuracy", "eeg_learning", "secondary"),
        ("hr_mean_vr_vs_pre", "delta_winstay", "hr_strategy", "primary"),
        ("hr_mean_vr_vs_pre", "delta_loseswitch", "hr_strategy", "primary"),
        ("hr_mean_vr_vs_pre", "delta_strategy_bias", "hr_strategy", "secondary"),
    ]
    for x, y, family, priority in pairs:
        correlate(df, x, y, family, priority, rows)
    # Separate VR stages, frontal bands and channels: exploratory only.
    for hr_metric in ("hr_prep_vs_pre", "hr_speech_vs_pre", "hr_math_vs_pre",
                      "hr_mean_vr_vs_pre", "hr_max_vr_vs_pre", "hr_post_minus_pre"):
        for ch in SENSORS:
            for measure in (*BANDS, "log_beta_alpha_ratio"):
                eeg_metric = f"eeg_post_minus_pre_{ch}_{measure}"
                correlate(df, hr_metric, eeg_metric, "hr_eeg_stress_screen", "exploratory", rows)
                for outcome in ("delta_k", "delta_accuracy"):
                    correlate(df, eeg_metric, outcome, "eeg_learning_screen", "exploratory", rows)
    for hr_metric in ("hr_prep_vs_pre", "hr_speech_vs_pre", "hr_math_vs_pre"):
        for outcome in ("delta_winstay", "delta_loseswitch", "delta_strategy_bias"):
            correlate(df, hr_metric, outcome, "hr_strategy_screen", "exploratory", rows)
    result = pd.DataFrame(rows)
    result["spearman_p_fdr"] = np.nan
    for _, idx in result.groupby(["family", "priority"]).groups.items():
        result.loc[idx, "spearman_p_fdr"] = bh(result.loc[idx, "spearman_p"].to_numpy())
    return result


def models(df):
    rows = []
    # No automatic feature selection; avoid fitting unstable interactions by default.
    specs = [
        ("learning_rate", "delta_k", "hr_mean_vr_vs_pre", "eeg_post_minus_pre_frontal_alpha"),
        ("learning_rate_beta_alpha", "delta_k", "hr_mean_vr_vs_pre", "eeg_post_minus_pre_frontal_log_beta_alpha_ratio"),
        ("accuracy", "delta_accuracy", "hr_mean_vr_vs_pre", "eeg_post_minus_pre_frontal_alpha"),
        ("strategy_winstay", "delta_winstay", "hr_mean_vr_vs_pre", "eeg_post_minus_pre_frontal_alpha"),
    ]
    for name, y, x1, x2 in specs:
        sub = df[[y, x1, x2]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        n = len(sub)
        if n < MIN_MODEL_N or any(sub[c].nunique() < 3 for c in (x1, x2)):
            rows.append(dict(model=name, outcome=y, term="SKIPPED", n=n, note=f"Requires >= {MIN_MODEL_N} complete cases and variable predictors"))
            continue
        X = sub[[x1, x2]]
        std = X.std(ddof=0)
        if (std <= 0).any():
            rows.append(dict(model=name, outcome=y, term="SKIPPED", n=n, note="Zero predictor variance"))
            continue
        X = sm.add_constant((X - X.mean()) / std)
        fit = sm.OLS(sub[y], X).fit(cov_type="HC3")
        ci = fit.conf_int()
        for term in fit.params.index:
            rows.append(dict(model=name, outcome=y, term=term, n=n, coefficient=fit.params[term],
                             se=fit.bse[term], p=fit.pvalues[term], ci_low=ci.loc[term, 0],
                             ci_high=ci.loc[term, 1], r_squared=fit.rsquared,
                             note="HC3 robust SE; predictors z-scored; exploratory association"))
    return pd.DataFrame(rows)


def plot_primary(df, results):
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    primary = results[(results.priority == "primary") & (results.n >= MIN_N)]
    for row in primary.itertuples():
        pair = df[[row.predictor, row.outcome]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < MIN_N:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.scatter(pair[row.predictor], pair[row.outcome], alpha=.8)
        ax.set_xlabel(row.predictor.replace("_", " "))
        ax.set_ylabel(row.outcome.replace("_", " "))
        ax.set_title(f"n={len(pair)}; Spearman ρ={row.spearman_rho:.2f}; p={row.spearman_p:.3g}")
        fig.tight_layout()
        fig.savefig(figdir / f"{row.family}_{row.predictor}_vs_{row.outcome}.png", dpi=160)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(DATA_FILE, index=False)
    availability = pd.DataFrame({"measure": ["subjects_any", "hr", "learning_change", "eeg", "hr_eeg_learning"],
                                 "n": [len(df), int(df.has_hr.sum()), int(df.has_learning_change.sum()),
                                       int(df.has_eeg.sum()), int((df.has_hr & df.has_eeg & df.has_learning_change).sum())]})
    availability.to_csv(OUT / "sample_availability.csv", index=False)
    corr = correlations(df)
    exclusions = []
    for row in corr.itertuples():
        for _, subject in df.iterrows():
            absent = [col for col in (row.predictor, row.outcome)
                      if col not in df.columns or not np.isfinite(pd.to_numeric(subject.get(col), errors="coerce"))]
            exclusions.append({"family": row.family, "predictor": row.predictor,
                               "outcome": row.outcome, "subject_id": subject.subject_id,
                               "included": not bool(absent), "missing_fields": ";".join(absent)})
    pd.DataFrame(exclusions).to_csv(OUT / "subject_inclusion_by_test.csv", index=False)
    corr.to_csv(OUT / "correlations.csv", index=False)
    models(df).to_csv(OUT / "combined_models.csv", index=False)
    plot_primary(df, corr)
    report = {"n_subjects": len(df), "available": dict(zip(availability.measure, availability.n.astype(int))),
              "notes": ["Associations, not causal effects", "FDR within family and priority", "No missing data imputed",
                        "EEG anticipation=preVR and acute_stress_response=postVR; no EEG recorded during prep/speech/math",
                        "HR recovery_mean_hr is recovery; Plearn1 before preVR, Plearn2 after postVR",
                        "EEG frontal averages require both AF7 and AF8", "No interaction models fitted due to sample-size concerns"]}
    (OUT / "analysis_manifest.json").write_text(json.dumps(report, indent=2))
    print(f"Saved multimodal associations to {OUT}; subject dataset: {DATA_FILE}")


if __name__ == "__main__":
    main()
