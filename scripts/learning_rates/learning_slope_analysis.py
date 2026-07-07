#!/usr/bin/env python3
"""
Learning Slope Analysis for Pre‑ vs Post‑VR Stress

This script:
1. Loads raw trial data for sessions 1 (Pre) and 2 (Post).
2. Creates a trial‑level CSV with columns: Subject, Session, Block, Trial, Correct, LearnerType.
3. For each subject and session, estimates a learning slope (linear regression of
   proportion correct on trial number, using data aggregated across blocks).
4. Performs paired t‑tests and Cohen's d on slopes (Pre vs Post) for:
   - All subjects
   - Good learners only
   - Bad learners only
5. Fits a linear mixed model (LMM) on the raw trial‑level data:
   Fixed effects: Trial, Session, Trial × Session
   Random intercept: Subject
   Interpretation: negative Trial×Session[Post] coefficient = slower learning after stress.
6. Generates graphs: paired slopes, group learning curves, individual learning curves.

Dependencies: pandas, numpy, scipy, statsmodels, matplotlib
"""

import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
from scipy.stats import ttest_rel, sem
import statsmodels.formula.api as smf

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys 

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR, REPORTS_DIR

# ---------- PATHS ----------
BASE_PATH = PLEARNING_DIR
LEARNING_RATES_CSV = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
OUTPUT_DIR = LEARNING_ANALYSIS_DATA_DIR / "slope_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIALS_PER_BLOCK = 18
BLOCKS_PER_SESSION = 10
SESSIONS = {1: "Pre", 2: "Post"}

# ---------- 1. CLASSIFY GOOD / BAD LEARNERS (using Session 1 k) ----------
lr_df = pd.read_csv(LEARNING_RATES_CSV, dtype={'subjectId': str})
lr_df = lr_df[lr_df['plearning_num'] == 1].copy()
lr_df = lr_df.dropna(subset=['learning_rate_k'])
lr_df["Subject"] = lr_df["subjectId"].str.zfill(3)
if "learner_group" in lr_df.columns:
    lr_df["LearnerType"] = lr_df["learner_group"].astype(str).str.lower()
    print("Using learner_group from learning_rates.csv")
else:
    median_k = lr_df["learning_rate_k"].median()
    lr_df["LearnerType"] = np.where(lr_df["learning_rate_k"] >= median_k, "good", "bad")
    print(f"Median k (Session 1) = {median_k:.4f}")
print(f"Good learners: {len(lr_df[lr_df['LearnerType'] == 'good'])}")
print(f"Bad learners:  {len(lr_df[lr_df['LearnerType'] == 'bad'])}")

# ---------- 2. BUILD TRIAL-LEVEL CSV ----------
all_rows = []
for csv_file in BASE_PATH.rglob("*_plearning_*.csv"):
    parts = csv_file.stem.split("_")
    if len(parts) < 3:
        continue
    subj_id = parts[0].zfill(3)
    try:
        session_num = int(parts[2])
    except:
        continue
    if session_num not in SESSIONS:
        continue

    if subj_id not in lr_df['Subject'].values:
        print(f"Skipping {subj_id} – missing learning rate")
        continue
    learner_type = lr_df.loc[lr_df['Subject'] == subj_id, 'LearnerType'].iloc[0]

    df = pd.read_csv(csv_file)
    if 'learned' not in df.columns or 'trialNumber' not in df.columns or 'blockNumber' not in df.columns:
        continue
    df['learned'] = df['learned'].astype(int)

    for _, row in df.iterrows():
        all_rows.append({
            'Subject': subj_id,
            'Session': SESSIONS[session_num],
            'Block': row['blockNumber'],
            'Trial': row['trialNumber'],
            'Correct': row['learned'],
            'LearnerType': learner_type
        })

trial_df = pd.DataFrame(all_rows)
trial_csv = OUTPUT_DIR / 'trial_level_data.csv'
trial_df.to_csv(trial_csv, index=False)
print(f"Saved trial-level data: {trial_csv}")
print(f"Shape: {trial_df.shape}")

# ---------- 3. COMPUTE LEARNING SLOPES ----------
def compute_slope(subject_data, session_name):
    subj_sess = trial_df[(trial_df['Subject'] == subject_data) & (trial_df['Session'] == session_name)]
    if subj_sess.empty:
        return np.nan
    props = []
    for t in range(1, TRIALS_PER_BLOCK+1):
        vals = subj_sess[subj_sess['Trial'] == t]['Correct']
        props.append(vals.mean() if len(vals) > 0 else np.nan)
    props = np.array(props)
    valid = ~np.isnan(props)
    if np.sum(valid) < 2:
        return np.nan
    x = np.arange(1, TRIALS_PER_BLOCK+1)[valid]
    y = props[valid]
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]

subjects = trial_df['Subject'].unique()
slope_data = []
for subj in subjects:
    slope_pre = compute_slope(subj, 'Pre')
    slope_post = compute_slope(subj, 'Post')
    lt = trial_df[trial_df['Subject'] == subj]['LearnerType'].iloc[0]
    slope_data.append({
        'Subject': subj,
        'LearnerType': lt,
        'slope_pre': slope_pre,
        'slope_post': slope_post,
        'delta_slope': slope_post - slope_pre
    })
slope_df = pd.DataFrame(slope_data)
slope_csv = OUTPUT_DIR / 'learning_slopes.csv'
slope_df.to_csv(slope_csv, index=False)
print(f"Saved slopes: {slope_csv}")

# ---------- 4. PAIRED T-TESTS & COHEN'S D ----------
def cohens_d_paired(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

print("\n" + "="*50)
print("PAIRED T-TESTS (Pre vs Post slopes)")
print("="*50)

for group, group_df in [('All subjects', slope_df),
                        ('Good learners', slope_df[slope_df['LearnerType']=='good']),
                        ('Bad learners',  slope_df[slope_df['LearnerType']=='bad'])]:
    if len(group_df) < 2:
        print(f"\n{group}: insufficient subjects ({len(group_df)}) – skipping")
        continue
    pre = group_df['slope_pre'].dropna()
    post = group_df['slope_post'].dropna()
    common_idx = pre.index.intersection(post.index)
    pre = pre[common_idx]
    post = post[common_idx]
    if len(pre) < 2:
        print(f"\n{group}: not enough complete pairs ({len(pre)}) – skipping")
        continue
    t, p = ttest_rel(pre, post, nan_policy='omit')
    d = cohens_d_paired(pre, post)
    print(f"\n{group} (n={len(pre)}):")
    print(f"  Mean Pre slope  = {pre.mean():.4f}")
    print(f"  Mean Post slope = {post.mean():.4f}")
    print(f"  t = {t:.3f}, p = {p:.4f}")
    print(f"  Cohen's d = {d:.3f} (negative = slower after stress)")

# ---------- 5. LINEAR MIXED MODEL (raw trial data, Trial × Session) ----------
print("\n" + "="*50)
print("LINEAR MIXED MODEL (Trial × Session interaction) – Raw trial data")
print("="*50)

trial_df_model = trial_df.copy()
trial_df_model['Session'] = pd.Categorical(trial_df_model['Session'], categories=['Pre', 'Post'], ordered=False)

try:
    model_raw = smf.mixedlm("Correct ~ Trial * Session", trial_df_model, groups=trial_df_model["Subject"])
    result_raw = model_raw.fit()
    print(result_raw.summary())
except Exception as e:
    print(f"Mixed model failed: {e}")
    result_raw = None

# ---------- 6. GRAPHS ----------
GRAPH_DIR = FIGURES_DIR / "slope_analysis"
GRAPH_DIR.mkdir(exist_ok=True)

# 6a. Paired slope plots
def plot_paired_slopes(data, group_name, color, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))
    else:
        fig = ax.figure
    valid = data.dropna(subset=['slope_pre', 'slope_post'])
    if len(valid) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return ax
    pre = valid['slope_pre'].values
    post = valid['slope_post'].values
    n = len(pre)
    mean_pre, mean_post = pre.mean(), post.mean()
    sem_pre, sem_post = sem(pre), sem(post)
    ax.bar([0, 1], [mean_pre, mean_post], yerr=[sem_pre, sem_post], capsize=5,
           color=color, alpha=0.6, edgecolor='black', width=0.6, label='Mean ± SEM')
    jitter = np.random.normal(0, 0.05, size=n)
    for i in range(n):
        ax.plot([0 + jitter[i], 1 + jitter[i]], [pre[i], post[i]], 
                color='gray', linewidth=0.8, alpha=0.5)
        ax.scatter(0 + jitter[i], pre[i], color=color, s=40, alpha=0.7, edgecolor='black', zorder=3)
        ax.scatter(1 + jitter[i], post[i], color=color, s=40, alpha=0.7, edgecolor='black', zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre', 'Post'])
    ax.set_ylabel('Learning slope (proportion change per trial)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{group_name} (n={n})')
    ax.legend(loc='best')
    return ax

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
groups = [('All subjects', slope_df, 'steelblue'),
          ('Good learners', slope_df[slope_df['LearnerType']=='good'], 'green'),
          ('Bad learners', slope_df[slope_df['LearnerType']=='bad'], 'coral')]
for ax, (name, data, color) in zip(axes, groups):
    plot_paired_slopes(data, name, color, ax=ax)
plt.tight_layout()
plt.savefig(GRAPH_DIR / 'paired_slopes_all_groups.png', dpi=150)
plt.close()
print(f"Saved: {GRAPH_DIR / 'paired_slopes_all_groups.png'}")

# 6b. Group learning curves (using aggregated proportions)
# First aggregate
agg_list = []
for subj in subjects:
    for sess_name in ['Pre', 'Post']:
        subj_sess = trial_df[(trial_df['Subject'] == subj) & (trial_df['Session'] == sess_name)]
        if subj_sess.empty:
            continue
        for t in range(1, TRIALS_PER_BLOCK+1):
            vals = subj_sess[subj_sess['Trial'] == t]['Correct']
            prop = vals.mean() if len(vals) > 0 else np.nan
            if not np.isnan(prop):
                agg_list.append({'Subject': subj, 'Session': sess_name, 'Trial': t, 'PropCorrect': prop})
agg_df = pd.DataFrame(agg_list)
agg_df['LearnerType'] = agg_df['Subject'].map(slope_df.set_index('Subject')['LearnerType'])

all_groups = [('All', agg_df), ('Good', agg_df[agg_df['LearnerType']=='good']), ('Bad', agg_df[agg_df['LearnerType']=='bad'])]
colors_curve = {'All': 'gray', 'Good': 'green', 'Bad': 'red'}
line_styles = {'Pre': '-', 'Post': '--'}

for group_name, group_data in all_groups:
    if group_data.empty:
        continue
    plt.figure(figsize=(8, 6))
    for sess in ['Pre', 'Post']:
        sess_data = group_data[group_data['Session'] == sess]
        if sess_data.empty:
            continue
        means, sems = [], []
        for t in range(1, TRIALS_PER_BLOCK+1):
            vals = sess_data[sess_data['Trial'] == t]['PropCorrect']
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            sems.append(sem(vals) if len(vals) > 1 else 0)
        means = np.array(means)
        sems = np.array(sems)
        valid = ~np.isnan(means)
        if not np.any(valid):
            continue
        x = np.arange(1, TRIALS_PER_BLOCK+1)[valid]
        y = means[valid]
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        fit_line = intercept + slope * x
        plt.errorbar(x, y, yerr=sems[valid], fmt='o', color=colors_curve[group_name], 
                     capsize=3, label=f'{sess} (observed)', alpha=0.6)
        plt.plot(x, fit_line, color=colors_curve[group_name], linestyle=line_styles[sess],
                 linewidth=2, label=f'{sess} (linear fit, slope={slope:.3f})')
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel('Trial number')
    plt.ylabel('Proportion correct')
    plt.ylim(0, 1)
    plt.title(f'{group_name} learners – Pre vs Post learning curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f'learning_curves_{group_name.lower()}.png', dpi=150)
    plt.close()
    print(f"Saved: {GRAPH_DIR / f'learning_curves_{group_name.lower()}.png'}")

# 6c. Individual subject learning curves
INDIV_DIR = GRAPH_DIR / "individual_curves"
INDIV_DIR.mkdir(exist_ok=True)
subjects_with_both = agg_df['Subject'].unique()
for subj in subjects_with_both:
    subj_data = agg_df[agg_df['Subject'] == subj]
    if subj_data.empty:
        continue
    lt = slope_df[slope_df['Subject'] == subj]['LearnerType'].iloc[0] if subj in slope_df['Subject'].values else 'unknown'
    plt.figure(figsize=(8, 6))
    for sess in ['Pre', 'Post']:
        sess_data = subj_data[subj_data['Session'] == sess]
        if sess_data.empty:
            continue
        x = sess_data['Trial'].values
        y = sess_data['PropCorrect'].values
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        fit_line = intercept + slope * x
        plt.plot(x, y, 'o', label=f'{sess} observed', alpha=0.7)
        plt.plot(x, fit_line, '--', linewidth=2, label=f'{sess} fit (slope={slope:.3f})')
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel('Trial number')
    plt.ylabel('Proportion correct')
    plt.ylim(0, 1)
    plt.title(f'Subject {subj} ({lt} learner) – Pre vs Post learning curves')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(INDIV_DIR / f'subject_{subj}_learning_curve.png', dpi=150)
    plt.close()
print(f"Individual subject graphs saved in: {INDIV_DIR}")

# ---------- 7. SAVE ALL RESULTS TO A SINGLE TXT FILE ----------
results_file = REPORTS_DIR / 'learning_slope_analysis_results.txt'
with open(results_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("INDIVIDUAL SUBJECT SLOPES (Pre vs Post)\n")
    f.write("="*60 + "\n\n")
    ind_slopes = slope_df[['Subject', 'LearnerType', 'slope_pre', 'slope_post', 'delta_slope']].copy()
    ind_slopes = ind_slopes.sort_values('Subject')
    f.write(f"{'Subject':<8} {'LearnerType':<12} {'slope_pre':>10} {'slope_post':>10} {'delta_slope':>12}\n")
    f.write("-" * 55 + "\n")
    for _, row in ind_slopes.iterrows():
        f.write(f"{row['Subject']:<8} {row['LearnerType']:<12} {row['slope_pre']:10.4f} {row['slope_post']:10.4f} {row['delta_slope']:12.4f}\n")
    f.write("\n")

    f.write("="*60 + "\n")
    f.write("PAIRED T-TESTS (Pre vs Post slopes)\n")
    f.write("="*60 + "\n\n")
    for group, group_df in [('All subjects', slope_df),
                            ('Good learners', slope_df[slope_df['LearnerType']=='good']),
                            ('Bad learners',  slope_df[slope_df['LearnerType']=='bad'])]:
        if len(group_df) < 2:
            f.write(f"{group}: insufficient subjects ({len(group_df)}) – skipping\n\n")
            continue
        pre = group_df['slope_pre'].dropna()
        post = group_df['slope_post'].dropna()
        common_idx = pre.index.intersection(post.index)
        pre = pre[common_idx]
        post = post[common_idx]
        if len(pre) < 2:
            f.write(f"{group}: not enough complete pairs ({len(pre)}) – skipping\n\n")
            continue
        t, p = ttest_rel(pre, post, nan_policy='omit')
        d = cohens_d_paired(pre, post)
        f.write(f"{group} (n={len(pre)}):\n")
        f.write(f"  Mean Pre slope  = {pre.mean():.4f}\n")
        f.write(f"  Mean Post slope = {post.mean():.4f}\n")
        f.write(f"  t = {t:.3f}, p = {p:.4f}\n")
        f.write(f"  Cohen's d = {d:.3f} (negative = slower after stress)\n\n")

    f.write("="*60 + "\n")
    f.write("LINEAR MIXED MODEL (Trial × Session interaction) – Raw trial data\n")
    f.write("="*60 + "\n\n")
    if result_raw is not None:
        f.write(result_raw.summary().as_text())
    else:
        f.write("Mixed model failed to fit.\n")

    f.write("\n" + "="*60 + "\n")
    f.write("SAVED GRAPHS\n")
    f.write("="*60 + "\n")
    f.write(f"Paired slopes plot (Pre vs Post for all groups):\n  {GRAPH_DIR / 'paired_slopes_all_groups.png'}\n")
    f.write(f"Learning curves with linear fits (group averages):\n")
    for group in ['all', 'good', 'bad']:
        f.write(f"  {GRAPH_DIR / f'learning_curves_{group}.png'}\n")
    f.write(f"Individual subject learning curves:\n  {INDIV_DIR}\n")
    f.write("\nNote: Cohen's d negative indicates lower learning slope after stress.\n")
    f.write("Benchmarks: 0.20 small, 0.50 medium, 0.80 large.\n")
    f.write("delta_slope = slope_post - slope_pre (negative = slower after stress).\n")

print(f"All results saved to: {results_file}")
print("\nAll done. Results are in:", OUTPUT_DIR)
