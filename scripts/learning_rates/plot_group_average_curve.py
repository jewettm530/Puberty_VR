#!/usr/bin/env python3
"""
Exponential Learning Curve Plotter for Plearning Task – Session 1 & 2

This script processes trial-level CSV files (each containing 10 blocks of 18 trials)
from subjects in the Puberty_VR study. It:
1. Loads precomputed individual learning rates (k) from learning_rates.csv (Session 1 only).
2. Performs a median split on Session 1 k values to classify subjects as 'good' or 'bad' learners.
3. For Session 1 and Session 2 separately:
   a. Computes the mean proportion of set-winner choices for each trial position (1-18)
      across all blocks and subjects for three groups: good learners, bad learners, and all subjects.
   b. Fits an exponential (diminishing-returns) curve: P(t) = a - b * exp(-k*t)
      to each group's average data.
   c. Plots the fitted smooth curve with a shaded band representing the interpolated
      standard error of the mean (SEM) of the observed proportions.
4. Saves two figures:
   - 'group_average_exponential_learning_curves_session_1.png'
   - 'group_average_exponential_learning_curves_session_2.png'
   under outputs/learning_rates/figures/group_average_curves/.

The overall (all subjects) curve is added to visualise how good/bad learners deviate
from the group average.

Dependencies: pandas, numpy, matplotlib, scipy, statsmodels (optional for LOWESS fallback)
"""

import pandas as pd
import numpy as np
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import sys 

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR

# ---------- EXPONENTIAL FUNCTION ----------
def exp_learning(t, a, b, k):
    return a - b * np.exp(-k * t)

# ---------- PATHS ----------
base_path = PLEARNING_DIR
learning_rates_csv = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
trials_per_block = 18
sessions = [1, 2]

# ---------- LOAD LEARNING RATES FROM SESSION 1 (for classification) ----------
lr_df = pd.read_csv(learning_rates_csv, dtype={'subjectId': str})
lr_df = lr_df[lr_df['plearning_num'] == 1].copy()
lr_df = lr_df.dropna(subset=['learning_rate_k'])

median_k = lr_df['learning_rate_k'].median()
lr_df['group'] = np.where(lr_df['learning_rate_k'] >= median_k, 'good', 'bad')
lr_df['subjectId'] = lr_df['subjectId'].str.zfill(3)
print(f"Median k (session 1) = {median_k:.4f}")
print(f"Good learners: {len(lr_df[lr_df['group']=='good'])}")
print(f"Bad learners:  {len(lr_df[lr_df['group']=='bad'])}")

# ---------- FUNCTION TO LOAD TRIAL DATA FOR A GIVEN SESSION ----------
def load_session_data(session_num):
    subject_means = []
    for csv_file in base_path.rglob("*_plearning_*.csv"):
        parts = csv_file.stem.split("_")
        if len(parts) < 3:
            continue
        subj_id = parts[0]
        try:
            p_num = int(parts[2])
        except:
            continue
        if p_num != session_num:
            continue
        df = pd.read_csv(csv_file)
        if 'learned' not in df.columns or 'trialNumber' not in df.columns:
            continue
        df['learned'] = df['learned'].astype(int)
        trial_means = []
        for t in range(1, trials_per_block+1):
            vals = df[df['trialNumber'] == t]['learned'].values
            trial_means.append(np.mean(vals) if len(vals) > 0 else np.nan)
        subject_means.append({'subject': str(subj_id), 'means': trial_means})
    if not subject_means:
        raise ValueError(f"No valid subject data found for session {session_num}")
    return subject_means

# ---------- FUNCTION TO GENERATE AND SAVE PLOT FOR ONE SESSION ----------
def plot_session(session_num, subject_means, output_dir):
    # Convert to long DataFrame and merge with group labels
    long_df = pd.DataFrame([
        {'subject': s['subject'], 'trial_pos': t+1, 'prop_correct': s['means'][t]}
        for s in subject_means for t in range(trials_per_block)
    ])
    # Keep only subjects that are in lr_df (i.e., have a group label)
    long_df = long_df.merge(lr_df[['subjectId', 'group']], left_on='subject', right_on='subjectId')

    # Compute stats for good, bad, and overall (all subjects in long_df)
    group_stats = {}
    for group in ['good', 'bad', 'all']:
        if group == 'all':
            group_data = long_df
        else:
            group_data = long_df[long_df['group'] == group]
        means, sems = [], []
        for t in range(1, trials_per_block+1):
            vals = group_data[group_data['trial_pos'] == t]['prop_correct'].values
            if len(vals) == 0:
                means.append(np.nan)
                sems.append(np.nan)
            else:
                means.append(np.mean(vals))
                sems.append(sem(vals) if len(vals) > 1 else 0)
        group_stats[group] = {'mean': np.array(means), 'sem': np.array(sems),
                              'trial': np.arange(1, trials_per_block+1)}

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = {'good': 'green', 'bad': 'red', 'all': 'gray'}
    line_styles = {'good': '-', 'bad': '-', 'all': '--'}  # overall dashed
    dense_x = np.linspace(1, 18, 200)

    for group in ['good', 'bad', 'all']:
        stats = group_stats[group]
        x = stats['trial']
        y = stats['mean']
        sem_vals = stats['sem']
        valid = ~np.isnan(y)
        xv, yv = x[valid], y[valid]
        if len(xv) < 4:
            print(f"Not enough valid points for {group} in session {session_num}, skipping fit.")
            # Still plot observed points
            plt.scatter(xv, yv, color=colors[group], s=50, edgecolor='black', zorder=5,
                        label=f'{group.capitalize()} (observed)' if group != 'all' else 'All subjects (observed)')
            continue

        try:
            p0 = [np.max(yv), np.max(yv) - np.min(yv), 0.3]
            bounds = ([0.5, 0, 0], [1, 1, 2])
            popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
            a_fit, b_fit, k_fit = popt
            print(f"Session {session_num} - {group.capitalize()}: asymptote = {a_fit:.3f}, deficit = {b_fit:.3f}, k = {k_fit:.3f}")

            y_smooth = exp_learning(dense_x, a_fit, b_fit, k_fit)

            # Interpolate SEM
            sem_valid = ~np.isnan(sem_vals)
            if np.sum(sem_valid) >= 2:
                interp_sem = interp1d(x[sem_valid], sem_vals[sem_valid], kind='linear', fill_value='extrapolate')
                sem_smooth = interp_sem(dense_x)
            else:
                sem_smooth = np.full_like(dense_x, np.nanmean(sem_vals[valid]))
            sem_smooth = np.maximum(sem_smooth, 0)

            # Plot fit line and shaded SEM
            plt.plot(dense_x, y_smooth, color=colors[group], linewidth=2, linestyle=line_styles[group],
                     label=f'{group.capitalize()} learners (exponential fit)' if group != 'all' else 'All subjects (exponential fit)')
            plt.fill_between(dense_x, y_smooth - sem_smooth, y_smooth + sem_smooth,
                             alpha=0.2, color=colors[group])
        except Exception as e:
            print(f"Exponential fit failed for {group} in session {session_num}: {e}")
            # Plot observed points only
            plt.scatter(xv, yv, color=colors[group], s=50, edgecolor='black', zorder=5,
                        label=f'{group.capitalize()} (observed)' if group != 'all' else 'All subjects (observed)')
            continue

        # Plot observed means as markers
        plt.scatter(xv, yv, color=colors[group], s=50, edgecolor='black', zorder=5)

    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance (50%)')
    plt.xlabel('Trial number', fontsize=12)
    plt.ylabel('Proportion choosing set-winner', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(range(1, 19))
    plt.title(f'Group average learning curves – Session {session_num}\nMedian split on learning rate (Session 1)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    outfile = output_dir / f'group_average_exponential_learning_curves_session_{session_num}.png'
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Figure saved to {outfile}")

# ---------- MAIN EXECUTION ----------
output_dir = FIGURES_DIR / "group_average_curves"
output_dir.mkdir(parents=True, exist_ok=True)

for sess in sessions:
    print(f"\nProcessing session {sess}...")
    subject_means = load_session_data(sess)
    plot_session(sess, subject_means, output_dir)

print("\nAll done.")
