#!/usr/bin/env python3
"""
This script processes all subjects’ trial‑level CSV files (each containing 10 blocks of 18 trials) 
from the Puberty_VR study. For each session (1 and 2) it:
- Computes the proportion of set‑winner (correct) choices for each trial position (1‑18) by averaging across all blocks.
- Saves a CSV file with columns: subject, session, trial, proportion_correct.
- Produces two graphs per session:
    (1) All individuals’ observed points (colored by subject) + individual exponential fits, with a legend outside the plot.
    (2) All individuals’ observed points (colored by subject) + group average exponential fit (with shaded SEM band), with a legend outside the plot.
Outputs: CSV and PNGs in outputs/all_individuals/.
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
from scipy.optimize import curve_fit
from scipy.stats import sem
from scipy.interpolate import interp1d
import sys 
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from project_paths import (PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR)

def exp_learning(t, a, b, k):
    return a - b * np.exp(-k * t)

# ---------- USER SETTINGS ----------
MAX_LEGEND_SUBJECTS = 20   # Show full subject legend only if number of subjects <= this value

# ---------- PATHS ----------
RESULTS_DIR = PLEARNING_DIR
OUTPUT_DIR = FIGURES_DIR / "all_individuals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

trials_per_block = 18
sessions = [1, 2]

# ---------- LOAD DATA FOR BOTH SESSIONS ----------
all_data_by_session = {sess: [] for sess in sessions}

for csv_file in RESULTS_DIR.rglob("*_plearning_*.csv"):
    parts = csv_file.stem.split("_")
    if len(parts) < 3:
        continue
    subj_id = parts[0]
    try:
        sess = int(parts[2])
    except:
        continue
    if sess not in sessions:
        continue
    df = pd.read_csv(csv_file)
    if 'learned' not in df.columns or 'trialNumber' not in df.columns:
        continue
    df['learned'] = df['learned'].astype(int)
    trial_props = [df[df['trialNumber'] == t]['learned'].mean() for t in range(1, trials_per_block+1)]
    all_data_by_session[sess].append({'subject': subj_id, 'proportions': trial_props})

for sess in sessions:
    if not all_data_by_session[sess]:
        print(f"Warning: No data for session {sess}, skipping.")
        continue

    data = all_data_by_session[sess]
    subjects = [d['subject'] for d in data]
    n_subj = len(subjects)

    # ---------- SAVE RAW PROPORTIONS CSV ----------
    rows = []
    for subj in data:
        for t, p in enumerate(subj['proportions'], start=1):
            rows.append({'subject': subj['subject'], 'session': sess, 'trial': t, 'proportion_correct': p})
    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(LEARNING_ANALYSIS_DATA_DIR / f"individual_proportions_session{sess}.csv", index=False)
    # ---------- PREPARE FOR PLOTTING ----------
    x = np.arange(1, trials_per_block+1)
    dense_x = np.linspace(1, 18, 200)

    # Assign a consistent color to each subject (using tab10 or tab20 colormap)
    if n_subj <= 10:
        colormap = plt.cm.tab10
    else:
        colormap = plt.cm.tab20
    colors = [colormap(i % colormap.N) for i in range(n_subj)]
    subject_color = {subj['subject']: colors[idx] for idx, subj in enumerate(data)}

    # ----- GRAPH 1: Individual points + individual fits (legend outside) -----
    plt.figure(figsize=(12, 6))
    for subj in data:
        y = np.array(subj['proportions'])
        valid = ~np.isnan(y)
        xv, yv = x[valid], y[valid]
        color = subject_color[subj['subject']]
        if len(xv) >= 4:
            try:
                p0 = [np.max(yv), np.max(yv)-np.min(yv), 0.3]
                bounds = ([0.5,0,0], [1,1,2])
                popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
                y_smooth = exp_learning(dense_x, *popt)
                plt.plot(dense_x, y_smooth, linewidth=1, alpha=0.6, color=color, label='_nolegend_')
            except:
                pass
        plt.plot(x, y, 'o', markersize=3, alpha=0.7, color=color,
                 label=subj['subject'] if n_subj <= MAX_LEGEND_SUBJECTS else '_nolegend_')
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel('Trial number')
    plt.ylabel('Proportion choosing set-winner')
    plt.ylim(0,1)
    plt.xlim(0.9, 18.1)
    plt.xticks(range(1, trials_per_block+1))
    plt.title(f'Session {sess} – All individuals (points = observed, lines = individual exponential fits)')
    if n_subj <= MAX_LEGEND_SUBJECTS:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        print(f"Session {sess}: Too many subjects ({n_subj}) to show legend; legend omitted.")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'all_individuals_with_curves_sess{sess}.png', dpi=150)
    plt.close()

    # ----- GRAPH 2: All individual points + group average curve (legend outside) -----
    props_array = np.array([d['proportions'] for d in data])
    group_mean = np.nanmean(props_array, axis=0)
    group_sem = sem(props_array, axis=0, nan_policy='omit')
    
    plt.figure(figsize=(12, 6))
    # Plot all individual points (colored by subject)
    for subj in data:
        y = np.array(subj['proportions'])
        color = subject_color[subj['subject']]
        plt.plot(x, y, 'o', markersize=3, alpha=0.5, color=color,
                 label=subj['subject'] if n_subj <= MAX_LEGEND_SUBJECTS else '_nolegend_')
    # Fit exponential to group average
    valid = ~np.isnan(group_mean)
    xv, yv = x[valid], group_mean[valid]
    if len(xv) >= 4:
        try:
            p0 = [np.max(yv), np.max(yv)-np.min(yv), 0.3]
            bounds = ([0.5,0,0], [1,1,2])
            popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
            y_smooth = exp_learning(dense_x, *popt)
            plt.plot(dense_x, y_smooth, color='blue', linewidth=2, label='Group average (exponential fit)')
            # Shaded SEM band
            sem_valid = ~np.isnan(group_sem)
            if np.sum(sem_valid) >= 2:
                interp_sem = interp1d(x[sem_valid], group_sem[sem_valid], kind='linear', fill_value='extrapolate')
                sem_smooth = interp_sem(dense_x)
                plt.fill_between(dense_x, y_smooth - sem_smooth, y_smooth + sem_smooth, alpha=0.2, color='blue')
        except Exception as e:
            print(f"Group fit failed for session {sess}: {e}")
            plt.plot(xv, yv, 'o-', color='blue', label='Group average (observed)')
    else:
        plt.plot(xv, yv, 'o-', color='blue', label='Group average (observed)')
    # Also plot group mean points
    plt.scatter(xv, yv, color='blue', s=40, edgecolor='black', zorder=5, label='_nolegend_')
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel('Trial number')
    plt.ylabel('Proportion choosing set-winner')
    plt.ylim(0,1)
    plt.xlim(0.9, 18.1)
    plt.xticks(range(1, trials_per_block+1))
    plt.title(f'Session {sess} – All individuals (colored points) + group average curve')
    # Place legend outside to the right
    if n_subj <= MAX_LEGEND_SUBJECTS:
        # Subject legend plus group average and chance
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        # Show only group average and chance
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out the subject entries (they have label '_nolegend_' when not shown)
        # We can simply call legend with the existing handles (they already exclude '_nolegend_')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'all_individuals_with_group_avg_sess{sess}.png', dpi=150)
    plt.close()

print(f"All outputs saved in {OUTPUT_DIR}")
