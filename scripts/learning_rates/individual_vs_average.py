#!/usr/bin/env python3
"""
individual_vs_average.py

For each subject, this script creates a plot that overlays:
- The subject’s own learning curve (observed proportion of set‑winner choices per trial + exponential fit) in blue.
- The group average learning curve (all subjects, regardless of classification) with an exponential fit and a shaded standard error band (SEM) in gray.
- The subject is classified as ‘good’ or ‘bad’ learner based on a median split of their fitted learning rate (k) from learning_rates.csv (or fallback overall accuracy).
- The subject’s fitted exponential parameters (a, b, k) are annotated on the plot.

Output: one PNG per subject in outputs/learning_rates/figures/individual_vs_average/.
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

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR

# ---------- EXPONENTIAL FUNCTION ----------
def exp_learning(t, a, b, k):
    return a - b * np.exp(-k * t)

# ---------- PORTABLE PATHS ----------
RESULTS_DIR = PLEARNING_DIR
OUTPUT_DIR = FIGURES_DIR / "individual_vs_average"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plearning_session = 1
trials_per_block = 18

# ---------- LOAD CLASSIFICATION (GOOD/BAD) ----------
# First, load learning_rates.csv if available, otherwise compute overall accuracy median split
learning_rates_csv = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
if learning_rates_csv.exists():
    lr_df = pd.read_csv(learning_rates_csv, dtype={'subjectId': str})
    lr_df = lr_df[lr_df['plearning_num'] == plearning_session].copy()
    lr_df = lr_df.dropna(subset=['learning_rate_k'])
    median_k = lr_df['learning_rate_k'].median()
    lr_df['group'] = np.where(lr_df['learning_rate_k'] >= median_k, 'good', 'bad')
    lr_df['subjectId'] = lr_df['subjectId'].str.zfill(3)
    group_map = lr_df.set_index('subjectId')['group'].to_dict()
    # Also store individual learning rates if needed for annotation
    k_map = lr_df.set_index('subjectId')['learning_rate_k'].to_dict()
else:
    # Fallback: compute overall proportion correct and split
    group_map = {}
    k_map = {}
    overall_acc = []
    subj_list = []
    for csv_file in RESULTS_DIR.rglob("*_plearning_*.csv"):
        parts = csv_file.stem.split("_")
        if len(parts) < 3: continue
        subj_id = parts[0]
        try: p_num = int(parts[2])
        except: continue
        if p_num != plearning_session: continue
        df = pd.read_csv(csv_file)
        if 'learned' not in df.columns: continue
        overall = df['learned'].mean()
        overall_acc.append(overall)
        subj_list.append(subj_id)
    median_val = np.median(overall_acc)
    for subj_id, acc in zip(subj_list, overall_acc):
        group_map[subj_id] = 'good' if acc >= median_val else 'bad'
        k_map[subj_id] = np.nan  # no individual k from logistic fit

# ---------- LOAD ALL SUBJECTS' TRIAL PROPORTIONS ----------
all_subjects = []   # each: {'subject': id, 'proportions': list(18)}
for csv_file in RESULTS_DIR.rglob("*_plearning_*.csv"):
    parts = csv_file.stem.split("_")
    if len(parts) < 3:
        continue
    subj_id = parts[0]
    try:
        p_num = int(parts[2])
    except:
        continue
    if p_num != plearning_session:
        continue
    df = pd.read_csv(csv_file)
    if 'learned' not in df.columns or 'trialNumber' not in df.columns:
        continue
    df['learned'] = df['learned'].astype(int)
    trial_props = []
    for t in range(1, trials_per_block+1):
        vals = df[df['trialNumber'] == t]['learned'].values
        trial_props.append(np.mean(vals) if len(vals) > 0 else np.nan)
    all_subjects.append({'subject': str(subj_id), 'proportions': trial_props})

if not all_subjects:
    raise ValueError("No valid subject data found")

# ---------- COMPUTE GROUP AVERAGE (ALL SUBJECTS) ----------
x = np.arange(1, trials_per_block+1)
all_props_array = np.array([s['proportions'] for s in all_subjects])
group_mean = np.nanmean(all_props_array, axis=0)
group_sem = sem(all_props_array, axis=0, nan_policy='omit')
valid_group = ~np.isnan(group_mean)
xv_group, yv_group = x[valid_group], group_mean[valid_group]

# Fit exponential to group average
dense_x = np.linspace(1, 18, 200)
group_fit_success = False
try:
    p0_group = [np.max(yv_group), np.max(yv_group)-np.min(yv_group), 0.3]
    bounds = ([0.5,0,0], [1,1,2])
    popt_group, _ = curve_fit(exp_learning, xv_group, yv_group, p0=p0_group, bounds=bounds, maxfev=5000)
    a_group, b_group, k_group = popt_group
    y_smooth_group = exp_learning(dense_x, a_group, b_group, k_group)
    # Interpolate SEM for shaded band
    sem_valid = ~np.isnan(group_sem)
    if np.sum(sem_valid) >= 2:
        interp_sem = interp1d(x[sem_valid], group_sem[sem_valid], kind='linear', fill_value='extrapolate')
        sem_smooth = interp_sem(dense_x)
    else:
        sem_smooth = np.full_like(dense_x, np.nanmean(group_sem[valid_group]))
    group_fit_success = True
except Exception as e:
    print(f"Group exponential fit failed: {e}")

# ---------- GENERATE ONE PLOT PER SUBJECT ----------
for subj in all_subjects:
    subj_id = subj['subject']
    y_ind = np.array(subj['proportions'])
    valid_ind = ~np.isnan(y_ind)
    xv_ind, yv_ind = x[valid_ind], y_ind[valid_ind]
    
    plt.figure(figsize=(10, 6))
    
    # ---- Individual curve (blue) ----
    if len(xv_ind) >= 4:
        try:
            p0_ind = [np.max(yv_ind), np.max(yv_ind)-np.min(yv_ind), 0.3]
            popt_ind, _ = curve_fit(exp_learning, xv_ind, yv_ind, p0=p0_ind, bounds=bounds, maxfev=5000)
            a_ind, b_ind, k_ind = popt_ind
            y_smooth_ind = exp_learning(dense_x, a_ind, b_ind, k_ind)
            plt.plot(dense_x, y_smooth_ind, color='blue', linewidth=2, label=f'Subject {subj_id} (fit)')
            # Annotate parameters
            param_text = f"a={a_ind:.2f}, b={b_ind:.2f}, k={k_ind:.3f}"
        except Exception as e:
            print(f"Individual fit failed for {subj_id}: {e}")
            plt.plot(xv_ind, yv_ind, 'o-', color='blue', label=f'Subject {subj_id} (observed)')
            param_text = "fit failed"
    else:
        plt.plot(xv_ind, yv_ind, 'o-', color='blue', label=f'Subject {subj_id} (observed)')
        param_text = "fit failed (insufficient data)"
    
    # Plot individual observed points
    plt.scatter(xv_ind, yv_ind, color='blue', s=50, edgecolor='black', zorder=5, label='_nolegend_')
    
    # ---- Group average curve (gray with shaded SEM) ----
    if group_fit_success:
        plt.plot(dense_x, y_smooth_group, color='gray', linewidth=2, label='Group average (exponential fit)')
        plt.fill_between(dense_x, y_smooth_group - sem_smooth, y_smooth_group + sem_smooth,
                         alpha=0.2, color='gray', label='Group SEM')
    else:
        # fallback: plot observed group means with points
        plt.plot(xv_group, yv_group, 'o-', color='gray', label='Group average (observed)')
    
    # Plot group observed means as markers (optional)
    plt.scatter(xv_group, yv_group, color='gray', s=40, edgecolor='black', zorder=4, label='_nolegend_')
    
    # Chance line
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance (50%)')
    
    # Labels, title, legend
    group_label = group_map.get(subj_id, 'unknown')
    plt.xlabel('Trial number', fontsize=12)
    plt.ylabel('Proportion choosing set-winner', fontsize=12)
    plt.ylim(0, 1)
    plt.xlim(0.9, 18.1)
    plt.xticks(range(1, trials_per_block+1))
    plt.title(f'Subject {subj_id} – {group_label.capitalize()} learner\nIndividual vs. group average (n={len(all_subjects)})')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # Add annotation for individual fit parameters (if available)
    if param_text:
        plt.annotate(param_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    outfile = OUTPUT_DIR / f'subject_{subj_id}_vs_average.png'
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved: {outfile}")

print("All individual vs average plots generated.")
