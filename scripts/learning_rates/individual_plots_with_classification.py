#!/usr/bin/env python3
"""
This script generates one individual learning curve plot per subject, with:
- Observed proportion correct (points) and an exponential fit (line).
- Classification of the subject as ‘good’ or ‘bad’ learner based on a median split of their fitted learning rate (k) from a pre‑computed file (learning_rates.csv) or, if unavailable, from overall mean accuracy.
- The plot includes the fitted parameters (a, b, k) as an annotation.
- A CSV file is also created that contains, for each subject and trial, the proportion correct and the group label (good/bad).
Outputs: individual PNGs and a CSV file in outputs/individual_plots/.
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
import sys 

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR, TABLES_DIR

def exp_learning(t, a, b, k):
    return a - b * np.exp(-k * t)

# ---------- CORRECTED PATHS ----------
RESULTS_DIR = PLEARNING_DIR
OUTPUT_DIR = FIGURES_DIR / "individual_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_OUTPUT_DIR = TABLES_DIR / "individual_plots"
TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plearning_session = 1
trials_per_block = 18

# Load or compute learning metric for classification
learning_rates_csv = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
if learning_rates_csv.exists():
    lr_df = pd.read_csv(learning_rates_csv, dtype={'subjectId': str})
    lr_df = lr_df[lr_df['plearning_num'] == plearning_session].copy()
    lr_df = lr_df.dropna(subset=['learning_rate_k'])
    lr_df['subjectId'] = lr_df['subjectId'].str.zfill(3)
    lr_df['metric'] = lr_df['learning_rate_k']
else:
    # Fallback: compute overall proportion correct from data
    lr_df = []
    for csv_file in RESULTS_DIR.rglob("*_plearning_*.csv"):
        parts = csv_file.stem.split("_")
        if len(parts) < 3: continue
        subj_id = parts[0]
        try: p_num = int(parts[2])
        except: continue
        if p_num != plearning_session: continue
        df = pd.read_csv(csv_file)
        df['learned'] = df['learned'].astype(int)
        overall = df['learned'].mean()
        lr_df.append({'subjectId': subj_id, 'metric': overall})
    lr_df = pd.DataFrame(lr_df)
median_val = lr_df['metric'].median()
lr_df['group'] = np.where(lr_df['metric'] >= median_val, 'good', 'bad')
print(f"Median metric = {median_val:.4f}")
print(f"Good learners: {sum(lr_df['group']=='good')}, Bad: {sum(lr_df['group']=='bad')}")

# Load trial data
all_subjects = []
for csv_file in RESULTS_DIR.rglob("*_plearning_*.csv"):
    parts = csv_file.stem.split("_")
    if len(parts) < 3: continue
    subj_id = parts[0]
    try: p_num = int(parts[2])
    except: continue
    if p_num != plearning_session: continue
    df = pd.read_csv(csv_file)
    if 'learned' not in df.columns: continue
    df['learned'] = df['learned'].astype(int)
    trial_props = [df[df['trialNumber'] == t]['learned'].mean() for t in range(1, trials_per_block+1)]
    all_subjects.append({'subject': subj_id, 'proportions': trial_props})

# Merge with group labels
group_map = lr_df.set_index('subjectId')['group'].to_dict()
classification_rows = []
x = np.arange(1, trials_per_block+1)
dense_x = np.linspace(1, 18, 200)

for subj in all_subjects:
    subj_id = subj['subject']
    y = np.array(subj['proportions'])
    valid = ~np.isnan(y)
    xv, yv = x[valid], y[valid]
    group = group_map.get(subj_id, 'unknown')
    # Save classification row
    for t, p in enumerate(subj['proportions'], start=1):
        classification_rows.append({'subject': subj_id, 'trial': t, 'proportion_correct': p, 'group': group})
    # Plot individual
    plt.figure(figsize=(7,5))
    plt.plot(x, y, 'o', color='blue', label='Observed')
    if len(xv) >= 4:
        try:
            p0 = [np.max(yv), np.max(yv)-np.min(yv), 0.3]
            bounds = ([0.5,0,0], [1,1,2])
            popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
            y_smooth = exp_learning(dense_x, *popt)
            plt.plot(dense_x, y_smooth, 'r-', label='Exponential fit')
            plt.annotate(f"a={popt[0]:.2f}, b={popt[1]:.2f}, k={popt[2]:.2f}",
                         xy=(0.05,0.95), xycoords='axes fraction', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.ylim(0,1)
    plt.xlim(1, 18.1)
    plt.xticks(range(1, trials_per_block+1))
    plt.xlabel('Trial number')
    plt.ylabel('Proportion choosing set-winner')
    plt.title(f'Subject {subj_id} – {group.capitalize()} learner')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'subject_{subj_id}_session{plearning_session}.png', dpi=150)
    plt.close()

# Save classification CSV
df_class = pd.DataFrame(classification_rows)
df_class.to_csv(TABLE_OUTPUT_DIR / f'individual_data_with_classification_sess{plearning_session}.csv', index=False)
print(f"All individual plots and CSV saved in {OUTPUT_DIR}")
