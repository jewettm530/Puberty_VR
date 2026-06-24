#!/usr/bin/env python3
"""
Equivalent to good_learners_only.py but for subjects classified as ‘bad’ learners.
- Plots a single figure with all bad learners’ observed proportions and their individual exponential fits.
- Uses the same classification method (median split of learning rate k or fallback overall accuracy).
Output: PNG in outputs/bad_learners/.
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
from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR

def exp_learning(t, a, b, k):
    return a - b * np.exp(-k * t)

# ---------- CORRECTED PATHS ----------
RESULTS_DIR = PLEARNING_DIR
OUTPUT_DIR = FIGURES_DIR / "bad_learners"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plearning_session = 1
trials_per_block = 18

# Load classification
learning_rates_csv = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
if learning_rates_csv.exists():
    lr_df = pd.read_csv(learning_rates_csv, dtype={'subjectId': str})
    lr_df = lr_df[lr_df['plearning_num'] == plearning_session].copy()
    lr_df = lr_df.dropna(subset=['learning_rate_k'])
    lr_df['subjectId'] = lr_df['subjectId'].str.zfill(3)
    lr_df['metric'] = lr_df['learning_rate_k']
else:
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
bad_ids = lr_df[lr_df['group']=='bad']['subjectId'].astype(str).tolist()

# Load trial data for bad subjects
all_bad = []
for csv_file in RESULTS_DIR.rglob("*_plearning_*.csv"):
    parts = csv_file.stem.split("_")
    if len(parts) < 3: continue
    subj_id = parts[0]
    if subj_id not in bad_ids: continue
    try: p_num = int(parts[2])
    except: continue
    if p_num != plearning_session: continue
    df = pd.read_csv(csv_file)
    if 'learned' not in df.columns: continue
    df['learned'] = df['learned'].astype(int)
    trial_props = [df[df['trialNumber'] == t]['learned'].mean() for t in range(1, trials_per_block+1)]
    all_bad.append({'subject': subj_id, 'proportions': trial_props})

# Plot
plt.figure(figsize=(12,6))
x = np.arange(1, trials_per_block+1)
dense_x = np.linspace(1,18,200)
for subj in all_bad:
    y = np.array(subj['proportions'])
    valid = ~np.isnan(y)
    xv, yv = x[valid], y[valid]
    if len(xv) >= 4:
        try:
            p0 = [np.max(yv), np.max(yv)-np.min(yv), 0.3]
            bounds = ([0.5,0,0], [1,1,2])
            popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
            y_smooth = exp_learning(dense_x, *popt)
            plt.plot(dense_x, y_smooth, linewidth=1, alpha=0.6, label='_nolegend_')
        except:
            pass
    plt.plot(x, y, 'o', markersize=3, alpha=0.5, label=subj['subject'])

plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
plt.xlabel('Trial number')
plt.ylabel('Proportion choosing set-winner')
plt.ylim(0,1)
plt.xlim(1, 18.1)
plt.xticks(range(1, trials_per_block+1))
plt.title(f'Bad learners – Session {plearning_session} (n={len(all_bad)})')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'bad_learners_curves_sess{plearning_session}.png', dpi=150)
plt.close()
