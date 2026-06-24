#!/usr/bin/env python3
"""
For each subject individually, this script creates a plot comparing their learning curves from the first half (blocks 1‑5) and the second half (blocks 6‑10) of the task.
- Both halves are overlaid on the same axes (trial numbers 1‑18).
- Exponential curves are fitted to each half’s data (if enough valid points exist).
- Observed proportions are shown as points.
- The plot includes a chance line at 0.5 and a legend.
Outputs: one PNG per subject in outputs/individual_half_comparison/.
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
from scipy.interpolate import interp1d
import sys 

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from project_paths import PLEARNING_DIR, FIGURES_DIR

# ---------- EXPONENTIAL FUNCTION ----------
def exp_learning(t, a, b, k):
    return a - b * np.exp(-k * t)

# ---------- PORTABLE PATHS ----------
RESULTS_DIR = PLEARNING_DIR
OUTPUT_DIR = FIGURES_DIR / "individual_half_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plearning_session = 1
trials_per_block = 18
first_blocks = [1, 2, 3, 4, 5]
second_blocks = [6, 7, 8, 9, 10]

# ---------- LOAD DATA FOR EACH SUBJECT ----------
all_subjects = []   # each: {'subject': id, 'first': list(18), 'second': list(18)}
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
    if 'learned' not in df.columns or 'blockNumber' not in df.columns or 'trialNumber' not in df.columns:
        continue
    df['learned'] = df['learned'].astype(int)

    first_props = []
    second_props = []
    for t in range(1, trials_per_block+1):
        # first half
        vals_first = df[(df['trialNumber'] == t) & (df['blockNumber'].isin(first_blocks))]['learned'].values
        first_props.append(np.mean(vals_first) if len(vals_first) > 0 else np.nan)
        # second half
        vals_second = df[(df['trialNumber'] == t) & (df['blockNumber'].isin(second_blocks))]['learned'].values
        second_props.append(np.mean(vals_second) if len(vals_second) > 0 else np.nan)

    all_subjects.append({
        'subject': str(subj_id),
        'first': first_props,
        'second': second_props
    })

if not all_subjects:
    raise ValueError("No valid subject data found")

# ---------- GENERATE ONE PLOT PER SUBJECT ----------
x = np.arange(1, trials_per_block+1)
dense_x = np.linspace(1, 18, 200)

for subj in all_subjects:
    subj_id = subj['subject']
    y_first = np.array(subj['first'])
    y_second = np.array(subj['second'])
    valid_first = ~np.isnan(y_first)
    valid_second = ~np.isnan(y_second)

    plt.figure(figsize=(10, 6))

    # ---- First half (blocks 1-5) in blue ----
    if np.sum(valid_first) >= 4:
        try:
            p0 = [np.max(y_first[valid_first]), np.max(y_first[valid_first]) - np.min(y_first[valid_first]), 0.3]
            bounds = ([0.5, 0, 0], [1, 1, 2])
            popt, _ = curve_fit(exp_learning, x[valid_first], y_first[valid_first],
                                p0=p0, bounds=bounds, maxfev=5000)
            y_smooth = exp_learning(dense_x, *popt)
            plt.plot(dense_x, y_smooth, color='blue', linewidth=2,
                     label='Blocks 1-5 (exponential fit)')
            # Shaded band: here we use the inter‑trial variability? For a single subject,
            # we cannot compute SEM across subjects; we simply show the fit without error band.
            # Alternatively, we could show the raw points as markers.
        except Exception as e:
            print(f"Subject {subj_id} first half fit failed: {e}")
            plt.plot(x[valid_first], y_first[valid_first], 'o-', color='blue', label='Blocks 1-5 (observed)')
    else:
        plt.plot(x[valid_first], y_first[valid_first], 'o-', color='blue', label='Blocks 1-5 (observed)')
    # Plot observed points
    plt.scatter(x[valid_first], y_first[valid_first], color='blue', s=40, edgecolor='black', zorder=5)

    # ---- Second half (blocks 6-10) in red ----
    if np.sum(valid_second) >= 4:
        try:
            p0 = [np.max(y_second[valid_second]), np.max(y_second[valid_second]) - np.min(y_second[valid_second]), 0.3]
            popt, _ = curve_fit(exp_learning, x[valid_second], y_second[valid_second],
                                p0=p0, bounds=bounds, maxfev=5000)
            y_smooth = exp_learning(dense_x, *popt)
            plt.plot(dense_x, y_smooth, color='red', linewidth=2,
                     label='Blocks 6-10 (exponential fit)')
        except Exception as e:
            print(f"Subject {subj_id} second half fit failed: {e}")
            plt.plot(x[valid_second], y_second[valid_second], 'o-', color='red', label='Blocks 6-10 (observed)')
    else:
        plt.plot(x[valid_second], y_second[valid_second], 'o-', color='red', label='Blocks 6-10 (observed)')
    plt.scatter(x[valid_second], y_second[valid_second], color='red', s=40, edgecolor='black', zorder=5)

    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel('Trial number', fontsize=12)
    plt.ylabel('Proportion choosing set-winner', fontsize=12)
    plt.ylim(0, 1)
    plt.xlim(0.9, 18.1)
    plt.xticks(range(1, trials_per_block+1))
    plt.title(f'Subject {subj_id} – Comparison of first half vs. second half (Session {plearning_session})')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    outfile = OUTPUT_DIR / f'subject_{subj_id}_first_vs_second_half.png'
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved plot: {outfile}")

print("All individual half‑comparison plots generated.")
