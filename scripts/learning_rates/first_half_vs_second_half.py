#!/usr/bin/env python3
"""
This script compares learning performance during the first half (blocks 1‑5) and the second half (blocks 6‑10) of the task. It produces ten output graphs:

1. boxplot_first_vs_second.png – Boxplot of overall proportions (paired t‑test printed to console).
2. group_average_first_half.png – Group average exponential curve for blocks 1‑5.
3. group_average_second_half.png – Group average exponential curve for blocks 6‑10.
4. overlay_group_average_curves.png – Overlay of the two group average curves (first vs second half).
5. all_individuals_first_half.png – All individuals’ curves (points + exponential fits) for blocks 1‑5.
6. all_individuals_second_half.png – Same for blocks 6‑10.
7. good_vs_bad_first_half.png – Overlay of good vs bad learners during blocks 1‑5.
8. good_vs_bad_second_half.png – Overlay of good vs bad learners during blocks 6‑10.
9. good_first_vs_second.png – For good learners only: first half vs second half.
10. bad_first_vs_second.png – For bad learners only: first half vs second half.

All plots use exponential fits with shaded SEM bands. Outputs are saved in outputs/half_comparison/.
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
from scipy.stats import sem, ttest_rel
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import sys 

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR

# ---------- EXPONENTIAL FUNCTION ----------
def exp_learning(t, a, b, k):
    return a - b * np.exp(-k * t)

# ---------- PATHS ----------
RESULTS_DIR = PLEARNING_DIR
OUTPUT_BASE = FIGURES_DIR / "half_comparison"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

plearning_session = 1
trials_per_block = 18
first_blocks = [1,2,3,4,5]
second_blocks = [6,7,8,9,10]

# ---------- LOAD CLASSIFICATION (GOOD/BAD) ----------
learning_rates_csv = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
if learning_rates_csv.exists():
    lr_df = pd.read_csv(learning_rates_csv, dtype={'subjectId': str})
    lr_df = lr_df[lr_df['plearning_num'] == plearning_session].copy()
    lr_df = lr_df.dropna(subset=['learning_rate_k'])
    median_k = lr_df['learning_rate_k'].median()
    lr_df['group'] = np.where(lr_df['learning_rate_k'] >= median_k, 'good', 'bad')
    lr_df['subjectId'] = lr_df['subjectId'].str.zfill(3)
    group_map = lr_df.set_index('subjectId')['group'].to_dict()
else:
    # Fallback: compute overall proportion correct for each subject and split
    group_map = {}
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
        if 'learned' not in df.columns:
            continue
        overall = df['learned'].mean()
        group_map[subj_id] = overall
    median_val = np.median(list(group_map.values()))
    group_map = {k: ('good' if v >= median_val else 'bad') for k, v in group_map.items()}

# ---------- LOAD DATA AND COMPUTE PROPORTIONS PER HALF ----------
all_subjects = []   # each: {'subject': id, 'first': list(18), 'second': list(18), 'group': str}
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
        vals_first = df[(df['trialNumber'] == t) & (df['blockNumber'].isin(first_blocks))]['learned'].values
        first_props.append(np.mean(vals_first) if len(vals_first) > 0 else np.nan)
        vals_second = df[(df['trialNumber'] == t) & (df['blockNumber'].isin(second_blocks))]['learned'].values
        second_props.append(np.mean(vals_second) if len(vals_second) > 0 else np.nan)
    
    all_subjects.append({
        'subject': str(subj_id),
        'first': first_props,
        'second': second_props,
        'group': group_map.get(str(subj_id), 'unknown')
    })

if not all_subjects:
    raise ValueError("No valid subject data found")

# ---------- OVERALL COMPARISON (t-test, boxplot) ----------
overall_first = [np.nanmean(s['first']) for s in all_subjects]
overall_second = [np.nanmean(s['second']) for s in all_subjects]
t_stat, p_val = ttest_rel(overall_first, overall_second)
print(f"Paired t-test: t({len(overall_first)-1}) = {t_stat:.3f}, p = {p_val:.4f}")
print(f"Mean first half: {np.mean(overall_first):.3f} ± {np.std(overall_first):.3f}")
print(f"Mean second half: {np.mean(overall_second):.3f} ± {np.std(overall_second):.3f}")

# Boxplot
plt.figure(figsize=(6,5))
data_to_plot = [overall_first, overall_second]
bp = plt.boxplot(data_to_plot, patch_artist=True,
                 boxprops=dict(facecolor='lightblue'),
                 medianprops=dict(color='red', linewidth=2))
plt.xticks([1,2], ['Blocks 1-5', 'Blocks 6-10'])
plt.ylabel('Proportion choosing set-winner')
plt.title(f'First vs second half – Session {plearning_session}')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_BASE / 'boxplot_first_vs_second.png', dpi=150)
plt.close()
print("Boxplot saved.")

# ---------- FUNCTION TO PLOT GROUP AVERAGE WITH EXPONENTIAL FIT ----------
def plot_group_average(half_name, half_data_list, output_filename, color='blue'):
    arr = np.array(half_data_list)
    mean = np.nanmean(arr, axis=0)
    sem_vals = sem(arr, axis=0, nan_policy='omit')
    x = np.arange(1, trials_per_block+1)
    valid = ~np.isnan(mean)
    xv, yv = x[valid], mean[valid]
    
    plt.figure(figsize=(10,6))
    try:
        p0 = [np.max(yv), np.max(yv)-np.min(yv), 0.3]
        bounds = ([0.5,0,0], [1,1,2])
        popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
        a_fit, b_fit, k_fit = popt
        dense_x = np.linspace(1, 18, 200)
        y_smooth = exp_learning(dense_x, a_fit, b_fit, k_fit)
        plt.plot(dense_x, y_smooth, color=color, linewidth=2, label='Exponential fit')
        sem_valid = ~np.isnan(sem_vals)
        if np.sum(sem_valid) >= 2:
            interp_sem = interp1d(x[sem_valid], sem_vals[sem_valid], kind='linear', fill_value='extrapolate')
            sem_smooth = interp_sem(dense_x)
        else:
            sem_smooth = np.full_like(dense_x, np.nanmean(sem_vals[valid]))
        plt.fill_between(dense_x, y_smooth - sem_smooth, y_smooth + sem_smooth, alpha=0.2, color=color)
    except Exception as e:
        print(f"Exponential fit failed for {half_name}: {e}")
        plt.plot(xv, yv, 'o-', color=color, label='Observed mean')
    
    plt.scatter(xv, yv, color=color, s=50, edgecolor='black', zorder=5)
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel('Trial number')
    plt.ylabel('Proportion choosing set-winner')
    plt.ylim(0,1)
    plt.xlim(0.9, 18.1)
    plt.xticks(range(1, trials_per_block+1))
    plt.title(f'Group average – {half_name} (n={len(half_data_list)})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / output_filename, dpi=150)
    plt.close()
    print(f"Saved {output_filename}")

# ---------- FUNCTION TO PLOT ALL INDIVIDUALS ----------
def plot_all_individuals(half_name, half_data_list, output_filename):
    plt.figure(figsize=(12,6))
    x = np.arange(1, trials_per_block+1)
    dense_x = np.linspace(1, 18, 200)
    for subj_data in half_data_list:
        y = np.array(subj_data)
        valid = ~np.isnan(y)
        xv, yv = x[valid], y[valid]
        if len(xv) >= 4:
            try:
                p0 = [np.max(yv), np.max(yv)-np.min(yv), 0.3]
                bounds = ([0.5,0,0], [1,1,2])
                popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
                y_smooth = exp_learning(dense_x, *popt)
                plt.plot(dense_x, y_smooth, linewidth=1, alpha=0.5, label='_nolegend_')
            except:
                pass
        plt.plot(x, y, 'o', markersize=3, alpha=0.5, label='_nolegend_')
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel('Trial number')
    plt.ylabel('Proportion choosing set-winner')
    plt.ylim(0,1)
    plt.xlim(0.9, 18.1)
    plt.xticks(range(1, trials_per_block+1))
    plt.title(f'All individuals – {half_name} (n={len(half_data_list)})')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / output_filename, dpi=150)
    plt.close()
    print(f"Saved {output_filename}")

# ---------- FUNCTION TO PLOT TWO CURVES (OVERLAY) ----------
def plot_overlay_two_curves(data1, label1, color1, data2, label2, color2, title, outfile, xlabel='Trial number', ylabel='Proportion choosing set-winner'):
    """data1, data2: list of subject data (each is a list of 18 proportions)"""
    arr1 = np.array(data1)
    arr2 = np.array(data2)
    mean1 = np.nanmean(arr1, axis=0)
    mean2 = np.nanmean(arr2, axis=0)
    sem1 = sem(arr1, axis=0, nan_policy='omit')
    sem2 = sem(arr2, axis=0, nan_policy='omit')
    x = np.arange(1, trials_per_block+1)
    dense_x = np.linspace(1, 18, 200)
    
    plt.figure(figsize=(10,6))
    # Fit and plot first curve
    valid1 = ~np.isnan(mean1)
    xv1, yv1 = x[valid1], mean1[valid1]
    try:
        p0 = [np.max(yv1), np.max(yv1)-np.min(yv1), 0.3]
        popt, _ = curve_fit(exp_learning, xv1, yv1, p0=p0, bounds=([0.5,0,0],[1,1,2]), maxfev=5000)
        y_smooth = exp_learning(dense_x, *popt)
        plt.plot(dense_x, y_smooth, color=color1, linewidth=2, label=f'{label1} (fit)')
        sem_valid1 = ~np.isnan(sem1)
        if np.sum(sem_valid1) >= 2:
            interp_sem = interp1d(x[sem_valid1], sem1[sem_valid1], kind='linear', fill_value='extrapolate')
            sem_smooth = interp_sem(dense_x)
        else:
            sem_smooth = np.full_like(dense_x, np.nanmean(sem1[valid1]))
        plt.fill_between(dense_x, y_smooth - sem_smooth, y_smooth + sem_smooth, alpha=0.2, color=color1)
    except Exception as e:
        print(f"Fit failed for {label1}: {e}")
        plt.plot(xv1, yv1, 'o-', color=color1, label=label1)
    
    # Fit and plot second curve
    valid2 = ~np.isnan(mean2)
    xv2, yv2 = x[valid2], mean2[valid2]
    try:
        p0 = [np.max(yv2), np.max(yv2)-np.min(yv2), 0.3]
        popt, _ = curve_fit(exp_learning, xv2, yv2, p0=p0, bounds=([0.5,0,0],[1,1,2]), maxfev=5000)
        y_smooth = exp_learning(dense_x, *popt)
        plt.plot(dense_x, y_smooth, color=color2, linewidth=2, label=f'{label2} (fit)')
        sem_valid2 = ~np.isnan(sem2)
        if np.sum(sem_valid2) >= 2:
            interp_sem = interp1d(x[sem_valid2], sem2[sem_valid2], kind='linear', fill_value='extrapolate')
            sem_smooth = interp_sem(dense_x)
        else:
            sem_smooth = np.full_like(dense_x, np.nanmean(sem2[valid2]))
        plt.fill_between(dense_x, y_smooth - sem_smooth, y_smooth + sem_smooth, alpha=0.2, color=color2)
    except Exception as e:
        print(f"Fit failed for {label2}: {e}")
        plt.plot(xv2, yv2, 'o-', color=color2, label=label2)
    
    # Plot observed means as markers (optional)
    plt.scatter(xv1, yv1, color=color1, s=40, edgecolor='black', zorder=5)
    plt.scatter(xv2, yv2, color=color2, s=40, edgecolor='black', zorder=5)
    
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0,1)
    plt.xlim(0.9, 18.1)
    plt.xticks(range(1, trials_per_block+1))
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / outfile, dpi=150)
    plt.close()
    print(f"Saved {outfile}")

# ---------- PREPARE DATA LISTS ----------
first_half_data = [s['first'] for s in all_subjects]
second_half_data = [s['second'] for s in all_subjects]

# Split by group
good_first = [s['first'] for s in all_subjects if s['group'] == 'good']
good_second = [s['second'] for s in all_subjects if s['group'] == 'good']
bad_first = [s['first'] for s in all_subjects if s['group'] == 'bad']
bad_second = [s['second'] for s in all_subjects if s['group'] == 'bad']

# ---------- GENERATE ALL 10 PLOTS ----------

# 1. Boxplot (already done above)
# 2. Group average first half
plot_group_average("Blocks 1-5", first_half_data, "group_average_first_half.png", color='blue')
# 3. Group average second half
plot_group_average("Blocks 6-10", second_half_data, "group_average_second_half.png", color='red')
# 4. Overlay of both group averages (first half vs second half)
plot_overlay_two_curves(first_half_data, "Blocks 1-5", 'blue',
                        second_half_data, "Blocks 6-10", 'red',
                        "First half vs. Second half (all subjects)",
                        "overlay_group_average_curves.png")
# 5. All individuals first half
plot_all_individuals("Blocks 1-5", first_half_data, "all_individuals_first_half.png")
# 6. All individuals second half
plot_all_individuals("Blocks 6-10", second_half_data, "all_individuals_second_half.png")
# 7. Good vs bad learners – first half
plot_overlay_two_curves(good_first, "Good learners (blocks 1-5)", 'green',
                        bad_first, "Bad learners (blocks 1-5)", 'red',
                        "Good vs. Bad learners – Blocks 1-5",
                        "good_vs_bad_first_half.png")
# 8. Good vs bad learners – second half
plot_overlay_two_curves(good_second, "Good learners (blocks 6-10)", 'green',
                        bad_second, "Bad learners (blocks 6-10)", 'red',
                        "Good vs. Bad learners – Blocks 6-10",
                        "good_vs_bad_second_half.png")
# 9. Good learners: first half vs second half
plot_overlay_two_curves(good_first, "Good learners – Blocks 1-5", 'blue',
                        good_second, "Good learners – Blocks 6-10", 'red',
                        "Good learners: First half vs. Second half",
                        "good_first_vs_second.png")
# 10. Bad learners: first half vs second half
plot_overlay_two_curves(bad_first, "Bad learners – Blocks 1-5", 'blue',
                        bad_second, "Bad learners – Blocks 6-10", 'red',
                        "Bad learners: First half vs. Second half",
                        "bad_first_vs_second.png")

print("All 10 plots have been generated.")
