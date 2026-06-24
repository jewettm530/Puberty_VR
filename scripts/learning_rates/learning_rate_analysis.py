#!/usr/bin/env python3
"""
Compute median learning rates (proportion of set‑winner choices) per trial and per session,
generate a CSV summary, and produce graphs for:
- Individual subjects: session 1 vs session 2, individual vs group average
- Group average curves with SEM bands
- Subject vs group average with standard deviation error bars
- Good vs bad learner comparisons (median split of learning rate k)
- Scatter plots of group average proportion correct split by winner type (face/house)

Input: subject CSV files (name pattern: *_{id}_plearning_{session}.csv) from project_paths.PLEARNING_DIR
Output: organised subfolders under outputs/learning_rates/figures/learning_rate_analysis/
        and outputs/learning_rates/tables/learning_rate_analysis/
Dependencies: pandas, numpy, matplotlib, scipy
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
from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR, TABLES_DIR

# ---------- EXPONENTIAL FUNCTION ----------
def exp_learning(t, a, b, k):
    return a - b * np.exp(-k * t)

# ---------- PATHS ----------
RESULTS_DIR = PLEARNING_DIR
BASE_OUTPUT_DIR = FIGURES_DIR / "learning_rate_analysis"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_OUTPUT_DIR = TABLES_DIR / "learning_rate_analysis"
TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories for different graph types
SUB_DIRS = {
    "s1_vs_s2": BASE_OUTPUT_DIR / "individual_s1_vs_s2",
    "with_group_avg": BASE_OUTPUT_DIR / "individual_with_group_avg",
    "group_avg": BASE_OUTPUT_DIR / "group_average",
    "vs_group_sd": BASE_OUTPUT_DIR / "subject_vs_group_sd",
}
for subdir in SUB_DIRS.values():
    subdir.mkdir(exist_ok=True)

trials_per_block = 18
sessions = [1, 2]

# ---------- LOAD ALL SUBJECT DATA ----------
all_data = {}
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
    props = []
    for t in range(1, trials_per_block+1):
        vals = df[df['trialNumber'] == t]['learned'].values
        props.append(np.mean(vals) if len(vals) > 0 else np.nan)
    all_data.setdefault(subj_id, {})[sess] = props

# Keep only subjects with both sessions
complete_subjects = [subj for subj in all_data if 1 in all_data[subj] and 2 in all_data[subj]]
missing = set(all_data.keys()) - set(complete_subjects)
if missing:
    print(f"Warning: The following subjects are missing one session and will be excluded: {missing}")

if not complete_subjects:
    raise ValueError("No subjects have both session 1 and session 2 data.")

all_data = {subj: all_data[subj] for subj in complete_subjects}
subjects = sorted(all_data.keys())
n_subj = len(subjects)

# --- Classify subjects as good/bad using median split of learning_rate_k from learning_rates.csv ---
learning_rates_csv = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
if learning_rates_csv.exists():
    lr_df = pd.read_csv(learning_rates_csv, dtype={'subjectId': str})
    lr_df = lr_df[lr_df['plearning_num'] == 1].copy()  # use session 1 for classification
    lr_df = lr_df.dropna(subset=['learning_rate_k'])
    median_k = lr_df['learning_rate_k'].median()
    lr_df['group'] = np.where(lr_df['learning_rate_k'] >= median_k, 'good', 'bad')
    lr_df['subjectId'] = lr_df['subjectId'].str.zfill(3)
    group_map = lr_df.set_index('subjectId')['group'].to_dict()
else:
    # Fallback: use overall mean accuracy from session 1
    print("learning_rates.csv not found. Using overall mean accuracy for classification.")
    overall_means = {}
    for subj in subjects:
        props = np.array(all_data[subj][1])  # session 1
        overall_means[subj] = np.nanmean(props)
    median_val = np.median(list(overall_means.values()))
    group_map = {s: ('good' if overall_means[s] >= median_val else 'bad') for s in subjects}

good_subjects = [s for s in subjects if group_map.get(s) == 'good']
bad_subjects = [s for s in subjects if group_map.get(s) == 'bad']
print(f"Good learners (n={len(good_subjects)}): {good_subjects}")
print(f"Bad learners (n={len(bad_subjects)}): {bad_subjects}")

# ========== CSV SUMMARY WITH AUC ==========
print("\nBuilding CSV summary with group averages and AUC...")

x_trials = np.arange(1, trials_per_block+1)

def get_k_from_curve(x, y):
    valid = ~np.isnan(y)
    xv, yv = x[valid], y[valid]
    if len(xv) < 4:
        return np.nan
    try:
        p0 = [np.max(yv), np.max(yv)-np.min(yv), 0.3]
        bounds = ([0.5,0,0], [1,1,2])
        popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
        return popt[2]
    except:
        return np.nan

# 1. Build rows for individual subjects
rows = []
subject_aucs = {subj: {} for subj in subjects}  # store for later

for subj in subjects:
    row = {'subject': subj}
    for sess in sessions:
        props = all_data[subj][sess]
        # Store per-trial proportions
        for t, p in enumerate(props, start=1):
            row[f'sess{sess}_t{t}'] = p
        # Median
        row[f'median_sess{sess}'] = np.nanmedian(props)
        # AUC = sum of proportions across 18 trials
        auc = np.nansum(props)
        row[f'auc_sess{sess}'] = auc
        subject_aucs[subj][sess] = auc
    row['diff_s2_minus_s1'] = row['median_sess2'] - row['median_sess1']
    row['k_sess1'] = np.nan
    row['k_sess2'] = np.nan
    rows.append(row)

df_subj = pd.DataFrame(rows)

# 2. Compute overall average row (all subjects)
avg_row = {'subject': 'AVERAGE'}
for col in df_subj.columns:
    if col == 'subject':
        continue
    elif col.startswith('sess') and '_t' in col:
        avg_row[col] = df_subj[col].mean()
    elif col == 'median_sess1':
        avg_row[col] = df_subj[col].mean()
    elif col == 'median_sess2':
        avg_row[col] = df_subj[col].mean()
    elif col == 'diff_s2_minus_s1':
        avg_row[col] = df_subj[col].mean()
    elif col == 'auc_sess1':
        avg_row[col] = df_subj[col].mean()   # mean of individual AUCs
    elif col == 'auc_sess2':
        avg_row[col] = df_subj[col].mean()
    elif col == 'k_sess1':
        # Compute k from overall average curve (as before)
        overall_curve_s1 = [df_subj[f'sess1_t{t}'].mean() for t in range(1, trials_per_block+1)]
        avg_row[col] = get_k_from_curve(x_trials, np.array(overall_curve_s1))
    elif col == 'k_sess2':
        overall_curve_s2 = [df_subj[f'sess2_t{t}'].mean() for t in range(1, trials_per_block+1)]
        avg_row[col] = get_k_from_curve(x_trials, np.array(overall_curve_s2))
    else:
        avg_row[col] = np.nan

# 3. Build rows for good and bad learners
def build_group_row(group_name, subject_list):
    if not subject_list:
        return None
    row = {'subject': group_name}
    for sess in sessions:
        # Average per trial across subjects
        group_curve = np.nanmean([all_data[s][sess] for s in subject_list], axis=0)
        for t, p in enumerate(group_curve, start=1):
            row[f'sess{sess}_t{t}'] = p
        row[f'median_sess{sess}'] = np.nanmedian(group_curve)
        # Average AUC of subjects in this group
        aucs = [subject_aucs[s][sess] for s in subject_list]
        row[f'auc_sess{sess}'] = np.nanmean(aucs) if aucs else np.nan
    row['diff_s2_minus_s1'] = row['median_sess2'] - row['median_sess1']
    # Compute k from group's average curve (not from individual ks)
    group_curve_s1 = [row[f'sess1_t{t}'] for t in range(1, trials_per_block+1)]
    row['k_sess1'] = get_k_from_curve(x_trials, np.array(group_curve_s1))
    group_curve_s2 = [row[f'sess2_t{t}'] for t in range(1, trials_per_block+1)]
    row['k_sess2'] = get_k_from_curve(x_trials, np.array(group_curve_s2))
    return row

good_row = build_group_row('good_learners', good_subjects)
bad_row = build_group_row('bad_learners', bad_subjects)

# 4. Concatenate all rows
rows_to_add = [pd.DataFrame([avg_row])]
if good_row is not None:
    rows_to_add.append(pd.DataFrame([good_row]))
if bad_row is not None:
    rows_to_add.append(pd.DataFrame([bad_row]))

df_subj = pd.concat([df_subj] + rows_to_add, ignore_index=True)

# 5. Reorder columns: subject, then session1 trials, session2 trials, then medians, AUCs, diff, ks
trial_cols = [f'sess{sess}_t{t}' for sess in sessions for t in range(1, trials_per_block+1)]
other_cols = [f'median_sess{sess}' for sess in sessions] + \
             [f'auc_sess{sess}' for sess in sessions] + \
             ['diff_s2_minus_s1', 'k_sess1', 'k_sess2']
final_columns = ['subject'] + trial_cols + other_cols
df_subj = df_subj[final_columns]

# 6. Save CSV
csv_path = TABLE_OUTPUT_DIR / 'median_learning_rates_summary.csv'
df_subj.to_csv(csv_path, index=False)
print(f"Saved CSV with AUC columns: {csv_path}")

# ---------- HELPER FIT FUNCTION ----------
def fit_exponential(x, y):
    valid = ~np.isnan(y)
    xv, yv = x[valid], y[valid]
    if len(xv) < 4:
        return None, None
    try:
        p0 = [np.max(yv), np.max(yv)-np.min(yv), 0.3]
        bounds = ([0.5,0,0], [1,1,2])
        popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
        return popt, xv
    except:
        return None, None

x_trials = np.arange(1, trials_per_block+1)
dense_x = np.linspace(1, 18, 200)

# ========== GRAPHS: TRIAL 1 vs TRIAL 18 (within session) ==========
T1T18_DIR = BASE_OUTPUT_DIR / "trial1_vs_trial18_graphs"
T1T18_DIR.mkdir(exist_ok=True)

def plot_trial1_vs_trial18(session, groups_dict, group_colors):
    """
    groups_dict: {'All subjects': subject_list, 'Good learners': good_list, 'Bad learners': bad_list}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_positions = {'All subjects': 0, 'Good learners': 1, 'Bad learners': 2}
    width = 0.35  # width of each bar pair
    
    for group_name, subj_list in groups_dict.items():
        if not subj_list:
            continue
        
        # Collect trial1 and trial18 proportions for each subject in this group
        t1_vals = []
        t18_vals = []
        for subj in subj_list:
            t1 = all_data[subj][session][0]
            t18 = all_data[subj][session][-1]
            if not np.isnan(t1) and not np.isnan(t18):
                t1_vals.append(t1)
                t18_vals.append(t18)
        
        if not t1_vals:
            continue
        
        # Means and SEMs
        mean_t1 = np.mean(t1_vals)
        mean_t18 = np.mean(t18_vals)
        sem_t1 = sem(t1_vals)
        sem_t18 = sem(t18_vals)
        
        x = x_positions[group_name]
        # Bar for trial1
        ax.bar(x - width/2, mean_t1, width, yerr=sem_t1, capsize=5,
               color=group_colors[group_name], alpha=0.6, edgecolor='black',
               label=f'{group_name} (Trial 1)' if x == 0 else "")
        # Bar for trial18
        ax.bar(x + width/2, mean_t18, width, yerr=sem_t18, capsize=5,
               color=group_colors[group_name], alpha=0.9, edgecolor='black',
               hatch='//', label=f'{group_name} (Trial 18)' if x == 0 else "")
        
        # Jittered individual points (optional, but adds detail)
        jitter = np.random.normal(0, 0.04, size=len(t1_vals))
        for i in range(len(t1_vals)):
            ax.scatter(x - width/2 + jitter[i], t1_vals[i], color='black', s=20, alpha=0.4, zorder=3)
            ax.scatter(x + width/2 + jitter[i], t18_vals[i], color='black', s=20, alpha=0.4, zorder=3)
            # Draw thin line connecting the pair for each subject
            ax.plot([x - width/2 + jitter[i], x + width/2 + jitter[i]], 
                    [t1_vals[i], t18_vals[i]], color='gray', linewidth=0.5, alpha=0.3)
    
    # Formatting
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['All subjects', 'Good learners', 'Bad learners'])
    ax.set_ylabel('Proportion choosing set-winner')
    ax.set_ylim(0, 1)
    ax.set_title(f'Session {session} – Performance on Trial 1 vs Trial 18')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    outfile = T1T18_DIR / f'session{session}_trial1_vs_trial18.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outfile}")

# Define groups and colours
groups = {
    'All subjects': subjects,
    'Good learners': good_subjects,
    'Bad learners': bad_subjects
}
group_colors = {
    'All subjects': 'steelblue',
    'Good learners': 'green',
    'Bad learners': 'coral'
}

# Generate for session 1 and session 2
for sess in [1, 2]:
    plot_trial1_vs_trial18(sess, groups, group_colors)

print(f"Trial 1 vs Trial 18 graphs saved in {T1T18_DIR}")

# ---------- 1. INDIVIDUAL: SESSION 1 vs SESSION 2 ----------
print("Generating individual graphs (session1 vs session2)...")
for subj in subjects:
    props1 = np.array(all_data[subj][1])
    props2 = np.array(all_data[subj][2])
    plt.figure(figsize=(10,6))
    plt.plot(x_trials, props1, 'o', color='blue', label='Session 1 observed', alpha=0.7)
    popt1, _ = fit_exponential(x_trials, props1)
    if popt1 is not None:
        plt.plot(dense_x, exp_learning(dense_x, *popt1), color='blue', linewidth=2, label='Session 1 fit')
    plt.plot(x_trials, props2, 'o', color='red', label='Session 2 observed', alpha=0.7)
    popt2, _ = fit_exponential(x_trials, props2)
    if popt2 is not None:
        plt.plot(dense_x, exp_learning(dense_x, *popt2), color='red', linewidth=2, label='Session 2 fit')
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel('Trial number')
    plt.ylabel('Proportion choosing set-winner')
    plt.ylim(0,1)
    plt.xlim(0.9,18.1)
    plt.xticks(range(1,19))
    plt.title(f'Subject {subj} – Session 1 vs Session 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    outfile = SUB_DIRS["s1_vs_s2"] / f'subject_{subj}_s1_vs_s2.png'
    plt.savefig(outfile, dpi=150)
    plt.close()

# ---------- 2. INDIVIDUAL + GROUP AVERAGE (combined and per-session) ----------
print("Generating individual vs group graphs (combined and per-session)...")
group_avg = {sess: np.nanmean([all_data[s][sess] for s in subjects], axis=0) for sess in sessions}
group_fits = {}
for sess in sessions:
    popt, _ = fit_exponential(x_trials, group_avg[sess])
    group_fits[sess] = popt

# Colors: group session1 = darkgreen, group session2 = orange
group_colors = {1: 'darkgreen', 2: 'orange'}
subject_colors = {1: 'blue', 2: 'red'}

for subj in subjects:
    props1 = np.array(all_data[subj][1])
    props2 = np.array(all_data[subj][2])
    
    # ----- Combined graph (both sessions) -----
    plt.figure(figsize=(10,6))
    # Subject curves
    plt.plot(x_trials, props1, 'o', color=subject_colors[1], label='Subject session1', alpha=0.7)
    popt1, _ = fit_exponential(x_trials, props1)
    if popt1 is not None:
        plt.plot(dense_x, exp_learning(dense_x, *popt1), color=subject_colors[1], linewidth=1.5, alpha=0.8)
    plt.plot(x_trials, props2, 'o', color=subject_colors[2], label='Subject session2', alpha=0.7)
    popt2, _ = fit_exponential(x_trials, props2)
    if popt2 is not None:
        plt.plot(dense_x, exp_learning(dense_x, *popt2), color=subject_colors[2], linewidth=1.5, alpha=0.8)
    # Group curves (distinct colors)
    for sess in sessions:
        avg = group_avg[sess]
        plt.plot(x_trials, avg, 's', color=group_colors[sess], label=f'Group session{sess} (average)', alpha=0.6, markersize=5)
        if group_fits[sess] is not None:
            y_fit = exp_learning(dense_x, *group_fits[sess])
            plt.plot(dense_x, y_fit, color=group_colors[sess], linewidth=1.5, linestyle='--')
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel('Trial number')
    plt.ylabel('Proportion choosing set-winner')
    plt.ylim(0,1)
    plt.xlim(0.9,18.1)
    plt.xticks(range(1,19))
    plt.title(f'Subject {subj} – individual vs group (both sessions)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    outfile_combined = SUB_DIRS["with_group_avg"] / f'subject_{subj}_with_group_avg.png'
    plt.savefig(outfile_combined, dpi=150)
    plt.close()
    
    # ----- Separate graphs: one per session -----
    for sess in sessions:
        props = np.array(all_data[subj][sess])
        plt.figure(figsize=(10,6))
        # Subject
        plt.plot(x_trials, props, 'o', color=subject_colors[sess], label=f'Subject {subj} session{sess}', alpha=0.7)
        popt_subj, _ = fit_exponential(x_trials, props)
        if popt_subj is not None:
            plt.plot(dense_x, exp_learning(dense_x, *popt_subj), color=subject_colors[sess], linewidth=1.5, linestyle='--')
        # Group
        avg = group_avg[sess]
        plt.plot(x_trials, avg, 's', color=group_colors[sess], label=f'Group session{sess} (average)', alpha=0.6, markersize=5)
        if group_fits[sess] is not None:
            y_fit = exp_learning(dense_x, *group_fits[sess])
            plt.plot(dense_x, y_fit, color=group_colors[sess], linewidth=1.5, linestyle='--')
        plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
        plt.xlabel('Trial number')
        plt.ylabel('Proportion choosing set-winner')
        plt.ylim(0,1)
        plt.xlim(0.9,18.1)
        plt.xticks(range(1,19))
        plt.title(f'Subject {subj} vs Group – Session {sess}')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        outfile_sep = SUB_DIRS["with_group_avg"] / f'subject_{subj}_sess{sess}_vs_group.png'
        plt.savefig(outfile_sep, dpi=150)
        plt.close()

# ---------- 3. GROUP AVERAGE PLOT ----------
print("Generating group average plot...")
plt.figure(figsize=(10,6))
for sess, color in [(1,'blue'), (2,'red')]:
    avg = group_avg[sess]
    sem_vals = sem([all_data[s][sess] for s in subjects], axis=0, nan_policy='omit')
    plt.plot(x_trials, avg, 'o', color=color, label=f'Session {sess} observed')
    popt, _ = fit_exponential(x_trials, avg)
    if popt is not None:
        y_fit = exp_learning(dense_x, *popt)
        plt.plot(dense_x, y_fit, color=color, linewidth=2, label=f'Session {sess} exponential fit')
        sem_valid = ~np.isnan(sem_vals)
        if np.sum(sem_valid) >= 2:
            interp_sem = interp1d(x_trials[sem_valid], sem_vals[sem_valid], kind='linear', fill_value='extrapolate')
            sem_smooth = interp_sem(dense_x)
            plt.fill_between(dense_x, y_fit - sem_smooth, y_fit + sem_smooth, alpha=0.2, color=color)
plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
plt.xlabel('Trial number')
plt.ylabel('Proportion choosing set-winner')
plt.ylim(0,1)
plt.xlim(0.9,18.1)
plt.xticks(range(1,19))
plt.title('Group average learning curves (all subjects with both sessions)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
outfile = SUB_DIRS["group_avg"] / 'group_average_curves.png'
plt.savefig(outfile, dpi=150)
plt.close()

# ---------- 4. PER SUBJECT PER SESSION: with SD error bars ----------
print("Generating subject vs group average with standard deviation error bars...")
group_mean = {sess: np.nanmean([all_data[s][sess] for s in subjects], axis=0) for sess in sessions}
group_std = {sess: np.nanstd([all_data[s][sess] for s in subjects], axis=0) for sess in sessions}

for subj in subjects:
    for sess in sessions:
        props = np.array(all_data[subj][sess])
        plt.figure(figsize=(10,6))
        plt.plot(x_trials, props, 'o', color='blue', label=f'Subject {subj} (session {sess})', alpha=0.7)
        plt.errorbar(x_trials, group_mean[sess], yerr=group_std[sess], fmt='s', color='gray',
                     capsize=3, label=f'Group average ± SD (n={n_subj})', alpha=0.6)
        popt_subj, _ = fit_exponential(x_trials, props)
        if popt_subj is not None:
            y_fit = exp_learning(dense_x, *popt_subj)
            plt.plot(dense_x, y_fit, color='blue', linewidth=1.5, linestyle='--', alpha=0.7)
        popt_group, _ = fit_exponential(x_trials, group_mean[sess])
        if popt_group is not None:
            y_fit = exp_learning(dense_x, *popt_group)
            plt.plot(dense_x, y_fit, color='gray', linewidth=1.5, linestyle='--', alpha=0.7)
        plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
        plt.xlabel('Trial number')
        plt.ylabel('Proportion choosing set-winner')
        plt.ylim(0,1)
        plt.xlim(0.9,18.1)
        plt.xticks(range(1,19))
        plt.title(f'Subject {subj} – Session {sess} vs group average (±SD)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        outfile = SUB_DIRS["vs_group_sd"] / f'subject_{subj}_sess{sess}_vs_group.png'
        plt.savefig(outfile, dpi=150)
        plt.close()

# ========== ADDITIONAL GRAPHS: GOOD/BAD LEARNER COMPARISONS ==========
print("\nGenerating good/bad learner comparison graphs...")

# Helper to plot a group's learning curve for a given session (points + exponential fit with SEM)
def plot_group_curve(ax, subject_list, session, color, label_prefix, show_points=True):
    """Adds group average curve (fit + SEM) and optionally individual points."""
    if not subject_list:
        return
    props_list = [all_data[s][session] for s in subject_list if session in all_data[s]]
    if not props_list:
        return
    arr = np.array(props_list)
    mean = np.nanmean(arr, axis=0)
    sem_vals = sem(arr, axis=0, nan_policy='omit')
    x = np.arange(1, trials_per_block+1)
    valid = ~np.isnan(mean)
    xv, yv = x[valid], mean[valid]
    if len(xv) >= 4:
        try:
            p0 = [np.max(yv), np.max(yv)-np.min(yv), 0.3]
            bounds = ([0.5,0,0], [1,1,2])
            popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
            y_fit = exp_learning(dense_x, *popt)
            ax.plot(dense_x, y_fit, color=color, linewidth=2, label=f'{label_prefix} (fit)')
            # Shaded SEM
            sem_valid = ~np.isnan(sem_vals)
            if np.sum(sem_valid) >= 2:
                interp_sem = interp1d(x[sem_valid], sem_vals[sem_valid], kind='linear', fill_value='extrapolate')
                sem_smooth = interp_sem(dense_x)
                ax.fill_between(dense_x, y_fit - sem_smooth, y_fit + sem_smooth, alpha=0.2, color=color)
        except Exception as e:
            print(f"Fit failed for {label_prefix} session {session}: {e}")
            ax.plot(xv, yv, 'o-', color=color, label=f'{label_prefix} (observed)')
    else:
        ax.plot(xv, yv, 'o-', color=color, label=f'{label_prefix} (observed)')
    if show_points:
        # Plot individual subject points (all subjects in the group)
        for s in subject_list:
            y = np.array(all_data[s][session])
            ax.plot(x, y, 'o', markersize=3, alpha=0.3, color=color, label='_nolegend_')
    # Also plot the group mean points
    ax.scatter(xv, yv, color=color, s=40, edgecolor='black', zorder=5, label='_nolegend_')

# Create output folder for these graphs
GROUP_COMP_DIR = BASE_OUTPUT_DIR / "group_comparisons"
GROUP_COMP_DIR.mkdir(exist_ok=True)

# 1. Good learners: session1 vs session2
fig, ax = plt.subplots(figsize=(10,6))
plot_group_curve(ax, good_subjects, 1, 'blue', 'Good learners S1', show_points=True)
plot_group_curve(ax, good_subjects, 2, 'red', 'Good learners S2', show_points=True)
ax.axhline(0.5, color='gray', linestyle='--', label='Chance')
ax.set_xlabel('Trial number')
ax.set_ylabel('Proportion choosing set-winner')
ax.set_ylim(0,1)
ax.set_xlim(0.9,18.1)
ax.set_xticks(range(1,19))
ax.set_title('Good learners – Session 1 vs Session 2')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GROUP_COMP_DIR / 'good_learners_s1_vs_s2.png', dpi=150)
plt.close()
print("Saved: good_learners_s1_vs_s2.png")

# 2. Bad learners: session1 vs session2
fig, ax = plt.subplots(figsize=(10,6))
plot_group_curve(ax, bad_subjects, 1, 'blue', 'Bad learners S1', show_points=True)
plot_group_curve(ax, bad_subjects, 2, 'red', 'Bad learners S2', show_points=True)
ax.axhline(0.5, color='gray', linestyle='--', label='Chance')
ax.set_xlabel('Trial number')
ax.set_ylabel('Proportion choosing set-winner')
ax.set_ylim(0,1)
ax.set_xlim(0.9,18.1)
ax.set_xticks(range(1,19))
ax.set_title('Bad learners – Session 1 vs Session 2')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GROUP_COMP_DIR / 'bad_learners_s1_vs_s2.png', dpi=150)
plt.close()
print("Saved: bad_learners_s1_vs_s2.png")

# 3. Good vs bad learners for session 1 + overall group average (all subjects)
fig, ax = plt.subplots(figsize=(10,6))
plot_group_curve(ax, good_subjects, 1, 'green', 'Good learners', show_points=False)
plot_group_curve(ax, bad_subjects, 1, 'red', 'Bad learners', show_points=False)
# Overall group average (all subjects)
all_subjects_list = subjects
plot_group_curve(ax, all_subjects_list, 1, 'gray', 'All subjects (average)', show_points=False)
ax.axhline(0.5, color='gray', linestyle='--', label='Chance')
ax.set_xlabel('Trial number')
ax.set_ylabel('Proportion choosing set-winner')
ax.set_ylim(0,1)
ax.set_xlim(0.9,18.1)
ax.set_xticks(range(1,19))
ax.set_title('Session 1 – Good vs Bad learners + overall average')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GROUP_COMP_DIR / 'good_vs_bad_s1.png', dpi=150)
plt.close()
print("Saved: good_vs_bad_s1.png")

# 4. Good vs bad learners for session 2 + overall group average
fig, ax = plt.subplots(figsize=(10,6))
plot_group_curve(ax, good_subjects, 2, 'green', 'Good learners', show_points=False)
plot_group_curve(ax, bad_subjects, 2, 'red', 'Bad learners', show_points=False)
# Overall group average (all subjects)
plot_group_curve(ax, all_subjects_list, 2, 'gray', 'All subjects (average)', show_points=False)
ax.axhline(0.5, color='gray', linestyle='--', label='Chance')
ax.set_xlabel('Trial number')
ax.set_ylabel('Proportion choosing set-winner')
ax.set_ylim(0,1)
ax.set_xlim(0.9,18.1)
ax.set_xticks(range(1,19))
ax.set_title('Session 2 – Good vs Bad learners + overall average')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GROUP_COMP_DIR / 'good_vs_bad_s2.png', dpi=150)
plt.close()
print("Saved: good_vs_bad_s2.png")

print(f"Group comparison graphs saved in {GROUP_COMP_DIR}")

# ---------- SCATTER PLOTS: GROUP AVERAGE PROPORTION CORRECT SPLIT BY WINNER (FACE / HOUSE) ----------
print("Generating scatter plots: group average proportion correct split by winner type...")
GROUP_AVG_DIR = BASE_OUTPUT_DIR / "group_average"
GROUP_AVG_DIR.mkdir(exist_ok=True)

# We need to compute per-subject per-trial proportion correct for face-winner and house-winner.
# We'll collect data from raw CSV files again.
face_acc = {sess: [] for sess in sessions}
house_acc = {sess: [] for sess in sessions}

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
    if 'learned' not in df.columns or 'trialNumber' not in df.columns or 'winner' not in df.columns:
        continue
    df['learned'] = df['learned'].astype(int)
    subj_face = []
    subj_house = []
    for t in range(1, trials_per_block+1):
        # Face‑winner trials at this position
        face_trials = df[(df['trialNumber'] == t) & (df['winner'] == 'face')]
        face_prop = face_trials['learned'].mean() if len(face_trials) > 0 else np.nan
        # House‑winner trials
        house_trials = df[(df['trialNumber'] == t) & (df['winner'] == 'house')]
        house_prop = house_trials['learned'].mean() if len(house_trials) > 0 else np.nan
        subj_face.append(face_prop)
        subj_house.append(house_prop)
    face_acc[sess].append(subj_face)
    house_acc[sess].append(subj_house)

# Compute group averages and overall proportion correct per trial (for exponential fit)
x_trials = np.arange(1, trials_per_block+1)
for sess in sessions:
    if not face_acc[sess]:
        continue
    face_arr = np.array(face_acc[sess])
    house_arr = np.array(house_acc[sess])
    face_mean = np.nanmean(face_arr, axis=0)
    house_mean = np.nanmean(house_arr, axis=0)
    
    # Compute overall proportion correct per trial (average of face and house, weighted by number of trials? Actually each trial position has both a face and a house condition in different blocks, so simple mean is fine)
    # Alternatively, compute from raw data directly:
    all_props = []
    for subj in subjects:
        # Find the CSV file for this subject and session (quick way)
        subj_file = next(RESULTS_DIR.rglob(f"{subj}_plearning_{sess}.csv"), None)
        if subj_file is None:
            continue
        df_subj = pd.read_csv(subj_file)
        if 'learned' not in df_subj.columns:
            continue
        trial_props = [df_subj[df_subj['trialNumber'] == t]['learned'].mean() for t in range(1, trials_per_block+1)]
        all_props.append(trial_props)
    overall_mean = np.nanmean(all_props, axis=0)
    
    # Fit exponential to overall_mean
    dense_x = np.linspace(1, 18, 200)
    valid = ~np.isnan(overall_mean)
    xv, yv = x_trials[valid], overall_mean[valid]
    if len(xv) >= 4:
        try:
            p0 = [np.max(yv), np.max(yv)-np.min(yv), 0.3]
            bounds = ([0.5,0,0], [1,1,2])
            popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=5000)
            y_fit = exp_learning(dense_x, *popt)
            fit_label = 'Exponential fit (overall)'
        except Exception as e:
            print(f"Fit failed for session {sess}: {e}")
            y_fit = None
            fit_label = None
    else:
        y_fit = None
        fit_label = None
    
    # Create scatter plot
    plt.figure(figsize=(10,6))
    plt.scatter(x_trials, face_mean, color='blue', s=60, label='Winner = Face', alpha=0.8, edgecolor='black')
    plt.scatter(x_trials, house_mean, color='red', s=60, label='Winner = House', alpha=0.8, edgecolor='black')
    if y_fit is not None:
        plt.plot(dense_x, y_fit, color='gray', linewidth=2, label=fit_label)
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
    plt.xlabel('Trial number')
    plt.ylabel('Proportion correct')
    plt.ylim(0,1)
    plt.xlim(0.9, 18.1)
    plt.xticks(range(1, 19))
    plt.title(f'Group average proportion correct by winner type – Session {sess} (n={len(subjects)})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    outfile = GROUP_AVG_DIR / f'group_scatter_session{sess}.png'
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved scatter plot: {outfile}")

print(f"All outputs saved in {BASE_OUTPUT_DIR}")
print(f"  - CSV: {csv_path}")
for name, path in SUB_DIRS.items():
    print(f"  - {name}: {path}")
