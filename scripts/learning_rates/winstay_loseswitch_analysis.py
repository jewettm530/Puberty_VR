#!/usr/bin/env python3
"""
Compute conditional win‑stay (stay after gain) and lose‑switch (switch after loss) proportions
per trial (2‑18), per subject, per session. Produces a CSV summary and graphs:
- Subject‑level bar graphs for overall proportions and trial‑by‑trial
- Group average bar graphs (overall and trial‑by‑trial)
- Good vs bad learner comparisons on stay after gain / switch after loss
- Subject vs group comparison bar graphs (subject vs group averages)

All proportions are conditional on the preceding feedback.
Input: subject CSV files (name pattern: *_{id}_plearning_{session}.csv) from project_paths.PLEARNING_DIR
Output: organised subfolders under outputs/learning_rates/figures/winstay_loseswitch/
        and outputs/learning_rates/tables/winstay_loseswitch/
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
from scipy.stats import sem
import sys 

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR, TABLES_DIR

# ---------- PATHS ----------
RESULTS_DIR = PLEARNING_DIR
OUTPUT_BASE = FIGURES_DIR / "winstay_loseswitch"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
WINSTAY_TABLES_DIR = TABLES_DIR / "winstay_loseswitch"
WINSTAY_TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Subfolders for each request
REQ_FOLDERS = {
    1: "subject_bars_per_session",
    2: "subject_trial_by_trial",
    3: "group_average_bars_per_session",
    4: "group_average_trial_by_trial",
    5: "good_vs_bad_by_feedback",
    6: "subject_vs_group_comparison"
}
for folder in REQ_FOLDERS.values():
    (OUTPUT_BASE / folder).mkdir(exist_ok=True)

trials_per_block = 18
sessions = [1, 2]

# ---------- LOAD DATA ----------
# all_data[subject][session] = list of 18 proportions (already from earlier, but we need raw trial data)
# Actually we need trial‑by‑trial choices and feedback to compute win‑stay / lose‑switch.
# We'll store per subject, per session, a list of blocks, each block with:
#   choices: list of 18 (face/house)
#   feedbacks: list of 18 (gain/loss)
raw_trial_data = {}   # raw_trial_data[subject][session] = list of blocks, each block: dict with 'choices', 'feedbacks'

for csv_file in RESULTS_DIR.rglob("*_plearning_*.csv"):
    parts = csv_file.stem.split("_")
    if len(parts) < 3:
        continue
    subj_id = parts[0].zfill(3)
    try:
        sess = int(parts[2])
    except:
        continue
    if sess not in sessions:
        continue
    df = pd.read_csv(csv_file)
    required = ['trialNumber', 'blockNumber', 'chosen', 'feedback']
    if not all(col in df.columns for col in required):
        print(f"Skipping {csv_file.name}: missing required columns")
        continue
    # Group by blockNumber
    blocks = []
    for block_num, block_df in df.groupby('blockNumber'):
        block_df = block_df.sort_values('trialNumber')
        if len(block_df) != trials_per_block:
            continue
        choices = block_df['chosen'].tolist()
        feedbacks = block_df['feedback'].tolist()
        blocks.append({'choices': choices, 'feedbacks': feedbacks})
    if blocks:
        raw_trial_data.setdefault(subj_id, {})[sess] = blocks

if not raw_trial_data:
    raise ValueError("No valid trial data found.")

subjects = sorted(raw_trial_data.keys())

# ---------- HELPER: COMPUTE WIN‑STAY AND LOSE‑SWITCH FOR ONE BLOCK ----------
def winstay_loseswitch_block(choices, feedbacks):
    """
    Given lists of length 18 (trial 1..18), returns two arrays of length 18
    where for trial t (0‑based) the value is:
        winstay[t] = 1 if t>=1 and feedbacks[t-1]=='gain' and choices[t]==choices[t-1] else 0
        loseswitch[t] = 1 if t>=1 and feedbacks[t-1]=='loss' and choices[t]!=choices[t-1] else 0
    For trial 0, both are 0.
    Also returns denominators: n_gain_prev[t] = number of previous gains, n_loss_prev[t] = number of previous losses
    """
    n = len(choices)
    winstay = np.zeros(n, dtype=int)
    loseswitch = np.zeros(n, dtype=int)
    for t in range(1, n):
        prev_fb = feedbacks[t-1]
        if prev_fb == 'gain':
            if choices[t] == choices[t-1]:
                winstay[t] = 1
        elif prev_fb == 'loss':
            if choices[t] != choices[t-1]:
                loseswitch[t] = 1
    return winstay, loseswitch

# ---------- COMPUTE PER‑SUBJECT PER‑SESSION PROPORTIONS ----------
# We'll store for each subject, session:
#   winstay_props[trial_pos] = proportion of win‑stay at that trial position (across blocks)
#   loseswitch_props[trial_pos] = proportion of lose‑switch
#   overall_winstay = mean across trials (2‑18)
#   overall_loseswitch = mean across trials
results = []  # for CSV

# Also store for group averages later
group_data = {sess: {'winstay': [], 'loseswitch': [], 'n_subj': 0} for sess in sessions}

for subj in subjects:
    subj_row = {'subject': subj}
    for sess in sessions:
        if sess not in raw_trial_data[subj]:
            print(f"Subject {subj} missing session {sess}, skipping.")
            continue
        blocks = raw_trial_data[subj][sess]
        n_blocks = len(blocks)
        # Accumulate winstay and loseswitch counts per trial position (1‑18)
        winstay_counts = np.zeros(trials_per_block)
        loseswitch_counts = np.zeros(trials_per_block)
        # We'll also need denominators: number of gain trials at t-1 and loss trials at t-1? Actually we want proportion per trial position
        # Better: For each trial position t (1-indexed), count how many times winstay occurred and how many times it was possible.
        winstay_possible = np.zeros(trials_per_block)   # number of times previous trial had gain
        loseswitch_possible = np.zeros(trials_per_block) # number of times previous trial had loss
        for block in blocks:
            choices = block['choices']
            feedbacks = block['feedbacks']
            winstay_arr, loseswitch_arr = winstay_loseswitch_block(choices, feedbacks)
            # winstay_arr[0]=0, so start from index 1
            for t in range(1, trials_per_block):
                # t is 0‑based index, corresponds to trial number t+1
                if feedbacks[t-1] == 'gain':
                    winstay_possible[t] += 1
                    if winstay_arr[t] == 1:
                        winstay_counts[t] += 1
                elif feedbacks[t-1] == 'loss':
                    loseswitch_possible[t] += 1
                    if loseswitch_arr[t] == 1:
                        loseswitch_counts[t] += 1
        # Compute proportions per trial position (avoid division by zero)
        winstay_props = np.divide(winstay_counts, winstay_possible, out=np.full_like(winstay_counts, np.nan), where=winstay_possible>0)
        loseswitch_props = np.divide(loseswitch_counts, loseswitch_possible, out=np.full_like(loseswitch_counts, np.nan), where=loseswitch_possible>0)
        # Store per‑trial values in row
        for t in range(2, trials_per_block+1):  # trial numbers 2‑18
            row_idx = t  # we'll use column names like 'sess1_t2_winstay'
            subj_row[f'sess{sess}_t{t}_winstay'] = winstay_props[t-1]
            subj_row[f'sess{sess}_t{t}_loseswitch'] = loseswitch_props[t-1]
        # Overall proportions (average over trials 2‑18)
        overall_winstay = np.nanmean(winstay_props[1:])  # index 1..17 (trials 2‑18)
        overall_loseswitch = np.nanmean(loseswitch_props[1:])
        subj_row[f'overall_winstay_sess{sess}'] = overall_winstay
        subj_row[f'overall_loseswitch_sess{sess}'] = overall_loseswitch
        # Also store for group averages
        group_data[sess]['winstay'].append(winstay_props)
        group_data[sess]['loseswitch'].append(loseswitch_props)
        group_data[sess]['n_subj'] += 1
    results.append(subj_row)

# Convert results to DataFrame and save CSV
df_results = pd.DataFrame(results)
csv_path = WINSTAY_TABLES_DIR / 'winstay_loseswitch_summary.csv'
df_results.to_csv(csv_path, index=False)
print(f"Saved CSV: {csv_path}")

# ---------- COMPUTE GROUP AVERAGES ----------
group_avg = {}
for sess in sessions:
    if group_data[sess]['n_subj'] == 0:
        continue
    winstay_arr = np.array(group_data[sess]['winstay'])
    loseswitch_arr = np.array(group_data[sess]['loseswitch'])
    group_avg[sess] = {
        'winstay_mean': np.nanmean(winstay_arr, axis=0),
        'winstay_sem': sem(winstay_arr, axis=0, nan_policy='omit'),
        'loseswitch_mean': np.nanmean(loseswitch_arr, axis=0),
        'loseswitch_sem': sem(loseswitch_arr, axis=0, nan_policy='omit'),
        'n': group_data[sess]['n_subj']
    }

# ---------- 1. ONE GRAPH PER SUBJECT: 3 bars per session (win‑stay, lose‑switch, neither) ----------
print("Generating request 1: subject bar graphs per session...")
for subj in subjects:
    # Gather data for this subject (may have only one session)
    fig, ax = plt.subplots(figsize=(8,5))
    sessions_present = []
    winstay_vals = []
    loseswitch_vals = []
    for sess in sessions:
        if f'overall_winstay_sess{sess}' not in df_results.columns:
            continue
        row = df_results[df_results['subject'] == subj].iloc[0]
        win = row[f'overall_winstay_sess{sess}']
        lose = row[f'overall_loseswitch_sess{sess}']
        sessions_present.append(f'Session {sess}')
        winstay_vals.append(win)
        loseswitch_vals.append(lose)
    if not sessions_present:
        continue
    x = np.arange(len(sessions_present))
    width = 0.25
    ax.bar(x - width, winstay_vals, width, label='Win‑stay', color='green')
    ax.bar(x, loseswitch_vals, width, label='Lose‑switch', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(sessions_present)
    ax.set_ylim(0,1)
    ax.set_ylabel('Proportion')
    ax.set_title(f'Subject {subj} – Win‑stay / Lose‑switch ')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / REQ_FOLDERS[1] / f'subject_{subj}_bars.png', dpi=150)
    plt.close()

# ---------- 2. ONE GRAPH PER SUBJECT PER SESSION: bar graphs for win‑stay, lose‑switch, neither by trial ----------
print("Generating request 2: subject trial‑by‑trial bar graphs...")
for subj in subjects:
    row = df_results[df_results['subject'] == subj].iloc[0]
    for sess in sessions:
        if f'overall_winstay_sess{sess}' not in row.index:
            continue
        # Collect values for trials 2‑18
        trials = list(range(2, trials_per_block+1))
        winstay_vals = [row[f'sess{sess}_t{t}_winstay'] for t in trials]
        loseswitch_vals = [row[f'sess{sess}_t{t}_loseswitch'] for t in trials]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(trials))  # trial positions
        width = 0.25
        ax.bar(x - width, winstay_vals, width, label='Win‑stay', color='green')
        ax.bar(x, loseswitch_vals, width, label='Lose‑switch', color='red')
        ax.set_xticks(x)
        ax.set_xticklabels(trials)
        ax.set_xlabel('Trial number')
        ax.set_ylabel('Proportion')
        ax.set_ylim(0, 1)
        ax.set_title(f'Subject {subj} – Session {sess} (trials 2‑18)')
        ax.legend()
        plt.tight_layout()
        outfile = OUTPUT_BASE / REQ_FOLDERS[2] / f'subject_{subj}_sess{sess}_bars.png'
        plt.savefig(outfile, dpi=150)
        plt.close()

# ---------- 3. GROUP AVERAGE: bar graph per session (win‑stay, lose‑switch, neither) ----------
print("Generating request 3: group average bar graphs...")
for sess in sessions:
    if sess not in group_avg:
        continue
    win_mean = np.nanmean(group_avg[sess]['winstay_mean'][1:])  # trials 2‑18
    lose_mean = np.nanmean(group_avg[sess]['loseswitch_mean'][1:])
    # SEM for bars? Not requested, but we can add error bars if desired.
    plt.figure(figsize=(5,5))
    bars = plt.bar(['Win‑stay', 'Lose‑switch'], [win_mean, lose_mean],
                   color=['green', 'red'])
    plt.ylim(0,1)
    plt.ylabel('Proportion')
    plt.title(f'Group average (n={group_avg[sess]["n"]}) – Session {sess}')
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / REQ_FOLDERS[3] / f'group_average_session{sess}_bars.png', dpi=150)
    plt.close()

# ---------- 4. GROUP AVERAGE TRIAL‑BY‑TRIAL BAR GRAPHS ----------
print("Generating request 4...")
for sess in sessions:
    if sess not in group_avg:
        continue
    win_mean = group_avg[sess]['winstay_mean'][1:]  # trials 2‑18
    lose_mean = group_avg[sess]['loseswitch_mean'][1:]
    trials = list(range(2, trials_per_block+1))
    x = np.arange(len(trials))
    width = 0.35
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, win_mean, width, label='Win‑stay', color='green')
    plt.bar(x + width/2, lose_mean, width, label='Lose‑switch', color='red')
    plt.xticks(x, trials)
    plt.xlabel('Trial number')
    plt.ylabel('Conditional proportion')
    plt.ylim(0,1)
    plt.title(f'Group average (n={group_avg[sess]["n"]}) – Session {sess}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / REQ_FOLDERS[4] / f'group_average_session{sess}_by_trial.png', dpi=150)
    plt.close()

# ---------- 5. PER SESSION: good vs bad learners for "switch after loss" and "stay after gain" ----------
# Need classification of good/bad learners (using learning_rates.csv or overall mean)
print("Generating request 5: good vs bad learners by feedback type...")
learning_rates_csv = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
if learning_rates_csv.exists():
    lr_df = pd.read_csv(learning_rates_csv, dtype={'subjectId': str})
    lr_df = lr_df[lr_df['plearning_num'] == 1].copy()  # use session1 for classification
    lr_df = lr_df.dropna(subset=['learning_rate_k'])
    lr_df["subjectId"] = lr_df["subjectId"].str.zfill(3)

    if "learner_group" in lr_df.columns:
        lr_df["group"] = lr_df["learner_group"].astype(str).str.lower()
    else:
        median_k = lr_df["learning_rate_k"].median()
        lr_df["group"] = np.where(lr_df["learning_rate_k"] >= median_k, "good", "bad")

    group_map = lr_df.set_index("subjectId")["group"].to_dict()
else:
    # Fallback: use overall win‑stay? Better use overall accuracy? We'll use overall win‑stay (or just accuracy) for simplicity.
    # Here we use overall proportion correct from learning_rate_analysis (but we don't have that here). We'll compute from our data.
    overall_acc = {}
    for subj in subjects:
        # Use session 1 winstay overall? Actually use mean proportion correct from raw data? Simpler: use overall win‑stay? Not good.
        # We'll compute mean proportion correct from the raw choice data? But we don't have that easily. Instead, we'll compute overall proportion correct from the learned column? Not available.
        # Fallback: use overall win‑stay proportion from session 1 as a proxy.
        row = df_results[df_results['subject'] == subj].iloc[0]
        if f'overall_winstay_sess1' in row:
            metric = row['overall_winstay_sess1']
        else:
            metric = np.nan
        overall_acc[subj] = metric
    median_val = np.median([v for v in overall_acc.values() if not np.isnan(v)])
    group_map = {s: ('good' if overall_acc[s] >= median_val else 'bad') for s in subjects}

good_subjs = [s for s in subjects if group_map.get(s) == 'good']
bad_subjs = [s for s in subjects if group_map.get(s) == 'bad']

# For each session, compute group averages for "stay after gain" and "switch after loss"
# We already have per‑trial win‑stay (stay after gain) and lose‑switch (switch after loss). For group average, we can average across trials.
for sess in sessions:
    if sess not in group_avg:
        continue
    # For good learners
    good_winstay = []
    good_loseswitch = []
    for subj in good_subjs:
        row = df_results[df_results['subject'] == subj].iloc[0]
        if f'overall_winstay_sess{sess}' in row:
            good_winstay.append(row[f'overall_winstay_sess{sess}'])
            good_loseswitch.append(row[f'overall_loseswitch_sess{sess}'])
    good_win_mean = np.nanmean(good_winstay) if good_winstay else 0
    good_lose_mean = np.nanmean(good_loseswitch) if good_loseswitch else 0
    # For bad learners
    bad_winstay = []
    bad_loseswitch = []
    for subj in bad_subjs:
        row = df_results[df_results['subject'] == subj].iloc[0]
        if f'overall_winstay_sess{sess}' in row:
            bad_winstay.append(row[f'overall_winstay_sess{sess}'])
            bad_loseswitch.append(row[f'overall_loseswitch_sess{sess}'])
    bad_win_mean = np.nanmean(bad_winstay) if bad_winstay else 0
    bad_lose_mean = np.nanmean(bad_loseswitch) if bad_loseswitch else 0
    
    # Bar plot: two pairs (stay after gain, switch after loss) each with two bars (good, bad)
    fig, ax = plt.subplots(figsize=(6,5))
    x = np.arange(2)  # two categories
    width = 0.35
    ax.bar(x - width/2, [good_win_mean, good_lose_mean], width, label='Good learners', color='green')
    ax.bar(x + width/2, [bad_win_mean, bad_lose_mean], width, label='Bad learners', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(['Stay after gain', 'Switch after loss'])
    ax.set_ylim(0,1)
    ax.set_ylabel('Proportion')
    ax.set_title(f'Session {sess} – Good vs Bad learners')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / REQ_FOLDERS[5] / f'session{sess}_good_vs_bad.png', dpi=150)
    plt.close()

# ---------- 6. PER SUBJECT: comparison with group averages (subject vs group) for win‑stay and lose‑switch ----------
print("Generating request 6: subject vs group comparison graphs...")
for subj in subjects:
    row = df_results[df_results['subject'] == subj].iloc[0]
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(2)  # session 1 and 2
    width = 0.2
    # For session 1
    if f'overall_winstay_sess1' in row:
        subj_win1 = row['overall_winstay_sess1']
        subj_lose1 = row['overall_loseswitch_sess1']
        group_win1 = np.nanmean(group_avg[1]['winstay_mean'][1:]) if 1 in group_avg else np.nan
        group_lose1 = np.nanmean(group_avg[1]['loseswitch_mean'][1:]) if 1 in group_avg else np.nan
        ax.bar(x[0] - width*1.5, subj_win1, width, label='Subject Win‑stay', color='darkgreen')
        ax.bar(x[0] - width/2, group_win1, width, label='Group Win‑stay', color='lightgreen')
        ax.bar(x[0] + width/2, subj_lose1, width, label='Subject Lose‑switch', color='darkred')
        ax.bar(x[0] + width*1.5, group_lose1, width, label='Group Lose‑switch', color='lightcoral')
    # For session 2
    if f'overall_winstay_sess2' in row:
        subj_win2 = row['overall_winstay_sess2']
        subj_lose2 = row['overall_loseswitch_sess2']
        group_win2 = np.nanmean(group_avg[2]['winstay_mean'][1:]) if 2 in group_avg else np.nan
        group_lose2 = np.nanmean(group_avg[2]['loseswitch_mean'][1:]) if 2 in group_avg else np.nan
        ax.bar(x[1] - width*1.5, subj_win2, width, color='darkgreen')
        ax.bar(x[1] - width/2, group_win2, width, color='lightgreen')
        ax.bar(x[1] + width/2, subj_lose2, width, color='darkred')
        ax.bar(x[1] + width*1.5, group_lose2, width, color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(['Session 1', 'Session 2'])
    ax.set_ylim(0,1)
    ax.set_ylabel('Proportion')
    ax.set_title(f'Subject {subj} vs Group averages')
    # Create custom legend (since we have 4 bars per session)
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates? The labels are repeated per session, we can take unique.
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / REQ_FOLDERS[6] / f'subject_{subj}_vs_group.png', dpi=150)
    plt.close()

print("\nAll win‑stay / lose‑switch analyses completed.")
print(f"Outputs saved in {OUTPUT_BASE}")
