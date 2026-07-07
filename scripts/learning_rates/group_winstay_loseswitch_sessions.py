#!/usr/bin/env python3
"""
Create bar graphs comparing win‑stay and lose‑switch proportions between Session 1 and Session 2
for three groups: all subjects, good learners, and bad learners.
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
import sys 

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from project_paths import LEARNING_ANALYSIS_DATA_DIR, FIGURES_DIR, TABLES_DIR

# ---------- PATHS ----------
OUTPUT_DIR = FIGURES_DIR / "winstay_loseswitch" / "group_session_bars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
WINSTAY_CSV = TABLES_DIR / "winstay_loseswitch" / "winstay_loseswitch_summary.csv"
LEARNING_RATES_CSV = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"

# ---------- LOAD DATA ----------
if not WINSTAY_CSV.exists():
    raise FileNotFoundError(f"Missing {WINSTAY_CSV}. Run winstay_loseswitch_analysis.py first.")

df = pd.read_csv(WINSTAY_CSV, dtype={'subject': str})
df['subject'] = df['subject'].str.zfill(3)
subjects = df['subject'].tolist()
required_cols = [
    "overall_winstay_sess1",
    "overall_winstay_sess2",
    "overall_loseswitch_sess1",
    "overall_loseswitch_sess2",
]

missing_required = [col for col in required_cols if col not in df.columns]

if missing_required:
    raise ValueError(
        f"{WINSTAY_CSV} is missing required columns: {missing_required}"
    )

complete_df = df.dropna(subset=required_cols).copy()
complete_subjects = complete_df["subject"].tolist()

# ---------- CLASSIFY GOOD/BAD LEARNERS ----------
if LEARNING_RATES_CSV.exists():
    lr_df = pd.read_csv(LEARNING_RATES_CSV, dtype={'subjectId': str})
    lr_df["plearning_num"] = lr_df["plearning_num"].astype(str)
    lr_df = lr_df[lr_df["plearning_num"] == "1"].copy()
    lr_df = lr_df.dropna(subset=['learning_rate_k'])
    lr_df["subjectId"] = lr_df["subjectId"].str.zfill(3)
    if "learner_group" in lr_df.columns:
        lr_df["group"] = lr_df["learner_group"].astype(str).str.lower()
    else:
        median_k = lr_df["learning_rate_k"].median()
        lr_df["group"] = np.where(lr_df["learning_rate_k"] >= median_k, "good", "bad")
    group_map = lr_df.set_index("subjectId")["group"].to_dict()
else:
    # Fallback: use overall win‑stay proportion from session 1
    print("learning_rates.csv not found. Using overall win‑stay (session 1) for classification.")
    winstay_s1 = []
    for subj in subjects:
        row = df[df['subject'] == subj].iloc[0]
        winstay_s1.append(row['overall_winstay_sess1'])
    median_val = np.median(winstay_s1)
    group_map = {subj: ('good' if winstay_s1[i] >= median_val else 'bad') for i, subj in enumerate(subjects)}

# Split subjects
good_subjects = [s for s in complete_subjects if group_map.get(s) == "good"]
bad_subjects = [s for s in complete_subjects if group_map.get(s) == "bad"]

print(f"Good learners (n={len(good_subjects)}): {good_subjects}")
print(f"Bad learners (n={len(bad_subjects)}): {bad_subjects}")

# ---------- COMPUTE GROUP AVERAGES ----------
def compute_group_averages(subject_list):
    if not subject_list:
        return {
            "winstay_s1": np.nan,
            "winstay_s2": np.nan,
            "loseswitch_s1": np.nan,
            "loseswitch_s2": np.nan,
        }

    winstay_s1 = []
    winstay_s2 = []
    loseswitch_s1 = []
    loseswitch_s2 = []
    for subj in subject_list:
        matches = df[df["subject"] == subj]
        if matches.empty:
            continue
        row = matches.iloc[0]
        winstay_s1.append(row['overall_winstay_sess1'])
        winstay_s2.append(row['overall_winstay_sess2'])
        loseswitch_s1.append(row['overall_loseswitch_sess1'])
        loseswitch_s2.append(row['overall_loseswitch_sess2'])
    return {
        'winstay_s1': np.nanmean(winstay_s1),
        'winstay_s2': np.nanmean(winstay_s2),
        'loseswitch_s1': np.nanmean(loseswitch_s1),
        'loseswitch_s2': np.nanmean(loseswitch_s2)
    }

all_avg = compute_group_averages(complete_subjects)
good_avg = compute_group_averages(good_subjects)
bad_avg = compute_group_averages(bad_subjects)

# ---------- PLOTTING FUNCTION ----------
def plot_group(data, title, filename):
    sessions = ['Session 1', 'Session 2']
    winstay = [data['winstay_s1'], data['winstay_s2']]
    loseswitch = [data['loseswitch_s1'], data['loseswitch_s2']]
    
    x = np.arange(len(sessions))
    width = 0.35
    
    plt.figure(figsize=(6,5))
    plt.bar(x - width/2, winstay, width, label='Win‑stay', color='green')
    plt.bar(x + width/2, loseswitch, width, label='Lose‑switch', color='red')
    plt.xticks(x, sessions)
    plt.ylim(0, 1)
    plt.ylabel('Proportion of trials')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

# ---------- GENERATE GRAPHS ----------
plot_group(all_avg, 'All subjects – Win‑stay vs Lose‑switch', 'all_subjects.png')
plot_group(good_avg, 'Good learners – Win‑stay vs Lose‑switch', 'good_learners.png')
plot_group(bad_avg, 'Bad learners – Win‑stay vs Lose‑switch', 'bad_learners.png')

print(f"Graphs saved in {OUTPUT_DIR}")
