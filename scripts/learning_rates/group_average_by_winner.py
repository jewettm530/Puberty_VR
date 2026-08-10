#!/usr/bin/env python3
"""
Create group average learning curves split by set‑winner type (face vs house).
For each session (1 and 2), plot two lines:
- Proportion correct when the winner was 'face' (chosen='face' and feedback='gain')
- Proportion correct when the winner was 'house' (chosen='house' and feedback='gain')
Both lines are shown with exponential fits (or linear interpolation if fit fails) and shaded SEM bands.
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
from file_naming import parse_plearning_csv_name
from learning_utils import exp_learning
from project_paths import PLEARNING_DIR, FIGURES_DIR

# ---------- EXPONENTIAL FUNCTION ----------

# ---------- PATHS ----------
RESULTS_DIR = PLEARNING_DIR
OUTPUT_DIR = FIGURES_DIR / "group_average_by_winner"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

trials_per_block = 18
sessions = [1, 2]

# ---------- LOAD DATA ----------
all_data_by_session = {1: [], 2: []}

for csv_file in RESULTS_DIR.rglob("*_plearning_*.csv"):
    parsed_name = parse_plearning_csv_name(csv_file)
    if parsed_name is None:
        continue
    subj_id, sess = parsed_name
    if sess not in sessions:
        continue
    df = pd.read_csv(csv_file)
    required = ['trialNumber', 'winner', 'chosen', 'feedback']
    if not all(col in df.columns for col in required):
        print(f"Skipping {csv_file.name}: missing required columns")
        continue
    
    face_correct = []
    house_correct = []
    for t in range(1, trials_per_block+1):
        trial_data = df[df['trialNumber'] == t]
        # Face‑winner trials
        face_trials = trial_data[trial_data['winner'] == 'face']
        if len(face_trials) > 0:
            correct_face = ((face_trials['chosen'] == 'face') & (face_trials['feedback'] == 'gain')).mean()
        else:
            correct_face = np.nan
        # House‑winner trials
        house_trials = trial_data[trial_data['winner'] == 'house']
        if len(house_trials) > 0:
            correct_house = ((house_trials['chosen'] == 'house') & (house_trials['feedback'] == 'gain')).mean()
        else:
            correct_house = np.nan
        face_correct.append(correct_face)
        house_correct.append(correct_house)
    
    all_data_by_session[sess].append({
        'subject': subj_id,
        'face_correct': face_correct,
        'house_correct': house_correct
    })

# ---------- HELPER: PLOT WITH EXPONENTIAL FIT (FALLBACK TO LINEAR INTERPOLATION) ----------
def plot_with_fit(ax, x, y, sem_vals, color, label, session_name):
    valid = ~np.isnan(y)
    xv, yv = x[valid], y[valid]
    n_valid = len(xv)
    print(f"Session {session_name} – {label}: {n_valid} valid points out of {len(x)}")
    
    if n_valid < 4:
        print(f"   Not enough points for exponential fit; plotting observed means as line.")
        ax.plot(xv, yv, 'o-', color=color, label=f'{label} (observed)', alpha=0.7)
        return
    
    try:
        # Initial guesses
        a0 = max(min(np.max(yv), 0.99), 0.1)   # ensure within bounds
        b0 = max(np.max(yv) - np.min(yv), 0.01)
        k0 = 0.3
        p0 = [a0, b0, k0]
        # Bounds: a in [0,1], b in [0,1], k in [0,2]
        bounds = ([0, 0, 0], [1, 1, 2])
        popt, _ = curve_fit(exp_learning, xv, yv, p0=p0, bounds=bounds, maxfev=10000)
        y_fit = exp_learning(dense_x, *popt)
        ax.plot(dense_x, y_fit, color=color, linewidth=2, label=f'{label} (exponential fit)')
        # Shaded SEM band (interpolated to dense x)
        sem_valid = ~np.isnan(sem_vals)
        if np.sum(sem_valid) >= 2:
            interp_sem = interp1d(x[sem_valid], sem_vals[sem_valid], kind='linear', fill_value='extrapolate')
            sem_smooth = interp_sem(dense_x)
            ax.fill_between(dense_x, y_fit - sem_smooth, y_fit + sem_smooth, alpha=0.2, color=color)
    except Exception as e:
        print(f"   Exponential fit failed: {e}")
        print(f"   Plotting observed means as line.")
        ax.plot(xv, yv, 'o-', color=color, label=f'{label} (observed)', alpha=0.7)
    
    # Always plot the observed mean points
    ax.scatter(xv, yv, color=color, s=40, edgecolor='black', zorder=5)

# ---------- GENERATE PLOTS ----------
x_trials = np.arange(1, trials_per_block+1)
dense_x = np.linspace(1, 18, 200)

for sess in sessions:
    data_list = all_data_by_session[sess]
    if not data_list:
        print(f"No data for session {sess}, skipping.")
        continue
    
    # Build arrays
    face_arr = np.array([d['face_correct'] for d in data_list])
    house_arr = np.array([d['house_correct'] for d in data_list])
    
    face_mean = np.nanmean(face_arr, axis=0)
    face_sem = sem(face_arr, axis=0, nan_policy='omit')
    house_mean = np.nanmean(house_arr, axis=0)
    house_sem = sem(house_arr, axis=0, nan_policy='omit')
    
    fig, ax = plt.subplots(figsize=(10,6))
    plot_with_fit(ax, x_trials, face_mean, face_sem, 'blue', 'Winner = face', sess)
    plot_with_fit(ax, x_trials, house_mean, house_sem, 'red', 'Winner = house', sess)
    
    ax.axhline(0.5, color='gray', linestyle='--', label='Chance (50%)')
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Proportion choosing set-winner')
    ax.set_ylim(0,1)
    ax.set_xlim(0.9, 18.1)
    ax.set_xticks(range(1,19))
    ax.set_title(f'Group average – Session {sess}\nPerformance separated by set‑winner type')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    outfile = OUTPUT_DIR / f'group_average_by_winner_session{sess}.png'
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved: {outfile}")

print("All group average split‑by‑winner graphs generated.")
