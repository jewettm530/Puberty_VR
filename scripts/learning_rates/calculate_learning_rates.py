"""
calculate_learning_rates.py

This script processes each subject’s trial‑level CSV file (from the Puberty_VR study) to compute an individual learning rate (k) using a logistic (sigmoid) function.

For each subject:
- Reads the CSV file (expected columns: 'trialNumber' and 'learned').
- Aggregates accuracy into bins of 5 trials (smoothing).
- Fits a sigmoid curve: P(t) = 1 / (1 + exp(-k*(t - t0))) to the binned data.
- Extracts the learning rate k (slope at the inflection point).
- Also records the number of trials and overall mean accuracy.

Outputs a CSV file (learning_rates.csv) with columns:
    subjectId, plearning_num, learning_rate_k, n_trials, mean_accuracy.

Paths are resolved through scripts/project_paths.py:
- Input CSVs: data/processed/plearning/
- Output CSV: data/analysis_data/learning_rates/learning_rates.csv

Dependencies: pandas, numpy, scipy.
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import sys 

sys.path.append(str(Path(__file__).resolve().parents[1])) 
from project_paths import PLEARNING_DIR, LEARNING_ANALYSIS_DATA_DIR

def sigmoid(t, k, t0):
    """Sigmoid function: 1 / (1 + exp(-k*(t - t0)))"""
    return 1 / (1 + np.exp(-k * (t - t0)))

def compute_learning_rate(trials_df):
    """
    Compute learning rate (k) from a trial dataframe.
    trials_df must have columns 'trialNumber' and 'learned' (boolean).
    Returns learning rate k (float), or NaN if fit fails.
    """
    # Aggregate accuracy in bins of 5 trials (smooth out noise)
    trials_df = trials_df.sort_values('trialNumber')
    bins = np.arange(0, trials_df['trialNumber'].max() + 5, 5)
    trials_df['bin'] = pd.cut(trials_df['trialNumber'], bins=bins, labels=bins[:-1])
    bin_accuracy = trials_df.groupby('bin', observed=False)['learned'].mean().values
    bin_centers = bins[:-1] + 2.5  # midpoints of bins
    
    # Ensure we have both positive and negative examples
    if (bin_accuracy <= 0.01).any() or (bin_accuracy >= 0.99).any():
        # Fit may fail; return NaN
        return np.nan
    
    try:
        # Initial guesses: k=0.5, t0=half of max trial number
        p0 = [0.5, trials_df['trialNumber'].max() / 2]
        popt, _ = curve_fit(sigmoid, bin_centers, bin_accuracy, p0=p0, maxfev=5000)
        k = popt[0]
        return k
    except Exception:
        return np.nan

def main():
    csv_files = list(PLEARNING_DIR.rglob("*_plearning_*.csv"))
    csv_files = [f for f in csv_files if 'subject_info_summary.csv' not in str(f)]
    
    results = []
    for csv_file in csv_files:
        parts = csv_file.stem.split('_')
        if len(parts) >= 3:
            subject_id = parts[0]
            plearning_num = parts[2]
        else:
            continue
        
        df = pd.read_csv(csv_file)
        if 'learned' not in df.columns or 'trialNumber' not in df.columns:
            print(f"Skipping {csv_file}: missing 'learned' or 'trialNumber'")
            continue
        
        if df['learned'].dtype == 'object':
            df['learned'] = df['learned'].astype(bool)
        
        k = compute_learning_rate(df)
        results.append({
            'subjectId': subject_id,
            'plearning_num': plearning_num,
            'learning_rate_k': k,
            'n_trials': len(df),
            'mean_accuracy': df['learned'].mean()
        })
    
    if results:
        out_df = pd.DataFrame(results)
        output_path = LEARNING_ANALYSIS_DATA_DIR / "learning_rates.csv"
        out_df.to_csv(output_path, index=False)
        print(f"Saved learning rates to {output_path}")
        print(out_df)
    else:
        print("No valid CSV files found.")

if __name__ == '__main__':
    main()
