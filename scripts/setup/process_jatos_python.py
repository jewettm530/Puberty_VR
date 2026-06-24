#!/usr/bin/env python3
# process_jatos_python.py
# Processes uniformly named JSON files: subject_{id}_plearning_{num}.json

import json
import re
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from project_paths import RAW_SUBJECT_DIR, PLEARNING_DIR, METADATA_DIR

def ensure_packages():
    try:
        import pandas
        import numpy
    except ImportError:
        print("Error: pandas and numpy are required. Install with: pip install pandas numpy")
        sys.exit(1)


def parse_jatos(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    raw_text = ' '.join(raw_text.splitlines())
    raw_text = re.sub(r'\]\s*\[', '],[', raw_text)
    stripped = raw_text.strip()
    if not stripped.startswith('['):
        raw_text = '[' + raw_text + ']'
    elif re.search(r'\],\[', stripped) and not (stripped.startswith('[') and stripped.endswith(']')):
        raw_text = '[' + stripped + ']'
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {file_path}: {e}")
        raise

    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        else:
            rows = []
            for elem in data:
                if isinstance(elem, dict):
                    rows.append(elem)
                elif isinstance(elem, list) and len(elem) == 1 and isinstance(elem[0], dict):
                    rows.append(elem[0])
                else:
                    rows.append({'_raw': str(elem)})
            df = pd.DataFrame(rows)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        raise ValueError("Unexpected JSON structure")
    return df


def process_file(file_path, output_dir=PLEARNING_DIR):
    print(f"Processing {file_path} ...")

    # 1. Read and parse
    dat = parse_jatos(file_path)

    # 2. Filter practice trials
    if 'blocktype' not in dat.columns:
        print("Warning: No 'blocktype' found.")
        behav = pd.DataFrame()
    else:
        behav = dat[dat['blocktype'] == 'experiment'].copy()

    if behav.empty:
        print("No 'experiment' blocks found.")
        return None

    # 3. Renumber blocknr and compute trialNumber
    if 'blocknr' in behav.columns:
        behav['blocknr'] = pd.to_numeric(behav['blocknr'], errors='coerce')
    else:
        behav['blocknr'] = np.nan

    unique_blocks = behav['blocknr'].dropna().unique()
    block_map = {blk: i+1 for i, blk in enumerate(unique_blocks)}
    behav['blockNumber'] = behav['blocknr'].map(block_map).fillna(0).astype(int)
    behav['trialNumber'] = behav.groupby('blockNumber').cumcount() + 1

    # 4. Learned column
    if 'chosen' in behav.columns and 'winner' in behav.columns:
        behav['learned'] = (behav['chosen'] == behav['winner']).astype(bool)
    else:
        behav['learned'] = np.nan

    # 5. Helper to get values from first row
    first_row = dat.iloc[0] if not dat.empty else pd.Series()
    def get_val(col, default=None):
        if col in first_row.index and pd.notna(first_row[col]):
            return first_row[col]
        return default

    # --- Extract subject_id and plearning_num from filename ---
    stem = Path(file_path).stem  # e.g. "subject_5_plearning_1"
    match = re.match(r'subject_(\d+)_plearning_(\d+)', stem, re.IGNORECASE)
    if not match:
        print(f"Warning: Filename '{stem}' does not match pattern 'subject_N_plearning_M'. Using fallback.")
        numbers = re.findall(r'\d+', stem)
        if numbers:
            subject_id = 'unknown'
            plearning_num = numbers[-1]
        else:
            subject_id = 'unknown'
            plearning_num = 'unknown'
    else:
        subject_id = match.group(1)
        plearning_num = match.group(2)

    # ========== UNIFORM WORKER_ID AND REDCAP_ID ==========
    # Set both to the full stem (e.g., "subject_5_plearning_1")
    worker_id = stem
    redcap_id = stem
    # =====================================================

    # 6. Calculate block metrics
    if not behav.empty and 'blockNumber' in behav.columns:
        block_summary = behav.groupby('blockNumber').agg(
            winner=('winner', 'first'),
            n_trials=('winner', 'size')
        ).reset_index()
        face_blocks = (block_summary['winner'] == 'face').sum() if 'winner' in block_summary else 0
        mix_blocks = 1 if block_summary['blockNumber'].nunique() > block_summary['blockNumber'].max() else 0
        seq_face = (block_summary['winner'] == 'face').astype(int)
        if seq_face.any():
            runs = []
            cur = seq_face.iloc[0]
            cnt = 1
            for val in seq_face.iloc[1:]:
                if val == cur:
                    cnt += 1
                else:
                    if cur == 1:
                        runs.append(cnt)
                    cur = val
                    cnt = 1
            if cur == 1:
                runs.append(cnt)
            max_seq = max(runs) if runs else 0
        else:
            max_seq = 0
        winner_first_letter = block_summary['winner'].astype(str).str[0].fillna('-')
        block_seq_str = ''.join(winner_first_letter)
    else:
        face_blocks = 0
        mix_blocks = 0
        max_seq = 0
        block_seq_str = ''

    sub_info = {
        'subjectId': subject_id,
        'worker_id': worker_id,
        'REDCapId': redcap_id,
        'batch_id': get_val('batchId'),
        'study_title': get_val('studyTitle', get_val('title')),
        'opensesame_ver': get_val('opensesame_version'),
        'datetime': get_val('datetime'),
        'screenwidth': get_val('width'),
        'screenheight': get_val('height'),
        'has_context': 'context' in dat.columns,
        'ntrials': len(behav),
        'face_blocks': face_blocks,
        'mix_blocks': mix_blocks,
        'max_seq': max_seq,
        'block_seq': block_seq_str,
    }

    # Parse datetime
    if sub_info['datetime'] and isinstance(sub_info['datetime'], str):
        parts = sub_info['datetime'].split()
        if len(parts) >= 5:
            sub_info['start_date'] = ' '.join(parts[1:4])
            sub_info['start_time'] = parts[4]
        else:
            sub_info['start_date'] = None
            sub_info['start_time'] = None
    else:
        sub_info['start_date'] = None
        sub_info['start_time'] = None

    # 7. Write trial CSV
    cols_to_keep = [
        'winner', 'blockNumber', 'blocktype', 'chosen', 'feedback',
        'response_time', 'choice_face_fb', 'response_choice_p',
        'response_time_1', 'time_stimulus_1', 'trialNumber',
        'learned', 'workerId'
    ]
    behav['workerId'] = worker_id
    existing_cols = [c for c in cols_to_keep if c in behav.columns]
    behav_out = behav[existing_cols].copy()

    # Output CSV name: ###_plearning_#.csv
    subject_id_padded = str(subject_id).zfill(3)
    out_name = f"{subject_id_padded}_plearning_{plearning_num}.csv"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / out_name
    behav_out.to_csv(out_path, index=False)
    print(f"Wrote trial data to: {out_path}")

    return sub_info


def main():
    parser = argparse.ArgumentParser(description='Process uniformly named JATOS JSON files.')
    parser.add_argument('paths', nargs='+', help='JSON file(s) or folder(s) containing JSON files')
    parser.add_argument('--output_dir', default=None, help='Directory for summary CSV (default: current directory)')
    parser.add_argument('--recursive', action='store_true', help='Recursively search subfolders for JSON files')
    args = parser.parse_args()

    ensure_packages()

    # Collect all JSON files
    json_files = []
    for p in args.paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() == '.json':
            json_files.append(path)
        elif path.is_dir():
            if args.recursive:
                json_files.extend(path.rglob('*.json'))
            else:
                json_files.extend(path.glob('*.json'))
        else:
            print(f"Skipping (not a .json file or directory): {p}")

    if not json_files:
        print("No JSON files found.")
        return

    all_sub_info = []
    for f in json_files:
        res = process_file(str(f), output_dir=PLEARNING_DIR)
        if res is not None:
            all_sub_info.append(res)

    if all_sub_info:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = METADATA_DIR / "subject_info_summary.csv"
        new_df = pd.DataFrame(all_sub_info)
        
        if summary_path.exists():
            existing = pd.read_csv(summary_path)
            # Combine and drop duplicates based on 'worker_id' (unique per run)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['worker_id'], keep='first')
            combined.to_csv(summary_path, index=False)
        else:
            new_df.to_csv(summary_path, index=False)
        print(f"Wrote subject summary to: {summary_path}")
    else:
        print("No subject info collected.")


if __name__ == '__main__':
    main()