from pathlib import Path
import re
import csv
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_paths import RAW_SUBJECT_DIR, METADATA_DIR
ROOT = RAW_SUBJECT_DIR
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def rel_path(path):
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

FILE_NAME_DICTIONARY = {
    "plearning_1_csv": {
        "filename_pattern": "{sub}_plearning_1.csv",
        "modality": "behavior",
        "phase": "plearning_1",
        "required": True,
    },
    "plearning_2_csv": {
        "filename_pattern": "{sub}_plearning_2.csv",
        "modality": "behavior",
        "phase": "plearning_2",
        "required": True,
    },
    "plearning_1_json": {
        "filename_pattern": "subject_{sub}_plearning_1.json",
        "modality": "behavior",
        "phase": "plearning_1",
        "required": True,
    },
    "plearning_2_json": {
        "filename_pattern": "subject_{sub}_plearning_2.json",
        "modality": "behavior",
        "phase": "plearning_2",
        "required": True,
    },

    "baseline_hr": {
        "filename_pattern": "{sub}_baseline_hr.csv",
        "modality": "heart_rate",
        "phase": "baseline",
        "required": True,
    },
    "baseline_eeg": {
        "filename_pattern": "{sub}_baseline_eeg.csv",
        "modality": "eeg",
        "phase": "baseline",
        "required": True,
    },
    "pre_vr_hr": {
        "filename_pattern": "{sub}_pre_vr_hr.csv",
        "modality": "heart_rate",
        "phase": "pre_vr",
        "required": True,
    },
    "pre_vr_eeg": {
        "filename_pattern": "{sub}_pre_vr_eeg.csv",
        "modality": "eeg",
        "phase": "pre_vr",
        "required": True,
    },
    "peak_hr": {
        "filename_pattern": "{sub}_peak_hr.csv",
        "modality": "heart_rate",
        "phase": "peak",
        "required": True,
    },
    "post_vr_hr": {
        "filename_pattern": "{sub}_post_vr_hr.csv",
        "modality": "heart_rate",
        "phase": "post_vr",
        "required": True,
    },
    "post_vr_eeg": {
        "filename_pattern": "{sub}_post_vr_eeg.csv",
        "modality": "eeg",
        "phase": "post_vr",
        "required": True,
    },
    "vr_math_hr": {
        "filename_pattern": "{sub}_vr_math_hr.csv",
        "modality": "heart_rate",
        "phase": "vr_math",
        "required": True,
    },
    "vr_math_eeg": {
        "filename_pattern": "{sub}_vr_math_eeg.csv",
        "modality": "eeg",
        "phase": "vr_math",
        "required": True,
    },
    "vr_speech_hr": {
        "filename_pattern": "{sub}_vr_speech_hr.csv",
        "modality": "heart_rate",
        "phase": "vr_speech",
        "required": True,
    },
    "vr_speech_eeg": {
        "filename_pattern": "{sub}_vr_speech_eeg.csv",
        "modality": "eeg",
        "phase": "vr_speech",
        "required": True,
    },

    "plearning_1_eeg": {
        "filename_pattern": "{sub}_plearning_1_eeg.csv",
        "modality": "eeg",
        "phase": "plearning_1",
        "required": False,
    },
    "plearning_2_eeg": {
        "filename_pattern": "{sub}_plearning_2_eeg.csv",
        "modality": "eeg",
        "phase": "plearning_2",
        "required": False,
    },
    "recovery_hr": {
        "filename_pattern": "{sub}_recovery_hr.csv",
        "modality": "heart_rate",
        "phase": "recovery",
        "required": False,
    },
    "recovery_eeg": {
        "filename_pattern": "{sub}_recovery_eeg.csv",
        "modality": "eeg",
        "phase": "recovery",
        "required": False,
    },
    "vr_prep_hr": {
        "filename_pattern": "{sub}_vr_prep_hr.csv",
        "modality": "heart_rate",
        "phase": "vr_prep",
        "required": False,
    },
    "vr_prep_eeg": {
        "filename_pattern": "{sub}_vr_prep_eeg.csv",
        "modality": "eeg",
        "phase": "vr_prep",
        "required": False,
    },
}


def get_subject_num(subject_folder_name):
    match = re.match(r"sub(\d+)$", subject_folder_name.lower())
    if not match:
        return None
    return match.group(1).zfill(3)


dictionary_rows = []
inventory_rows = []
raw_data_rows = []
missing_rows = []

for file_key, info in FILE_NAME_DICTIONARY.items():
    dictionary_rows.append({
        "file_key": file_key,
        "filename_pattern": info["filename_pattern"],
        "modality": info["modality"],
        "phase": info["phase"],
        "required": info["required"],
    })

for subject_dir in sorted(ROOT.glob("sub*")):
    if not subject_dir.is_dir():
        continue

    subject_num = get_subject_num(subject_dir.name)
    if subject_num is None:
        continue

    inventory_row = {
        "subject_id": subject_num,
        "subject_folder": subject_dir.name,
    }

    for file_key, info in FILE_NAME_DICTIONARY.items():
        expected_filename = info["filename_pattern"].format(sub=subject_num)
        expected_path = subject_dir / expected_filename
        exists = expected_path.exists()

        inventory_row[f"{file_key}_exists"] = exists
        inventory_row[f"{file_key}_path"] = rel_path(expected_path) if exists else ""
        if exists:
            raw_data_rows.append({
                "subject_id": subject_num,
                "subject_folder": subject_dir.name,
                "file_key": file_key,
                "modality": info["modality"],
                "phase": info["phase"],
                "required": info["required"],
                "filename": expected_filename,
                "file_path": rel_path(expected_path),
            })
        else:
            missing_rows.append({
                "subject_id": subject_num,
                "subject_folder": subject_dir.name,
                "file_key": file_key,
                "modality": info["modality"],
                "phase": info["phase"],
                "required": info["required"],
                "expected_filename": expected_filename,
                "expected_path": rel_path(expected_path),
            })

    inventory_rows.append(inventory_row)


METADATA_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(dictionary_rows).to_csv(METADATA_DIR / "file_name_dictionary.csv", index=False)
pd.DataFrame(inventory_rows).to_csv(METADATA_DIR / "subject_level_inventory.csv", index=False)
pd.DataFrame(raw_data_rows).to_csv(METADATA_DIR / "all_raw_data.csv", index=False)
pd.DataFrame(missing_rows).to_csv(METADATA_DIR / "missing_raw_data_files.csv", index=False)

print("Created:")
print(f"  {METADATA_DIR / 'file_name_dictionary.csv'}")
print(f"  {METADATA_DIR / 'subject_level_inventory.csv'}")
print(f"  {METADATA_DIR / 'all_raw_data.csv'}")
print(f"  {METADATA_DIR / 'missing_raw_data_files.csv'}")