#!/usr/bin/env python3
"""Build raw-data inventories and missing-file reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from file_naming import subject_id_from_folder  # noqa: E402
from project_paths import METADATA_DIR, PROJECT_ROOT, RAW_SUBJECT_DIR  # noqa: E402

# Behavioral CSVs inside raw subject folders are legacy/derived copies; the JSON
# exports are the required raw behavioral source. EEG entries permit CSV or a
# single-file ZIP archive to avoid unnecessarily inflating the project folder.
FILE_NAME_DICTIONARY = {
    "plearning_1_csv": {
        "filename_patterns": ["{sub}_plearning_1.csv"],
        "modality": "behavior",
        "phase": "plearning_1",
        "required": False,
    },
    "plearning_2_csv": {
        "filename_patterns": ["{sub}_plearning_2.csv"],
        "modality": "behavior",
        "phase": "plearning_2",
        "required": False,
    },
    "plearning_1_json": {
        "filename_patterns": ["subject_{sub}_plearning_1.json"],
        "modality": "behavior",
        "phase": "plearning_1",
        "required": True,
    },
    "plearning_2_json": {
        "filename_patterns": ["subject_{sub}_plearning_2.json"],
        "modality": "behavior",
        "phase": "plearning_2",
        "required": True,
    },
    "baseline_hr": {
        "filename_patterns": ["{sub}_baseline_hr.csv"],
        "modality": "heart_rate",
        "phase": "baseline",
        "required": True,
    },
    "baseline_eeg": {
        "filename_patterns": ["{sub}_baseline_eeg.csv", "{sub}_baseline_eeg.zip"],
        "modality": "eeg",
        "phase": "baseline",
        "required": True,
    },
    "pre_vr_hr": {
        "filename_patterns": ["{sub}_pre_vr_hr.csv"],
        "modality": "heart_rate",
        "phase": "pre_vr",
        "required": True,
    },
    "pre_vr_eeg": {
        "filename_patterns": ["{sub}_pre_vr_eeg.csv", "{sub}_pre_vr_eeg.zip"],
        "modality": "eeg",
        "phase": "pre_vr",
        "required": True,
    },
    "peak_hr": {
        "filename_patterns": ["{sub}_peak_hr.csv"],
        "modality": "heart_rate",
        "phase": "peak",
        "required": False,
    },
    "post_vr_hr": {
        "filename_patterns": ["{sub}_post_vr_hr.csv"],
        "modality": "heart_rate",
        "phase": "post_vr",
        "required": True,
    },
    "post_vr_eeg": {
        "filename_patterns": ["{sub}_post_vr_eeg.csv", "{sub}_post_vr_eeg.zip"],
        "modality": "eeg",
        "phase": "post_vr",
        "required": True,
    },
    "vr_math_hr": {
        "filename_patterns": ["{sub}_vr_math_hr.csv"],
        "modality": "heart_rate",
        "phase": "vr_math",
        "required": True,
    },
    "vr_math_eeg": {
        "filename_patterns": ["{sub}_vr_math_eeg.csv", "{sub}_vr_math_eeg.zip"],
        "modality": "eeg",
        "phase": "vr_math",
        "required": True,
    },
    "vr_speech_hr": {
        "filename_patterns": ["{sub}_vr_speech_hr.csv"],
        "modality": "heart_rate",
        "phase": "vr_speech",
        "required": True,
    },
    "vr_speech_eeg": {
        "filename_patterns": ["{sub}_vr_speech_eeg.csv", "{sub}_vr_speech_eeg.zip"],
        "modality": "eeg",
        "phase": "vr_speech",
        "required": True,
    },
    "plearning_1_eeg": {
        "filename_patterns": ["{sub}_plearning_1_eeg.csv", "{sub}_plearning_1_eeg.zip"],
        "modality": "eeg",
        "phase": "plearning_1",
        "required": False,
    },
    "plearning_2_eeg": {
        "filename_patterns": ["{sub}_plearning_2_eeg.csv", "{sub}_plearning_2_eeg.zip"],
        "modality": "eeg",
        "phase": "plearning_2",
        "required": False,
    },
    "recovery_hr": {
        "filename_patterns": ["{sub}_recovery_hr.csv"],
        "modality": "heart_rate",
        "phase": "recovery",
        "required": False,
    },
    "recovery_eeg": {
        "filename_patterns": ["{sub}_recovery_eeg.csv", "{sub}_recovery_eeg.zip"],
        "modality": "eeg",
        "phase": "recovery",
        "required": False,
    },
    "vr_prep_hr": {
        "filename_patterns": ["{sub}_vr_prep_hr.csv"],
        "modality": "heart_rate",
        "phase": "vr_prep",
        "required": False,
    },
    "vr_prep_eeg": {
        "filename_patterns": ["{sub}_vr_prep_eeg.csv", "{sub}_vr_prep_eeg.zip"],
        "modality": "eeg",
        "phase": "vr_prep",
        "required": False,
    },
}


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def first_existing(subject_dir: Path, patterns: list[str], subject_id: str) -> Path | None:
    for pattern in patterns:
        candidate = subject_dir / pattern.format(sub=subject_id)
        if candidate.exists():
            return candidate
    return None


def build_inventory(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dictionary_rows = []
    inventory_rows = []
    raw_data_rows = []
    missing_rows = []
    unrecognized_rows = []

    for file_key, info in FILE_NAME_DICTIONARY.items():
        dictionary_rows.append({
            "file_key": file_key,
            "filename_patterns": "; ".join(info["filename_patterns"]),
            "modality": info["modality"],
            "phase": info["phase"],
            "required": info["required"],
        })

    direct_subject_dirs = [
        path for path in sorted(root.iterdir())
        if path.is_dir() and subject_id_from_folder(path.name) is not None
    ]

    # Surface accidental nested subject folders rather than silently ignoring them.
    nested_subject_dirs = [
        path for path in root.rglob("sub*")
        if path.is_dir()
        and path.parent != root
        and subject_id_from_folder(path.name) is not None
    ]
    for nested in nested_subject_dirs:
        unrecognized_rows.append({
            "subject_id": subject_id_from_folder(nested.name),
            "subject_folder": nested.name,
            "filename": "",
            "file_path": rel_path(nested),
            "reason": "nested_subject_folder",
        })

    for subject_dir in direct_subject_dirs:
        subject_id = subject_id_from_folder(subject_dir.name)
        assert subject_id is not None

        inventory_row: dict[str, object] = {
            "subject_id": subject_id,
            "subject_folder": subject_dir.name,
        }
        recognized_paths: set[Path] = set()

        for file_key, info in FILE_NAME_DICTIONARY.items():
            existing_path = first_existing(subject_dir, info["filename_patterns"], subject_id)
            exists = existing_path is not None

            inventory_row[f"{file_key}_exists"] = exists
            inventory_row[f"{file_key}_path"] = rel_path(existing_path) if existing_path else ""

            if existing_path:
                recognized_paths.add(existing_path.resolve())
                raw_data_rows.append({
                    "subject_id": subject_id,
                    "subject_folder": subject_dir.name,
                    "file_key": file_key,
                    "modality": info["modality"],
                    "phase": info["phase"],
                    "required": info["required"],
                    "filename": existing_path.name,
                    "file_path": rel_path(existing_path),
                    "storage_format": existing_path.suffix.lower().lstrip("."),
                })
            else:
                expected = [p.format(sub=subject_id) for p in info["filename_patterns"]]
                missing_rows.append({
                    "subject_id": subject_id,
                    "subject_folder": subject_dir.name,
                    "file_key": file_key,
                    "modality": info["modality"],
                    "phase": info["phase"],
                    "required": info["required"],
                    "expected_filenames": "; ".join(expected),
                    "expected_folder": rel_path(subject_dir),
                })

        for path in sorted(p for p in subject_dir.iterdir() if p.is_file()):
            if path.resolve() not in recognized_paths and not path.name.startswith("."):
                unrecognized_rows.append({
                    "subject_id": subject_id,
                    "subject_folder": subject_dir.name,
                    "filename": path.name,
                    "file_path": rel_path(path),
                    "reason": "not_in_filename_dictionary",
                })

        inventory_rows.append(inventory_row)

    return (
        pd.DataFrame(dictionary_rows),
        pd.DataFrame(inventory_rows),
        pd.DataFrame(raw_data_rows),
        pd.DataFrame(missing_rows),
        pd.DataFrame(unrecognized_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=RAW_SUBJECT_DIR,
        help=f"Subject-data directory (default: {RAW_SUBJECT_DIR})",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=METADATA_DIR,
        help=f"Output directory (default: {METADATA_DIR})",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Subject-data directory not found: {root}")

    outputs = build_inventory(root)
    names = [
        "file_name_dictionary.csv",
        "subject_level_inventory.csv",
        "all_raw_data.csv",
        "missing_raw_data_files.csv",
        "unrecognized_raw_files.csv",
    ]

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    for dataframe, name in zip(outputs, names):
        path = args.metadata_dir / name
        dataframe.to_csv(path, index=False)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
