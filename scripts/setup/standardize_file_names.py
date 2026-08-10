#!/usr/bin/env python3
"""Safely standardize raw subject filenames.

The script performs a dry run by default. Pass ``--apply`` to rename files.
It supports CSV, JSON, and single-file EEG ZIP archives while preserving ZIP
compression. A detailed log is written to ``data/metadata/rename_log.csv``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from file_naming import (  # noqa: E402
    canonical_plearning_json_name,
    infer_plearning_json_identity,
    parse_plearning_csv_name,
    subject_id_from_folder,
)
from project_paths import METADATA_DIR, RAW_SUBJECT_DIR  # noqa: E402

CANONICAL_STEMS = {
    "plearning_1_eeg": "plearning_1_eeg",
    "plearning_2_eeg": "plearning_2_eeg",
    "baseline_hr": "baseline_hr",
    "baseline_eeg": "baseline_eeg",
    "pre_vr_hr": "pre_vr_hr",
    "pre_vr_eeg": "pre_vr_eeg",
    "vr_prep_hr": "vr_prep_hr",
    "vr_prep_eeg": "vr_prep_eeg",
    "peak_hr": "peak_hr",
    "post_vr_hr": "post_vr_hr",
    "post_vr_eeg": "post_vr_eeg",
    "vr_math_hr": "vr_math_hr",
    "vr_math_eeg": "vr_math_eeg",
    "vr_speech_hr": "vr_speech_hr",
    "vr_speech_eeg": "vr_speech_eeg",
    "recovery_hr": "recovery_hr",
    "recovery_eeg": "recovery_eeg",
}


def normalize_name(filename: str) -> str:
    name = Path(filename).stem.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return name.strip()


def classify_measurement_file(filename: str) -> str | None:
    """Classify a nonbehavioral raw file into a canonical measurement type."""
    path = Path(filename)
    if path.suffix.lower() not in {".csv", ".zip"}:
        return None

    # Canonical behavioral CSVs are handled by the JATOS processor, not renamed here.
    if parse_plearning_csv_name(path) is not None:
        return None

    name = normalize_name(filename)
    name = re.sub(r"^sub\s*\d+\s*", "", name)
    name = re.sub(r"^\d+\s*", "", name)

    tokens = set(name.split())
    is_eeg = "eeg" in tokens or "egg" in tokens
    is_hr = (
        "hr" in tokens
        or "heartrate" in tokens
        or ("heart" in tokens and "rate" in tokens)
        or ("hear" in tokens and "rate" in tokens)
    )

    if is_eeg and ("plearning" in name or "plearn" in name):
        session_match = re.search(r"p(?:learning|learn)\s*([12])", name)
        session = session_match.group(1) if session_match else "1"
        return f"plearning_{session}_eeg"

    if not (is_hr or is_eeg):
        return None

    modality = "eeg" if is_eeg else "hr"

    if name.startswith("t1 ") or "baseline" in tokens:
        phase = "baseline"
    elif name.startswith("t5 ") or "recovery" in tokens:
        phase = "recovery"
    elif "peak" in tokens:
        phase = "peak"
    elif "math" in tokens:
        phase = "vr_math"
    elif "speech" in tokens:
        phase = "vr_speech"
    elif "prep" in tokens or "vrprep" in name:
        phase = "vr_prep"
    elif name.startswith("t3 ") or "postvr" in name or ("post" in tokens and "vr" in tokens):
        phase = "post_vr"
    elif name.startswith("t2 ") or "prevr" in name or ("pre" in tokens and "vr" in tokens):
        phase = "pre_vr"
    else:
        return None

    # There is no defined peak EEG file type in the current schema.
    if phase == "peak" and modality == "eeg":
        return None
    return f"{phase}_{modality}"


def case_safe_rename(source: Path, target: Path) -> None:
    """Rename safely even when only filename capitalization changes."""
    if source == target:
        return
    if source.name.lower() == target.name.lower():
        temporary = source.with_name(source.name + ".rename_tmp")
        source.rename(temporary)
        temporary.rename(target)
    else:
        source.rename(target)


def plan_subject_renames(subject_dir: Path) -> list[dict[str, str]]:
    subject_id = subject_id_from_folder(subject_dir.name)
    if subject_id is None:
        return []

    rows: list[dict[str, str]] = []
    reserved_targets: set[str] = set()

    for file_path in sorted(p for p in subject_dir.iterdir() if p.is_file()):
        target_name: str | None = None
        category: str | None = None

        if file_path.suffix.lower() == ".json":
            identity = infer_plearning_json_identity(file_path)
            if identity is not None:
                json_subject, session = identity
                if json_subject != subject_id:
                    rows.append({
                        "subject": subject_dir.name,
                        "category": "plearning_json",
                        "old_name": file_path.name,
                        "new_name": "",
                        "status": "subject_mismatch",
                        "detail": f"filename implies {json_subject}; folder implies {subject_id}",
                    })
                    continue
                target_name = canonical_plearning_json_name(subject_id, session)
                category = "plearning_json"
        else:
            category = classify_measurement_file(file_path.name)
            if category is not None:
                target_name = f"{subject_id}_{CANONICAL_STEMS[category]}{file_path.suffix.lower()}"

        if target_name is None:
            continue

        target_path = file_path.with_name(target_name)
        target_key = target_name.lower()

        if file_path.name == target_name:
            status = "already_standard"
            detail = ""
        elif file_path.name.lower() == target_name.lower():
            status = "rename"
            detail = "case-only normalization"
            reserved_targets.add(target_key)
        elif target_key in reserved_targets:
            status = "duplicate_target_in_plan"
            detail = "another source file maps to the same target"
        elif target_path.exists() and target_path != file_path:
            status = "target_exists"
            detail = "target file already exists"
        else:
            status = "rename"
            detail = ""
            reserved_targets.add(target_key)

        rows.append({
            "subject": subject_dir.name,
            "category": category or "",
            "old_name": file_path.name,
            "new_name": target_name,
            "status": status,
            "detail": detail,
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=RAW_SUBJECT_DIR,
        help=f"Subject-data directory (default: {RAW_SUBJECT_DIR})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, only a dry run is performed.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=METADATA_DIR / "rename_log.csv",
        help="CSV path for the rename report.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Subject-data directory not found: {root}")

    rows: list[dict[str, str]] = []
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir() or subject_id_from_folder(subject_dir.name) is None:
            continue
        subject_rows = plan_subject_renames(subject_dir)
        rows.extend(subject_rows)

        print(f"\n{subject_dir.name}/")
        for row in subject_rows:
            print(f"  {row['status'].upper()}: {row['old_name']}" + (f" -> {row['new_name']}" if row["new_name"] else ""))
            if args.apply and row["status"] == "rename":
                case_safe_rename(subject_dir / row["old_name"], subject_dir / row["new_name"])
                row["status"] = "renamed"

    args.log.parent.mkdir(parents=True, exist_ok=True)
    columns = ["subject", "category", "old_name", "new_name", "status", "detail"]
    pd.DataFrame(rows, columns=columns).to_csv(args.log, index=False)

    rename_count = sum(row["status"] in {"rename", "renamed"} for row in rows)
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"\n{mode}: {rename_count} file(s) eligible for renaming")
    print(f"Saved log: {args.log}")


if __name__ == "__main__":
    main()
