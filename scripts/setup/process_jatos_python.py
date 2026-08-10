#!/usr/bin/env python3
"""Convert JATOS pLearning JSON exports into canonical trial-level CSVs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from file_naming import (  # noqa: E402
    canonical_plearning_csv_name,
    infer_plearning_json_identity,
    normalize_subject_id,
)
from project_paths import METADATA_DIR, PLEARNING_DIR, PROJECT_ROOT, RAW_SUBJECT_DIR  # noqa: E402



def stored_source_path(path: Path) -> str:
    """Store project files relatively while preserving external absolute paths."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def parse_jatos(file_path: str | Path) -> pd.DataFrame:
    """Parse a normal JSON array or the concatenated-array format JATOS may export."""
    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding="utf-8")
    raw_text = " ".join(raw_text.splitlines())
    raw_text = re.sub(r"\]\s*\[", "],[", raw_text)
    stripped = raw_text.strip()

    if not stripped.startswith("["):
        raw_text = f"[{stripped}]"
    elif not stripped.endswith("]"):
        raw_text = f"[{stripped}]"

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decode error in {file_path}: {exc}") from exc

    if isinstance(data, dict):
        return pd.DataFrame([data])
    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON structure in {file_path}: {type(data).__name__}")

    rows = []
    for element in data:
        if isinstance(element, dict):
            rows.append(element)
        elif isinstance(element, list) and len(element) == 1 and isinstance(element[0], dict):
            rows.append(element[0])
        else:
            rows.append({"_raw": str(element)})
    return pd.DataFrame(rows)


def extract_datetime_parts(value: object) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, None
    parts = value.split()
    if len(parts) >= 5:
        return " ".join(parts[1:4]), parts[4]
    return None, None


def process_file(file_path: str | Path, output_dir: str | Path = PLEARNING_DIR) -> dict[str, object] | None:
    file_path = Path(file_path)
    identity = infer_plearning_json_identity(file_path)
    if identity is None:
        print(f"Skipping {file_path}: could not infer a numeric subject and pLearning session")
        return None

    subject_id, plearning_num = identity
    worker_id = f"subject_{subject_id}_plearning_{plearning_num}"
    print(f"Processing {file_path} as subject {subject_id}, session {plearning_num} ...")

    dat = parse_jatos(file_path)
    if "blocktype" not in dat.columns:
        print(f"Skipping {file_path}: no 'blocktype' column")
        return None

    behav = dat[dat["blocktype"].astype(str).str.lower() == "experiment"].copy()
    if behav.empty:
        print(f"Skipping {file_path}: no experiment blocks found")
        return None

    if "blocknr" in behav.columns:
        behav["blocknr"] = pd.to_numeric(behav["blocknr"], errors="coerce")
    else:
        behav["blocknr"] = np.nan

    unique_blocks = behav["blocknr"].dropna().unique()
    block_map = {block: index + 1 for index, block in enumerate(unique_blocks)}
    behav["blockNumber"] = behav["blocknr"].map(block_map).fillna(0).astype(int)
    behav["trialNumber"] = behav.groupby("blockNumber").cumcount() + 1

    if {"chosen", "winner"}.issubset(behav.columns):
        behav["learned"] = behav["chosen"].astype(str) == behav["winner"].astype(str)
    else:
        behav["learned"] = np.nan

    first_row = dat.iloc[0] if not dat.empty else pd.Series(dtype=object)

    def get_val(column: str, default: object = None) -> object:
        if column in first_row.index and pd.notna(first_row[column]):
            return first_row[column]
        return default

    if "winner" in behav.columns:
        block_summary = (
            behav.groupby("blockNumber", sort=True)
            .agg(winner=("winner", "first"), n_trials=("winner", "size"))
            .reset_index()
        )
        winner_variation = behav.groupby("blockNumber")["winner"].nunique(dropna=True)
        mix_blocks = int((winner_variation > 1).sum())
        face_blocks = int((block_summary["winner"].astype(str).str.lower() == "face").sum())

        face_sequence = (block_summary["winner"].astype(str).str.lower() == "face").astype(int)
        runs: list[int] = []
        current = 0
        for value in face_sequence:
            current = current + 1 if value == 1 else 0
            if current:
                runs.append(current)
        max_face_sequence = max(runs, default=0)
        block_sequence = "".join(block_summary["winner"].astype(str).str[0].str.lower())
    else:
        face_blocks = 0
        mix_blocks = 0
        max_face_sequence = 0
        block_sequence = ""

    start_date, start_time = extract_datetime_parts(get_val("datetime"))
    sub_info = {
        "subjectId": subject_id,
        "plearning_num": str(plearning_num),
        "worker_id": worker_id,
        "REDCapId": worker_id,
        "source_json": stored_source_path(file_path),
        "batch_id": get_val("batchId"),
        "study_title": get_val("studyTitle", get_val("title")),
        "opensesame_ver": get_val("opensesame_version"),
        "datetime": get_val("datetime"),
        "screenwidth": get_val("width"),
        "screenheight": get_val("height"),
        "has_context": "context" in dat.columns,
        "ntrials": len(behav),
        "face_blocks": face_blocks,
        "mix_blocks": mix_blocks,
        "max_seq": max_face_sequence,
        "block_seq": block_sequence,
        "start_date": start_date,
        "start_time": start_time,
    }

    behav["workerId"] = worker_id
    columns_to_keep = [
        "winner",
        "blockNumber",
        "blocktype",
        "chosen",
        "feedback",
        "response_time",
        "choice_face_fb",
        "response_choice_p",
        "response_time_1",
        "time_stimulus_1",
        "trialNumber",
        "learned",
        "workerId",
    ]
    existing_columns = [column for column in columns_to_keep if column in behav.columns]
    behav_out = behav[existing_columns].copy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / canonical_plearning_csv_name(subject_id, plearning_num)
    behav_out.to_csv(output_path, index=False)
    print(f"Wrote trial data: {output_path}")
    return sub_info


def infer_existing_session(frame: pd.DataFrame) -> pd.Series:
    if "plearning_num" in frame.columns:
        session = frame["plearning_num"].astype(str).str.extract(r"([12])")[0]
    else:
        session = pd.Series(index=frame.index, dtype=object)
    if "worker_id" in frame.columns:
        fallback = frame["worker_id"].astype(str).str.extract(r"plearn(?:ing)?[_-]?([12])", flags=re.I)[0]
        session = session.fillna(fallback)
    return session


def update_subject_summary(new_rows: list[dict[str, object]], summary_path: Path) -> None:
    new_df = pd.DataFrame(new_rows)
    new_df["subjectId"] = new_df["subjectId"].astype(str).map(normalize_subject_id)
    new_df["plearning_num"] = new_df["plearning_num"].astype(str)

    if summary_path.exists():
        existing = pd.read_csv(summary_path, dtype=str)
        if "subjectId" in existing.columns:
            existing["subjectId"] = existing["subjectId"].map(normalize_subject_id)
        else:
            existing["subjectId"] = None
        existing["plearning_num"] = infer_existing_session(existing)
        existing = existing.dropna(subset=["subjectId", "plearning_num"])

        new_keys = set(zip(new_df["subjectId"], new_df["plearning_num"]))
        existing_keys = list(zip(existing["subjectId"], existing["plearning_num"]))
        existing = existing[[key not in new_keys for key in existing_keys]]
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["subjectId", "plearning_num"], keep="last")
    combined = combined.sort_values(["subjectId", "plearning_num"]).reset_index(drop=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(summary_path, index=False)
    print(f"Wrote subject summary: {summary_path}")


def collect_json_files(paths: list[Path], recursive: bool) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        path = path.expanduser()
        if path.is_file() and path.suffix.lower() == ".json":
            files.add(path.resolve())
        elif path.is_dir():
            matches = path.rglob("*.json") if recursive else path.glob("*.json")
            files.update(match.resolve() for match in matches)
        else:
            print(f"Skipping invalid input: {path}")
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=None,
        help="JSON files or directories (default: recursively search the project's raw subject directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLEARNING_DIR,
        help=f"Processed trial CSV directory (default: {PLEARNING_DIR})",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=METADATA_DIR / "subject_info_summary.csv",
        help="Subject/session summary CSV path",
    )
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively")
    args = parser.parse_args()

    input_paths = args.paths or [RAW_SUBJECT_DIR]
    recursive = args.recursive or not args.paths
    json_files = collect_json_files(input_paths, recursive=recursive)
    if not json_files:
        print("No JSON files found.")
        return

    identities: dict[tuple[str, int], Path] = {}
    for json_file in json_files:
        identity = infer_plearning_json_identity(json_file)
        if identity is None:
            print(f"Skipping unrecognized JSON filename: {json_file}")
            continue
        if identity in identities:
            raise ValueError(
                f"Multiple JSON files map to subject/session {identity}: "
                f"{identities[identity]} and {json_file}"
            )
        identities[identity] = json_file

    summaries = []
    for _, json_file in sorted(identities.items()):
        result = process_file(json_file, output_dir=args.output_dir)
        if result is not None:
            summaries.append(result)

    if summaries:
        update_subject_summary(summaries, args.summary_path)
    else:
        print("No subject information collected.")


if __name__ == "__main__":
    main()
