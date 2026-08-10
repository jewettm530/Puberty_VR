#!/usr/bin/env python3
"""Validate raw-file schemas, filename conflicts, and exact duplicates."""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from file_naming import subject_id_from_folder  # noqa: E402
from project_paths import METADATA_DIR, PROJECT_ROOT, RAW_SUBJECT_DIR, resolve_project_path  # noqa: E402


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def first_line(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_members) != 1:
                raise ValueError(f"expected one CSV in ZIP; found {len(csv_members)}")
            with archive.open(csv_members[0]) as stream:
                return stream.readline().decode("utf-8-sig", errors="replace").strip()
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        return stream.readline().strip()


def schema_issue(modality: str, path: Path) -> str | None:
    try:
        header = first_line(path)
    except Exception as exc:
        return f"unable_to_read_header: {exc}"

    if modality == "eeg":
        has_timestamp = "TimeStamp" in header
        has_signal = any(token in header for token in ("Delta_", "Theta_", "Alpha_", "Beta_", "Gamma_", "RAW_"))
        if not (has_timestamp and has_signal):
            return "expected Mind Monitor EEG columns"
    elif modality == "heart_rate":
        if "Average heart rate (bpm)" not in header and "HR (bpm)" not in header:
            return "expected Polar heart-rate export columns"
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_quality_report(root: Path, metadata_dir: Path) -> pd.DataFrame:
    issues: list[dict[str, object]] = []
    raw_index_path = metadata_dir / "all_raw_data.csv"
    if not raw_index_path.exists():
        raise FileNotFoundError(f"Missing {raw_index_path}. Run raw_data_inventory.py first.")

    raw_index = pd.read_csv(raw_index_path, dtype={"subject_id": str})
    for _, row in raw_index.iterrows():
        modality = str(row["modality"])
        if modality not in {"eeg", "heart_rate"}:
            continue
        path = resolve_project_path(row["file_path"])
        issue = schema_issue(modality, path)
        if issue:
            issues.append({
                "subject_id": str(row["subject_id"]).zfill(3),
                "issue_type": "modality_schema_mismatch",
                "severity": "error",
                "file_key": row["file_key"],
                "file_path": relative_path(path),
                "related_file_path": "",
                "detail": issue,
            })

    # Hash only same-size files within each subject, which avoids reading most
    # large recordings while still detecting exact duplicate content.
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        subject_id = subject_id_from_folder(subject_dir.name)
        if subject_id is None:
            continue
        by_size: dict[int, list[Path]] = defaultdict(list)
        for path in subject_dir.iterdir():
            if path.is_file() and not path.name.startswith("."):
                by_size[path.stat().st_size].append(path)
        for same_size in by_size.values():
            if len(same_size) < 2:
                continue
            by_hash: dict[str, list[Path]] = defaultdict(list)
            for path in same_size:
                by_hash[sha256(path)].append(path)
            for matches in by_hash.values():
                if len(matches) < 2:
                    continue
                primary = sorted(matches)[0]
                for duplicate in sorted(matches)[1:]:
                    issues.append({
                        "subject_id": subject_id,
                        "issue_type": "exact_duplicate_content",
                        "severity": "warning",
                        "file_key": "",
                        "file_path": relative_path(primary),
                        "related_file_path": relative_path(duplicate),
                        "detail": "files are byte-for-byte identical; verify phase labels and remove redundant copies only after review",
                    })

    rename_log_path = metadata_dir / "rename_log.csv"
    if rename_log_path.exists():
        rename_log = pd.read_csv(rename_log_path)
        conflicts = rename_log[rename_log["status"].isin(["target_exists", "duplicate_target_in_plan", "subject_mismatch"])]
        for _, row in conflicts.iterrows():
            subject_dir = root / str(row["subject"])
            issues.append({
                "subject_id": subject_id_from_folder(str(row["subject"])) or "",
                "issue_type": "filename_conflict",
                "severity": "warning",
                "file_key": row.get("category", ""),
                "file_path": relative_path(subject_dir / str(row["old_name"])),
                "related_file_path": relative_path(subject_dir / str(row["new_name"])),
                "detail": str(row.get("detail", "")),
            })

    columns = [
        "subject_id", "issue_type", "severity", "file_key", "file_path",
        "related_file_path", "detail",
    ]
    return pd.DataFrame(issues, columns=columns).sort_values(
        ["severity", "subject_id", "issue_type", "file_path"],
        kind="stable",
    ) if issues else pd.DataFrame(columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RAW_SUBJECT_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    report = build_quality_report(root, args.metadata_dir)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.metadata_dir / "raw_data_quality_issues.csv"
    report.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Issues: {len(report)}")
    if not report.empty:
        print(report[["subject_id", "issue_type", "severity", "file_path", "related_file_path"]].to_string(index=False))


if __name__ == "__main__":
    main()
