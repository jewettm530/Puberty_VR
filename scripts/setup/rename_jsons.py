#!/usr/bin/env python3
"""Standardize pLearning JSON filenames only.

This is a focused alternative to ``standardize_file_names.py``. It is a dry run
unless ``--apply`` is supplied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from file_naming import (  # noqa: E402
    canonical_plearning_json_name,
    infer_plearning_json_identity,
)
from project_paths import RAW_SUBJECT_DIR  # noqa: E402


def rename_json_files(base_path: Path, apply: bool = False) -> tuple[int, int]:
    renamed = 0
    skipped = 0

    for json_path in sorted(base_path.rglob("*.json")):
        identity = infer_plearning_json_identity(json_path)
        if identity is None:
            print(f"SKIP: could not infer subject/session from {json_path}")
            skipped += 1
            continue

        subject_id, session = identity
        target = json_path.with_name(canonical_plearning_json_name(subject_id, session))

        if json_path.name == target.name:
            print(f"OK: {json_path}")
            continue
        if target.exists():
            print(f"SKIP: target already exists: {target}")
            skipped += 1
            continue

        verb = "RENAME" if apply else "WOULD RENAME"
        print(f"{verb}: {json_path.name} -> {target.name}")
        if apply:
            json_path.rename(target)
        renamed += 1

    return renamed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=RAW_SUBJECT_DIR,
        help=f"Subject-data directory (default: {RAW_SUBJECT_DIR})",
    )
    parser.add_argument("--apply", action="store_true", help="Actually rename files")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Subject-data directory not found: {root}")

    renamed, skipped = rename_json_files(root, apply=args.apply)
    mode = "renamed" if args.apply else "eligible for renaming"
    print(f"\nSummary: {renamed} {mode}; {skipped} skipped")


if __name__ == "__main__":
    main()
