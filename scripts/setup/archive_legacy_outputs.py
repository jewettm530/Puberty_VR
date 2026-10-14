#!/usr/bin/env python3
"""Archive known, unused historical artifacts (never delete). Dry-run by default.

Examples:
  python3 scripts/setup/archive_legacy_outputs.py
  python3 scripts/setup/archive_legacy_outputs.py --apply

Only moves two specifically reviewed legacy paths. EEG output values differ
across legacy and current folders, so do not treat them as interchangeable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT  # noqa: E402

LEGACY = [
    Path("outputs/eeg/batch"),  # superseded storage location, not numerically equivalent
    Path("data/processed/plearning/RL_parameter_results.csv"),  # unused legacy table
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Move legacy files into archive")
    args = parser.parse_args()
    found = [item for item in LEGACY if (PROJECT_ROOT / item).exists()]
    if not found:
        print("No recognized legacy items found.")
        return 0
    archive = (PROJECT_ROOT / "outputs" / "_archived_legacy" /
               datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    for item in found:
        print(("ARCHIVE" if args.apply else "WOULD ARCHIVE"), item,
              "->", archive / item)
        if args.apply:
            destination = archive / item
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                parser.error(f"Archive collision: {destination}")
            shutil.move(str(PROJECT_ROOT / item), str(destination))
    if not args.apply:
        print("Dry run only. Review files, then re-run with --apply if desired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
