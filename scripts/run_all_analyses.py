#!/usr/bin/env python3
"""Update Puberty & VR analyses after adding subjects.

From project root: python3 scripts/run_all_analyses.py

All required stages stop on failure. EEG individual-file exclusions are audited by
process_all_eeg.py. Age comparisons are optional when the required classification
workbook is missing, unreadable, or missing subjects; they are NEVER silently
performed using guessed ages. See outputs/pipeline_run_status.json for status.

Options: --dry-run, --skip-eeg, --no-eeg-plots, --allow-unverified-eeg,
         --standardize, --strict-age-groups, --skip-age-comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
RAW_SUBJECTS = PROJECT / "data" / "raw" / "subject_data"
AGE_WORKBOOK = RAW_SUBJECTS / "Real_Subject_Data_ID.xlsx"
STATUS_PATH = PROJECT / "outputs" / "pipeline_run_status.json"
AGE_STALE_MARKER = PROJECT / "outputs" / "minor_comparison" / "NOT_CURRENT.txt"

STEPS_BEFORE_EEG = [
    "setup/raw_data_inventory.py",
    "setup/validate_raw_data.py",
    "setup/process_jatos_python.py",
    "heart_rate/summarize_heart_rate.py",
    "eeg/summarize_eeg.py",  # Schema/availability only; NOT a spectral QC pass.
]
STEPS_AFTER_EEG = [
    "learning_rates/calculate_learning_rates.py",
    "learning_rates/reward_value_alpha.py",
    "learning_rates/all_individuals_with_curve.py",
    "learning_rates/learning_summary_statistics.py",
    "learning_rates/plot_individual_learning_curves.py",
    "learning_rates/plot_group_average_curve.py",
    "learning_rates/group_average_by_winner.py",
    "learning_rates/individual_first_vs_second_halves.py",
    "learning_rates/first_half_vs_second_half.py",
    "learning_rates/learning_rate_analysis.py",
    "learning_rates/learning_slope_analysis.py",
    "learning_rates/winstay_loseswitch_analysis.py",
    "learning_rates/group_winstay_loseswitch_sessions.py",
    "learning_rates/professor_individual_learning_curves.py",
    "multimodal/build_multimodal_dataset.py",
    "multimodal/heart_rate_learning_correlations.py",
]
AGE_STEPS = [
    "multimodal/compare_minors_to_older.py",
    "multimodal/summarize_minor_comparison.py",
]


def check_age_workbook() -> tuple[bool, str]:
    """Validate classification inputs before calling the age analysis scripts.

    An Apple Numbers package can be a valid ZIP renamed '.xlsx'. Merely checking
    that it is a ZIP cannot establish that it is a valid Excel workbook.
    """
    if not AGE_WORKBOOK.is_file():
        return False, f"Missing workbook: {AGE_WORKBOOK}"
    if not zipfile.is_zipfile(AGE_WORKBOOK):
        return False, (f"Not a valid .xlsx archive: {AGE_WORKBOOK}. "
                       "Export the sheet to Excel (.xlsx), do not rename the file.")
    try:
        with zipfile.ZipFile(AGE_WORKBOOK) as archive:
            members = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as exc:
        return False, f"Cannot open workbook: {exc}"
    if not {"[Content_Types].xml", "xl/workbook.xml"}.issubset(members):
        if "Index/Document.iwa" in members:
            return False, ("Real_Subject_Data_ID.xlsx is an Apple Numbers document "
                           "with the wrong extension. Open the original in Numbers "
                           "and use File > Export To > Excel to create a genuine .xlsx.")
        return False, ("Real_Subject_Data_ID.xlsx is not an Excel .xlsx workbook. "
                       "Export it to genuine Excel format.")

    try:
        import pandas as pd
        from multimodal.compare_minors_to_older import (
            MINIMUM_ANALYZABLE_MINORS, MULTIMODAL_CSV, load_age_groups
        )
        groups = load_age_groups(AGE_WORKBOOK)
        data = pd.read_csv(MULTIMODAL_CSV, dtype={"subject_id": str})
        if "subject_id" not in data:
            return False, f"Missing subject_id column: {MULTIMODAL_CSV}"
        ids = set(data["subject_id"].dropna().astype(str).str.zfill(3))
        assigned = set(groups["subject_id"])
        missing = sorted(ids - assigned)
        if missing:
            return False, ("Missing age-workbook classifications for analyzed subjects: "
                           + ", ".join(missing) + ". Add them to the exported workbook.")
        n_minors = len(ids & set(groups.loc[groups["age_group"].eq("minor"), "subject_id"]))
        if n_minors < MINIMUM_ANALYZABLE_MINORS:
            return False, (f"Only {n_minors} analyzed minors have workbook classifications; "
                           f"analysis requires {MINIMUM_ANALYZABLE_MINORS}.")
    except Exception as exc:
        # Only the OPTIONAL age stage is skipped. The reason is explicitly
        # recorded and shown, rather than an unhandled pandas traceback.
        return False, f"Could not validate age-group workbook: {type(exc).__name__}: {exc}"
    return True, "Ready"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardize", action="store_true",
                        help="Apply raw filename renames (review a dry-run of the standardizer first).")
    parser.add_argument("--skip-eeg", action="store_true", help="Do not run spectral EEG preprocessing.")
    parser.add_argument("--eeg", action="store_true", help="Compatibility flag; EEG already runs by default.")
    parser.add_argument("--allow-unverified-eeg", action="store_true",
                        help="Process EEGs with unknown rate but label them unverified (does not bypass known mismatch).")
    parser.add_argument("--no-eeg-plots", action="store_true", help="Skip large per-recording TIFF figures.")
    parser.add_argument("--strict-age-groups", action="store_true",
                        help="Return a nonzero code if age comparison cannot be updated.")
    parser.add_argument("--skip-age-comparison", action="store_true",
                        help="Skip age comparison explicitly and mark its previous outputs as not current.")
    parser.add_argument("--dry-run", action="store_true", help="Show steps; write no files.")
    args = parser.parse_args()
    if not RAW_SUBJECTS.is_dir():
        parser.error(f"Raw subject directory not found: {RAW_SUBJECTS}")
    if args.skip_age_comparison and args.strict_age_groups:
        parser.error("--skip-age-comparison and --strict-age-groups cannot be combined")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    start = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = {"started_utc": start, "finished_utc": None,
              "project": str(PROJECT), "result": "running", "stages": [],
              "age_comparison": "pending", "age_reason": "",
              "eeg_spectral_processing": "skipped" if args.skip_eeg else "pending"}

    def record(stage: str, result: str, seconds: float = 0, detail: str = "") -> None:
        status["stages"].append({"script": stage, "status": result,
                                 "duration_seconds": round(seconds, 2), "detail": detail})

    def save_status() -> None:
        if args.dry_run:
            return
        status["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    def mark_age_stale(reason: str) -> None:
        if args.dry_run:
            return
        AGE_STALE_MARKER.parent.mkdir(parents=True, exist_ok=True)
        AGE_STALE_MARKER.write_text(
            "AGE COMPARISON OUTPUTS ARE NOT CURRENT\n"
            f"Last pipeline attempt: {start}\n"
            f"Reason: {reason}\n\n"
            "Files in outputs/minor_comparison/ and "
            "data/analysis_data/minor_comparison/ may be from an older run.\n"
            "Correct the workbook and rerun run_all_analyses.py.\n",
            encoding="utf-8",
        )

    def run(relative_script: str, *extra_args: str) -> None:
        path = SCRIPTS / relative_script
        if not path.is_file():
            raise FileNotFoundError(f"Missing project script: {path}")
        command = [sys.executable, str(path), *map(str, extra_args)]
        print(f"\n>>> {relative_script} {' '.join(map(str, extra_args))}", flush=True)
        if args.dry_run:
            return
        t0 = time.monotonic()
        try:
            subprocess.run(command, cwd=PROJECT, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            record(relative_script, "failed", time.monotonic() - t0, f"exit code {exc.returncode}")
            raise
        record(relative_script, "completed", time.monotonic() - t0)

    try:
        if args.standardize:
            run("setup/standardize_file_names.py", "--apply")
        for script in STEPS_BEFORE_EEG:
            run(script)
        if not args.skip_eeg:
            eeg_options = []
            if args.allow_unverified_eeg:
                eeg_options.append("--allow-unverified")
            if args.no_eeg_plots:
                eeg_options.append("--no-plots")
            run("eeg/process_all_eeg.py", *eeg_options)
            status["eeg_spectral_processing"] = "completed_with_file_level_qc_report"
            report = PROJECT / "data" / "analysis_data" / "eeg" / "eeg_batch_processing_report.csv"
            if report.exists():
                with report.open(newline="", encoding="utf-8") as stream:
                    qc_counts = Counter(row["processing_status"] for row in csv.DictReader(stream))
                status["eeg_file_counts"] = dict(qc_counts)
                print("EEG file-level processing statuses:", dict(qc_counts), flush=True)
        for script in STEPS_AFTER_EEG:
            run(script)
        if args.skip_age_comparison:
            reason = "Age comparison explicitly skipped (--skip-age-comparison)."
            print("\nWARNING:", reason, flush=True)
            status["age_comparison"] = "skipped"
            status["age_reason"] = reason
            mark_age_stale(reason)
        elif args.dry_run:
            print("\nAge workbook classification check deferred (dry run).")
            for script in AGE_STEPS:
                run(script)
        else:
            ready, reason = check_age_workbook()
            if not ready:
                status["age_comparison"] = "skipped"
                status["age_reason"] = reason
                mark_age_stale(reason)
                print("\nWARNING: Skipping age comparison:", reason, flush=True)
                print("Previous age results are marked NOT CURRENT.", flush=True)
                if args.strict_age_groups:
                    raise ValueError(reason)
            else:
                for script in AGE_STEPS:
                    run(script)
                status["age_comparison"] = "completed"
                AGE_STALE_MARKER.unlink(missing_ok=True)

    except subprocess.CalledProcessError as exc:
        if any(str(SCRIPTS / step) in str(exc.cmd) for step in AGE_STEPS):
            reason = f"Age comparison script failed: {exc.cmd[1]}, exit code {exc.returncode}"
            mark_age_stale(reason)
            status["age_comparison"], status["age_reason"] = "failed", reason
        print(f"\nSTOPPED: {exc.cmd[1]} failed (exit code {exc.returncode}).", file=sys.stderr)
        if status["age_comparison"] != "completed":
            mark_age_stale(status["age_reason"] or "A required upstream pipeline stage failed before the age comparison could be refreshed.")
        status["result"] = "failed"
        save_status()
        return exc.returncode or 1
    except (OSError, ValueError) as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        if status["age_comparison"] != "completed":
            mark_age_stale(status["age_reason"] or "Pipeline failed before the age comparison could be refreshed.")
        status["result"] = "failed"
        save_status()
        return 1

    status["result"] = "completed_with_age_skipped" if status["age_comparison"] == "skipped" else "completed"
    save_status()
    if args.dry_run:
        print("\nDry run complete; nothing was executed or written.")
    else:
        print("\nRequired analyses finished. Age comparison:", status["age_comparison"])
        print("Run status:", STATUS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
