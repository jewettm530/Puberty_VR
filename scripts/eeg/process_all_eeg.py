#!/usr/bin/env python3
"""Batch-run the existing EEG preprocessing for recognized EEG files per subject.

Output is separated from raw data. Processes CSV and single-CSV ZIP archives.
Each recording gets a status in eeg_batch_processing_report.csv. Quality checks
are screening aids, NOT a validation of EEG signal quality or artefact rejection.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from file_naming import subject_id_from_folder  # noqa: E402
from project_paths import EEG_ANALYSIS_DATA_DIR, EEG_OUTPUTS_DIR, RAW_SUBJECT_DIR, PROJECT_ROOT  # noqa: E402
from PreprocessingScript_v1 import FS, analyze  # noqa: E402

EEG_NAME = re.compile(r"(?P<id>\d+)_(?P<phase>.+)_eeg\.(?:csv|zip)$", re.I)
RAW_CHANNELS = ("RAW_AF7", "RAW_AF8")


def portable_path(path: Path) -> str:
    """Keep audit paths valid when the project moves to another machine."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def discover_files(root: Path) -> list[tuple[str, str, Path]]:
    found = {}
    for folder in sorted(root.iterdir()):
        sid = subject_id_from_folder(folder.name) if folder.is_dir() else None
        if sid is None:
            continue
        for path in sorted(folder.iterdir()):
            match = EEG_NAME.fullmatch(path.name) if path.is_file() else None
            if not match or match.group("id").zfill(3) != sid:
                continue
            phase = match.group("phase").lower()
            key = (sid, phase)
            # Prefer uncompressed CSV when both representations exist.
            if key not in found or (path.suffix.lower() == ".csv" and found[key].suffix.lower() != ".csv"):
                found[key] = path
    return [(sid, phase, path) for (sid, phase), path in sorted(found.items())]


def basic_quality_check(path: Path) -> tuple[str, str, float | None]:
    """Conservatively screen for obvious non-256-Hz recordings.

    This check does not judge muscle/motion artifacts or signal reliability.
    Unknown sampling rate remains unverified; do not assume it has passed QC.
    """
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.namelist() if not m.endswith("/") and m.lower().endswith(".csv")]
            if len(members) != 1:
                return "excluded", f"ZIP has {len(members)} CSV members; expected exactly 1", None

    headers = list(pd.read_csv(path, nrows=0).columns)
    missing = [column for column in RAW_CHANNELS if column not in headers]
    if missing:
        return "excluded", "Missing frontal raw EEG channels: " + ", ".join(missing), None

    if "TimeStamp" not in headers:
        return "unverified", "No TimeStamp column: 256-Hz assumption not verified", None

    stamps = pd.read_csv(path, usecols=["TimeStamp"])["TimeStamp"]
    parsed = pd.to_numeric(stamps, errors="coerce")
    if parsed.notna().sum() < 2:
        # Some exports store ISO datetime stamps instead of numbers.
        dates = pd.to_datetime(stamps, errors="coerce", utc=True)
        if dates.notna().sum() < 2:
            return "unverified", "Could not infer sampling frequency from TimeStamp", None
        duration = (dates.dropna().iloc[-1] - dates.dropna().iloc[0]).total_seconds()
        sample_count = int(dates.notna().sum())
        possible_rates = [(sample_count - 1) / duration] if duration > 0 else []
    else:
        valid = parsed.dropna()
        span = float(valid.iloc[-1] - valid.iloc[0])
        sample_count = len(valid)
        # Numeric timestamps might be seconds or milliseconds.
        possible_rates = [((sample_count - 1) / (span / unit)) for unit in (1, 1000) if span > 0]

    if not possible_rates:
        return "unverified", "TimeStamp duration is zero or invalid", None
    closest = min(possible_rates, key=lambda x: abs(x - FS))
    if not np.isfinite(closest):
        return "unverified", "Could not estimate sampling rate", None
    if not 0.85 * FS <= closest <= 1.15 * FS:
        return "excluded", f"Estimated {closest:.1f} Hz is inconsistent with fixed {FS}-Hz analysis", closest
    return "rate_screen_pass", "Sampling rate appears compatible with 256 Hz; artifacts remain unchecked", closest


def run_batch(root: Path, *, allow_unverified: bool = False, save_plots: bool = True) -> int:
    if not root.is_dir():
        raise FileNotFoundError(f"Subject folder not found: {root}")
    files = discover_files(root)
    rows = []
    for i, (subject, phase, path) in enumerate(files, 1):
        print(f"\nEEG {i}/{len(files)}: subject {subject}, {phase}: {path.name}", flush=True)
        row = {"subject_id": subject, "phase": phase, "file_path": portable_path(path),
               "sampling_rate_est_hz": np.nan, "qc_status": "", "qc_reason": "",
               "processing_status": "", "af7_beta_alpha": np.nan,
               "af8_beta_alpha": np.nan, "result_path": "", "error": ""}
        try:
            qc, reason, sample_rate = basic_quality_check(path)
            row["qc_status"], row["qc_reason"] = qc, reason
            row["sampling_rate_est_hz"] = sample_rate
            if qc == "excluded" or (qc == "unverified" and not allow_unverified):
                row["processing_status"] = "skipped_qc"
                print(f"  SKIPPED ({qc}): {reason}", flush=True)
            else:
                # The original code also requires four RAW EEG + four AUX channels.
                # We preserve that method instead of changing the scientific analysis.
                out = EEG_OUTPUTS_DIR / "bandpower" / f"sub_{subject}"
                _, results_path, matrix = analyze(path, show_figure=False, output_dir=out,
                                                   save_figure=save_plots)
                # Original MATLAB-compatible order follows CSV channel order.
                channel_order = [c for c in pd.read_csv(path, nrows=0).columns
                                 if c in {"RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"}]
                row["af7_beta_alpha"] = float(matrix[1, channel_order.index("RAW_AF7")])
                row["af8_beta_alpha"] = float(matrix[1, channel_order.index("RAW_AF8")])
                row["result_path"] = portable_path(results_path)
                row["processing_status"] = "processed" if qc == "rate_screen_pass" else "processed_unverified"
        except Exception as exc:
            row["processing_status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {row['error']}", flush=True)
        rows.append(row)

    EEG_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    report = EEG_ANALYSIS_DATA_DIR / "eeg_batch_processing_report.csv"
    pd.DataFrame(rows, columns=["subject_id", "phase", "file_path", "sampling_rate_est_hz",
                                "qc_status", "qc_reason", "processing_status", "af7_beta_alpha",
                                "af8_beta_alpha", "result_path", "error"]).to_csv(report, index=False)
    done = sum(r["processing_status"] in {"processed", "processed_unverified"} for r in rows)
    flagged = sum(r["processing_status"] == "skipped_qc" for r in rows)
    failed = sum(r["processing_status"] == "failed" for r in rows)
    print(f"\nEEG batch: {done} processed, {flagged} skipped QC, {failed} failed, {len(files)} discovered.")
    print(f"Audit report: {report}")
    print("A rate-screen pass does not establish artifact-free EEG. Review quality before inference.")
    return 0  # File-level issues are captured in report; do not block other modalities.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RAW_SUBJECT_DIR)
    parser.add_argument("--allow-unverified", action="store_true",
                        help="Also process recordings whose sampling rate could not be verified; mark them unverified.")
    parser.add_argument("--no-plots", action="store_true", help="Save bandpower only, not TIFF figures.")
    args = parser.parse_args()
    return run_batch(args.root, allow_unverified=args.allow_unverified, save_plots=not args.no_plots)


if __name__ == "__main__":
    raise SystemExit(main())
