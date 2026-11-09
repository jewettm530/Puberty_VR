#!/usr/bin/env python3
"""Summarize EEG CSVs or single-file ZIP archives from the raw-data index."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_paths import EEG_ANALYSIS_DATA_DIR, METADATA_DIR, resolve_project_path  # noqa: E402


def summarize_eeg_file(row: pd.Series) -> dict[str, object]:
    stored_path = Path(row["file_path"])
    file_path = resolve_project_path(stored_path)

    base: dict[str, object] = {
        "subject_id": str(row["subject_id"]).zfill(3),
        "file_key": row["file_key"],
        "phase": row["phase"],
        "modality": row["modality"],
        "file_path": str(stored_path),
        "storage_format": file_path.suffix.lower().lstrip("."),
        "file_exists": file_path.exists(),
        "n_rows": 0,
        "n_columns": 0,
        "column_names": "",
        "eeg_schema_valid": False,
        "read_error": "",
    }

    if not file_path.exists():
        base["read_error"] = "file_not_found"
        return base

    try:
        # pandas directly reads ZIP archives that contain exactly one CSV.
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as exc:
        base["read_error"] = str(exc)
        return base

    column_names = [str(column).strip() for column in df.columns]
    has_timestamp = "TimeStamp" in column_names
    has_eeg_signal = any(
        column.startswith(("Delta_", "Theta_", "Alpha_", "Beta_", "Gamma_", "RAW_"))
        for column in column_names
    )
    schema_valid = has_timestamp and has_eeg_signal

    base.update({
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "column_names": "; ".join(column_names),
        "eeg_schema_valid": schema_valid,
        "read_error": "" if schema_valid else "invalid_eeg_schema",
    })
    return base


def main() -> None:
    raw_index_path = METADATA_DIR / "all_raw_data.csv"
    if not raw_index_path.exists():
        raise FileNotFoundError(f"Missing {raw_index_path}. Run raw_data_inventory.py first.")

    raw = pd.read_csv(raw_index_path, dtype={"subject_id": str})
    eeg_files = raw[raw["modality"] == "eeg"].copy()
    long_df = pd.DataFrame([summarize_eeg_file(row) for _, row in eeg_files.iterrows()])

    EEG_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    long_path = EEG_ANALYSIS_DATA_DIR / "eeg_file_summary_long.csv"
    wide_path = EEG_ANALYSIS_DATA_DIR / "eeg_file_summary_wide.csv"
    long_df.to_csv(long_path, index=False)

    availability = long_df.copy()
    availability["available"] = (
        availability["file_exists"].astype(bool)
        & availability["eeg_schema_valid"].astype(bool)
        & availability["read_error"].fillna("").eq("")
    ).astype(int)

    wide_available = availability.pivot_table(
        index="subject_id", columns="phase", values="available", aggfunc="max", fill_value=0
    )
    wide_available.columns = [f"{phase}_eeg_available" for phase in wide_available.columns]

    wide_rows = long_df.pivot_table(index="subject_id", columns="phase", values="n_rows", aggfunc="first")
    wide_rows.columns = [f"{phase}_eeg_n_rows" for phase in wide_rows.columns]

    pd.concat([wide_available, wide_rows], axis=1).reset_index().to_csv(wide_path, index=False)
    print(f"Saved: {long_path}")
    print(f"Saved: {wide_path}")
    write_eeg_usability_report()


def write_eeg_usability_report() -> None:
    """Create a readable report from the actual EEG artifact-audit outputs."""
    from collections import Counter
    from datetime import datetime

    audit_dir = Path(__file__).resolve().parents[2] / "outputs" / "eeg" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    output_file = audit_dir / "EEG_Usability_Report.txt"
    recordings_file = audit_dir / "eeg_recording_qc.csv"
    priorities_file = audit_dir / "eeg_review_priorities.csv"
    windows_file = audit_dir / "eeg_window_qc_and_power.csv"

    if not recordings_file.exists():
        print(
            f"Cannot create usability report: {recordings_file} is missing.\n"
            "Run EEG_Audit.py first."
        )
        return

    recordings = pd.read_csv(recordings_file, dtype={"subject": str})
    priorities = (
        pd.read_csv(priorities_file, dtype={"subject": str})
        if priorities_file.exists()
        else pd.DataFrame()
    )
    windows = (
        pd.read_csv(windows_file, dtype={"subject": str})
        if windows_file.exists()
        else pd.DataFrame()
    )

    def subject_id(value):
        text = str(value).replace("sub", "").strip()
        return text.zfill(3)

    def number(value, default=0):
        value = pd.to_numeric(value, errors="coerce")
        return default if pd.isna(value) else int(value)

    def flag_counts(text):
        if pd.isna(text) or not str(text).strip():
            return "None reported"
        return str(text).replace("|", ", ")

    for df in (recordings, priorities, windows):
        if not df.empty:
            df["subject"] = df["subject"].map(subject_id)

    # Identify rejection reasons separately for AF7 and AF8.
    rejection_reasons = {}
    if not windows.empty and {"subject", "phase", "channel",
                              "hard_reasons", "hard_pass"}.issubset(windows.columns):
        for (sid, phase, channel), group in windows.groupby(
            ["subject", "phase", "channel"]
        ):
            rejected = group[
                group["hard_pass"].astype(str).str.lower().eq("false")
            ]
            reasons = Counter()
            for entry in rejected["hard_reasons"].dropna():
                reasons.update(
                    reason for reason in str(entry).split("|") if reason
                )
            rejection_reasons[(sid, phase, channel)] = reasons

    def describe_rejections(sid, phase, channel):
        counts = rejection_reasons.get((sid, phase, channel), Counter())
        if not counts:
            return "No hard-rejection details available"
        return ", ".join(
            f"{reason} ({count})"
            for reason, count in counts.most_common(5)
        )

    records = []
    candidate_count = 0
    audited_count = 0

    phase_order = {
        "baseline": 0,
        "pre_vr": 1,
        "post_vr": 2,
        "recovery": 3
    }

    recordings = recordings.sort_values(
        ["subject", "phase"],
        key=lambda col: (
            col.map(lambda p: phase_order.get(p, 99))
            if col.name == "phase" else col
        )
    )

    for _, rec in recordings.iterrows():
        sid = rec["subject"]
        phase = str(rec["phase"])
        audit_status = str(rec.get("status", "UNKNOWN"))

        details = priorities[
            (priorities["subject"] == sid) &
            (priorities["phase"] == phase)
        ] if not priorities.empty else pd.DataFrame()

        total = number(rec.get("windows_total"))
        af7 = number(rec.get("af7_pass"))
        af8 = number(rec.get("af8_pass"))
        paired = number(rec.get("paired_pass_windows"))
        retention = paired / total * 100 if total else 0

        is_audited = audit_status == "AUDITED"
        if is_audited:
            audited_count += 1

        candidate = is_audited and total > 0 and (
            retention >= 50 and paired >= 5
        )
        if candidate:
            candidate_count += 1

        if not is_audited:
            decision = "NOT ELIGIBLE / NOT AUDITED"
            explanation = (
                f"Audit status: {audit_status}. "
                "Recording did not complete full-rate EEG artifact QC."
            )
        elif total == 0:
            decision = "NOT ELIGIBLE"
            explanation = "No complete, acceptable 4-second epochs."
        elif candidate:
            decision = "PROVISIONAL PASS"
            explanation = (
                "Meets >=50% paired AF7/AF8 hard-pass retention "
                "and >=5 paired accepted windows."
            )
        else:
            decision = "FAILS PROVISIONAL QC SCREEN"
            problems = []
            if retention < 50:
                problems.append(
                    f"paired retention {retention:.1f}% is below 50%"
                )
            if paired < 5:
                problems.append(
                    f"only {paired} paired passing windows (minimum 5)"
                )
            explanation = "; ".join(problems)

        lines = [
            f"Subject {sid} | {phase.upper().replace('_', '-')}",
            f"  Recording: {rec.get('recording', 'Unknown')}",
            f"  Decision: {decision}",
            f"  Explanation: {explanation}",
        ]

        if is_audited and total:
            lines.extend([
                f"  Total 4-second positions: {total}",
                f"  AF7 hard passes: {af7}/{total} ({af7/total*100:.1f}%)",
                f"  AF8 hard passes: {af8}/{total} ({af8/total*100:.1f}%)",
                f"  Paired hard passes: {paired}/{total} ({retention:.1f}%)",
                "  AF7 hard-rejection reasons: "
                + describe_rejections(sid, phase, "AF7"),
                "  AF8 hard-rejection reasons: "
                + describe_rejections(sid, phase, "AF8"),
            ])

            if not details.empty:
                d = details.iloc[0]
                lines.append(
                    "  Overall hard flags: "
                    + flag_counts(d.get("top_hard_flags"))
                )
                lines.append(
                    "  Additional warnings (not automatic exclusions): "
                    + flag_counts(d.get("top_warnings"))
                )

        lines.append("")
        records.append((sid, phase, candidate, "\n".join(lines)))

    subjects = sorted(set(rec[0] for rec in records))
    prepost_subjects = [
        sid for sid in subjects
        if all(
            any(r[0] == sid and r[1] == phase and r[2] for r in records)
            for phase in ("pre_vr", "post_vr")
        )
    ]

    report_lines = [
        "PUBERTY & VR - EEG AUTOMATED QC USABILITY REPORT",
        "=" * 65,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Subjects in audit: {len(subjects)}",
        f"Recordings in audit: {len(recordings)}",
        f"Full-rate audited recordings: {audited_count}",
        f"Provisional passing recordings: {candidate_count}",
        "Subjects with passing pre/post pairs: "
        + (", ".join(prepost_subjects) or "None"),
        "",
        "PASS CRITERIA",
        "  >=50% paired AF7/AF8 hard-pass window retention",
        "  AND >=5 paired passing 4-second windows",
        "",
        "IMPORTANT:",
        "  These are automated, provisional results.",
        "  Hard-pass does NOT mean artifact-free EEG.",
        "  Manual neuroscientist review is still required.",
        "",
        "=" * 65,
        "INDIVIDUAL RECORDING RESULTS",
        "=" * 65,
        ""
    ]

    for _, _, _, entry in records:
        report_lines.append(entry)

    report_lines.extend([
        "=" * 65,
        "PRE/POST ANALYSIS ELIGIBILITY",
        "=" * 65,
        ""
    ])

    for sid in subjects:
        pre = next(
            (r for r in records if r[0] == sid and r[1] == "pre_vr"),
            None
        )
        post = next(
            (r for r in records if r[0] == sid and r[1] == "post_vr"),
            None
        )

        if pre and post and pre[2] and post[2]:
            result = "PROVISIONAL PRE/POST CANDIDATE"
        elif not pre or not post:
            result = "MISSING PRE OR POST RECORDING"
        else:
            result = "NOT ELIGIBLE UNDER CURRENT QC SCREEN"

        report_lines.append(f"Subject {sid}: {result}")

    report_lines.extend([
        "",
        "NOTE: Flag counts are channel-window occurrences, not distinct",
        "artifact events. Overlapping windows and flags can count repeatedly.",
        "Descriptions of signal abnormalities do not establish their cause.",
        "",
        "Source: EEG_Audit.py outputs in outputs/eeg/audit/",
    ])

    output_file.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8"
    )

    print(f"EEG usability report saved: {output_file}")


if __name__ == "__main__":
    main()
