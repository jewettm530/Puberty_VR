from pathlib import Path
import re
import csv

ROOT = Path("/Users/maddiemac/Puberty_VR/results")
DRY_RUN = False

STANDARD_SUFFIXES = {
    "plearning_1_eeg": "plearning_1_eeg.csv",
    "plearning_2_eeg": "plearning_2_eeg.csv",

    "baseline_hr": "baseline_hr.csv",
    "baseline_eeg": "baseline_eeg.csv",
    "pre_vr_hr": "pre_vr_hr.csv",
    "pre_vr_eeg": "pre_vr_eeg.csv",
    "vr_prep_hr": "vr_prep_hr.csv",
    "vr_prep_eeg": "vr_prep_eeg.csv",
    "peak_hr": "peak_hr.csv",
    "post_vr_hr": "post_vr_hr.csv",
    "post_vr_eeg": "post_vr_eeg.csv",
    "vr_math_hr": "vr_math_hr.csv",
    "vr_math_eeg": "vr_math_eeg.csv",
    "vr_speech_hr": "vr_speech_hr.csv",
    "vr_speech_eeg": "vr_speech_eeg.csv",

    "recovery_hr": "recovery_hr.csv",
    "recovery_eeg": "recovery_eeg.csv",
}

CORE_EXPECTED = [
    "plearning_1.csv",
    "plearning_2.csv",
    "baseline_hr.csv",
    "baseline_eeg.csv",
    "pre_vr_hr.csv",
    "pre_vr_eeg.csv",
    "peak_hr.csv",
    "post_vr_hr.csv",
    "post_vr_eeg.csv",
    "vr_math_hr.csv",
    "vr_math_eeg.csv",
    "vr_speech_hr.csv",
    "vr_speech_eeg.csv",
]

OPTIONAL_EXPECTED = [
    "plearning_1_eeg.csv",
    "plearning_2_eeg.csv",
    "recovery_hr.csv",
    "recovery_eeg.csv",
]

def normalize_name(filename):
    name = filename.lower()
    name = re.sub(r"\.(csv|json)$", "", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return name.strip()

def classify_file(filename):
    lower = filename.lower()
    name = normalize_name(filename)

    # Remove subject prefixes
    name = re.sub(r"^sub\s*\d+\s*", "", name)
    name = re.sub(r"^\d+\s*", "", name)

    # Already-standard pLearning behavioral CSV / JSON files
    if re.match(r"^\d{3}_plearning_[12]\.csv$", lower):
        return None
    if re.match(r"^subject_\d{3}_plearning_[12]\.json$", lower):
        return None

    is_hr = (
        "heart" in name
        or "hear rate" in name
        or "heartrate" in name
        or "hr" in name
    )
    is_eeg = "eeg" in name or "egg" in name

    # pLearning EEG
    if is_eeg and "plearning" in name:
        if "2" in name:
            return "plearning_2_eeg"
        return "plearning_1_eeg"

    # HR
    if is_hr:
        if name.startswith("t1 ") or "baseline" in name:
            return "baseline_hr"
        if name.startswith("t2 "):
            return "pre_vr_hr"
        if name.startswith("t3 ") or "postvr" in name or ("post" in name and "vr" in name):
            return "post_vr_hr"
        if name.startswith("t5 ") or "recovery" in name:
            return "recovery_hr"
        if "peak" in name:
            return "peak_hr"
        if "vrprep" in name or "prep" in name:
            return "vr_prep_hr"
        if ("pre" in name and "vr" in name) or name.startswith("pre "):
            return "pre_vr_hr"
        if "math" in name:
            return "vr_math_hr"
        if "speech" in name:
            return "vr_speech_hr"

    # EEG
    if is_eeg:
        if name.startswith("t1 ") or "baseline" in name:
            return "baseline_eeg"
        if name.startswith("t2 "):
            return "pre_vr_eeg"
        if name.startswith("t3 ") or "postvr" in name or ("post" in name and "vr" in name):
            return "post_vr_eeg"
        if name.startswith("t5 ") or "recovery" in name:
            return "recovery_eeg"
        if "vrprep" in name or "prep" in name:
            return "vr_prep_eeg"
        if ("pre" in name and "vr" in name) or name.startswith("pre "):
            return "pre_vr_eeg"
        if "math" in name:
            return "vr_math_eeg"
        if "speech" in name:
            return "vr_speech_eeg"

    return None

rename_log = []

for subject_dir in sorted(ROOT.glob("sub*")):
    if not subject_dir.is_dir():
        continue

    m = re.match(r"sub(\d+)$", subject_dir.name.lower())
    if not m:
        continue

    subject_num = m.group(1).zfill(3)
    planned_targets = set()

    print(f"\n{subject_dir.name}/")

    for file_path in sorted(subject_dir.iterdir()):
        if file_path.suffix.lower() not in [".csv", ".json"]:
            continue

        file_type = classify_file(file_path.name)

        if file_type is None:
            print(f"  SKIP: {file_path.name}")
            continue

        new_name = f"{subject_num}_{STANDARD_SUFFIXES[file_type]}"
        new_path = file_path.with_name(new_name)

        if file_path.name == new_name:
            print(f"  OK: {file_path.name}")
            planned_targets.add(new_name)
            continue

        if new_name in planned_targets:
            print(f"  DUPLICATE PLAN: {file_path.name} -> {new_name}")
            continue

        if new_path.exists():
            print(f"  WARNING: target exists, skipping {file_path.name} -> {new_name}")
            continue

        print(f"  RENAME: {file_path.name} -> {new_name}")
        planned_targets.add(new_name)

        rename_log.append({
            "subject": subject_dir.name,
            "old_name": file_path.name,
            "new_name": new_name,
            "old_path": str(file_path),
            "new_path": str(new_path),
        })

        if not DRY_RUN:
            file_path.rename(new_path)

# Missing-file inventory after planned renames
print("\n\nMISSING FILE REPORT")

missing_rows = []

for subject_dir in sorted(ROOT.glob("sub*")):
    if not subject_dir.is_dir():
        continue

    m = re.match(r"sub(\d+)$", subject_dir.name.lower())
    if not m:
        continue

    subject_num = m.group(1).zfill(3)

    existing = {p.name for p in subject_dir.iterdir() if p.is_file()}
    planned_new = {
        row["new_name"]
        for row in rename_log
        if row["subject"] == subject_dir.name
    }

    available = existing | planned_new

    expected_core = [
        f"{subject_num}_{suffix}" for suffix in CORE_EXPECTED
    ] + [
        f"subject_{subject_num}_plearning_1.json",
        f"subject_{subject_num}_plearning_2.json",
    ]

    expected_optional = [
        f"{subject_num}_{suffix}" for suffix in OPTIONAL_EXPECTED
    ]

    missing_core = [f for f in expected_core if f not in available]
    missing_optional = [f for f in expected_optional if f not in available]

    if missing_core or missing_optional:
        print(f"\n{subject_dir.name}")
        if missing_core:
            print("  Missing core:")
            for f in missing_core:
                print(f"    - {f}")
        if missing_optional:
            print("  Missing optional/additional:")
            for f in missing_optional:
                print(f"    - {f}")

        missing_rows.append({
            "subject": subject_dir.name,
            "missing_core": "; ".join(missing_core),
            "missing_optional": "; ".join(missing_optional),
        })

# Save logs
with open(ROOT / "rename_log.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["subject", "old_name", "new_name", "old_path", "new_path"]
    )
    writer.writeheader()
    writer.writerows(rename_log)

with open(ROOT / "missing_file_report.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["subject", "missing_core", "missing_optional"]
    )
    writer.writeheader()
    writer.writerows(missing_rows)

print(f"\nDRY_RUN = {DRY_RUN}")
print("Saved rename_log.csv")
print("Saved missing_file_report.csv")