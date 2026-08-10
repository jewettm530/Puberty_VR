"""Filename parsing and normalization shared by setup and analysis scripts."""

from __future__ import annotations

import re
from pathlib import Path

SUBJECT_FOLDER_RE = re.compile(r"^sub[_-]?(\d+)$", re.IGNORECASE)
PLEARNING_CSV_RE = re.compile(
    r"^(?P<subject>\d+)_plearning_(?P<session>[12])\.csv$",
    re.IGNORECASE,
)
CANONICAL_PLEARNING_JSON_RE = re.compile(
    r"^subject_(?P<subject>\d+)_plearning_(?P<session>[12])\.json$",
    re.IGNORECASE,
)


def normalize_subject_id(value: object) -> str | None:
    """Return a zero-padded numeric subject ID, or None when invalid."""
    text = str(value).strip()
    if not text.isdigit():
        return None
    return text.zfill(3)


def subject_id_from_folder(name: str) -> str | None:
    match = SUBJECT_FOLDER_RE.fullmatch(name.strip())
    return normalize_subject_id(match.group(1)) if match else None


def parse_plearning_csv_name(path: str | Path) -> tuple[str, int] | None:
    """Parse only canonical behavioral CSV names; EEG files are excluded."""
    path = Path(path)
    match = PLEARNING_CSV_RE.fullmatch(path.name)
    if not match:
        return None
    return normalize_subject_id(match.group("subject")), int(match.group("session"))


def infer_plearning_json_identity(path: str | Path) -> tuple[str, int] | None:
    """Infer subject and session from canonical or common legacy JATOS names."""
    path = Path(path)

    canonical = CANONICAL_PLEARNING_JSON_RE.fullmatch(path.name)
    if canonical:
        return normalize_subject_id(canonical.group("subject")), int(canonical.group("session"))

    stem = path.stem
    lower = stem.lower()
    if "plearn" not in lower and "plearning" not in lower:
        return None

    session_match = re.search(r"p(?:learning|learn)[^0-9]*([12])(?:\D|$)", lower)
    if not session_match:
        return None
    session = int(session_match.group(1))

    subject = subject_id_from_folder(path.parent.name)
    if subject is None:
        prefix_match = re.match(r"^(\d+)", stem)
        if prefix_match:
            subject = normalize_subject_id(prefix_match.group(1))

    if subject is None:
        return None
    return subject, session


def canonical_plearning_json_name(subject_id: str, session: int) -> str:
    subject_id = normalize_subject_id(subject_id)
    if subject_id is None or session not in (1, 2):
        raise ValueError("Invalid pLearning subject/session")
    return f"subject_{subject_id}_plearning_{session}.json"


def canonical_plearning_csv_name(subject_id: str, session: int) -> str:
    subject_id = normalize_subject_id(subject_id)
    if subject_id is None or session not in (1, 2):
        raise ValueError("Invalid pLearning subject/session")
    return f"{subject_id}_plearning_{session}.csv"
