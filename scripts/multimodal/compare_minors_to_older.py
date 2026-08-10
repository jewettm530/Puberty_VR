#!/usr/bin/env python3
"""Compare minor-coded subjects with older subjects across multimodal outcomes.

Minor status is defined only by the first column of
``data/raw/subject_data/Real_Subject_Data_ID.xlsx``. After whitespace and case
normalization, the value ``N/A Minor`` is classified as ``minor``; every other
matched subject is classified as ``older_than_minor``.

The analysis avoids pseudoreplication in two ways:

* variables that are constant within subject are compared once per subject;
* variables that vary within subject are compared separately by session.

Continuous variables use Welch's independent-samples t-test and Hedges' g.
Binary variables use Fisher's exact test and an odds ratio. Benjamini-Hochberg
false-discovery-rate correction is applied across all valid tests.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_ind

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_paths import (  # noqa: E402
    MINOR_COMPARISON_ANALYSIS_DATA_DIR,
    MINOR_COMPARISON_OUTPUTS_DIR,
    MULTIMODAL_ANALYSIS_DATA_DIR,
    RAW_SUBJECT_DIR,
)


SUBJECT_WORKBOOK = RAW_SUBJECT_DIR / "Real_Subject_Data_ID.xlsx"
MULTIMODAL_CSV = MULTIMODAL_ANALYSIS_DATA_DIR / "multimodal_subject_session_summary.csv"
MINIMUM_ANALYZABLE_MINORS = 5

NON_OUTCOME_COLUMNS = {
    "plearning_num",
    "session",
    "subjectId_subjectinfo",
    "plearning_num_subjectinfo",
    "batch_id",
    "screenwidth",
    "screenheight",
    "ntrials",
    "n_trials",
    "median_session1_learning_rate_k",
    "classification_learning_rate_k",
    "reported_age",
    "classification_source_row",
    "fit_a",
    "fit_b",
    "face_blocks",
    "mix_blocks",
    "max_seq",
    "has_context",
    "k_sess1",
    "k_sess2",
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def clean_subject_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.search(r"(\d+)", str(value))
    return match.group(1).zfill(3) if match else None


def find_column(columns: pd.Index, expected: str) -> str:
    normalized = {normalize_text(column): str(column) for column in columns}
    if expected not in normalized:
        raise ValueError(
            f"Subject workbook is missing expected column {expected!r}; "
            f"found {list(columns)!r}"
        )
    return normalized[expected]


def load_age_groups(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing subject classification workbook: {path}")

    raw = pd.read_excel(path, sheet_name=0, dtype=object)
    if raw.empty or len(raw.columns) < 3:
        raise ValueError("Subject classification workbook has no usable data")

    first_column = raw.columns[0]
    subject_column = find_column(raw.columns, "subj #")
    age_column = find_column(raw.columns, "age")

    groups = pd.DataFrame({
        "subject_id": raw[subject_column].map(clean_subject_id),
        "age_group": np.where(
            raw[first_column].map(normalize_text).eq("n/a minor"),
            "minor",
            "older_than_minor",
        ),
        "reported_age": pd.to_numeric(raw[age_column], errors="coerce"),
        "classification_source_row": raw.index + 2,
    }).dropna(subset=["subject_id"])

    inconsistent = groups.groupby("subject_id")["age_group"].nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Conflicting minor classifications for subjects: "
            + ", ".join(inconsistent.index.tolist())
        )

    groups = groups.drop_duplicates("subject_id", keep="last")
    groups = groups.sort_values("subject_id").reset_index(drop=True)
    return groups


def first_nonmissing(series: pd.Series) -> object:
    values = series.dropna()
    return values.iloc[0] if not values.empty else np.nan


def hedges_g(minor: np.ndarray, older: np.ndarray) -> float:
    n_minor = len(minor)
    n_older = len(older)
    degrees_freedom = n_minor + n_older - 2
    if degrees_freedom <= 0:
        return np.nan

    pooled_variance = (
        (n_minor - 1) * np.var(minor, ddof=1)
        + (n_older - 1) * np.var(older, ddof=1)
    ) / degrees_freedom
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return np.nan

    cohen_d = (np.mean(minor) - np.mean(older)) / np.sqrt(pooled_variance)
    correction = 1 - 3 / (4 * (n_minor + n_older) - 9)
    return float(cohen_d * correction)


def compare_variable(frame: pd.DataFrame, variable: str, scope: str) -> dict[str, object]:
    minor = pd.to_numeric(
        frame.loc[frame["age_group"] == "minor", variable], errors="coerce"
    ).dropna().to_numpy(dtype=float)
    older = pd.to_numeric(
        frame.loc[frame["age_group"] == "older_than_minor", variable], errors="coerce"
    ).dropna().to_numpy(dtype=float)

    row: dict[str, object] = {
        "scope": scope,
        "variable": variable,
        "n_minor": len(minor),
        "n_older": len(older),
        "minor_mean": np.mean(minor) if len(minor) else np.nan,
        "minor_sd": np.std(minor, ddof=1) if len(minor) > 1 else np.nan,
        "minor_median": np.median(minor) if len(minor) else np.nan,
        "older_mean": np.mean(older) if len(older) else np.nan,
        "older_sd": np.std(older, ddof=1) if len(older) > 1 else np.nan,
        "older_median": np.median(older) if len(older) else np.nan,
        "minor_minus_older": (
            np.mean(minor) - np.mean(older) if len(minor) and len(older) else np.nan
        ),
        "test": "not_tested",
        "statistic": np.nan,
        "effect_size": np.nan,
        "effect_size_name": "",
        "p_value": np.nan,
        "test_note": "",
    }

    if len(minor) < 2 or len(older) < 2:
        row["test_note"] = "fewer_than_2_observations_in_a_group"
        return row

    observed = np.concatenate([minor, older])
    is_binary = set(np.unique(observed)).issubset({0.0, 1.0})
    if is_binary:
        table = np.array([
            [np.sum(minor == 1), np.sum(minor == 0)],
            [np.sum(older == 1), np.sum(older == 0)],
        ])
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        row.update({
            "test": "fisher_exact",
            "statistic": odds_ratio,
            "effect_size": odds_ratio,
            "effect_size_name": "odds_ratio",
            "p_value": p_value,
        })
        return row

    if np.var(minor, ddof=1) == 0 and np.var(older, ddof=1) == 0:
        row["test_note"] = "zero_variance_in_both_groups"
        return row

    result = ttest_ind(minor, older, equal_var=False, nan_policy="omit")
    row.update({
        "test": "welch_t_test",
        "statistic": float(result.statistic),
        "effect_size": hedges_g(minor, older),
        "effect_size_name": "hedges_g",
        "p_value": float(result.pvalue),
    })
    return row


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return adjusted

    count = len(valid)
    ranked = valid.to_numpy() * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted.loc[valid.index] = np.clip(ranked, 0, 1)
    return adjusted


def is_non_outcome(variable: str) -> bool:
    return (
        variable in NON_OUTCOME_COLUMNS
        or variable.endswith("_n_samples")
        or variable.endswith("_n_rows")
        or variable.endswith("_available")
    )


def build_availability_summary(joined: pd.DataFrame) -> pd.DataFrame:
    quality_columns = [
        column
        for column in joined.columns
        if column.endswith(("_n_samples", "_n_rows", "_available"))
    ]
    subject_rows = joined.sort_values("session").groupby("subject_id", as_index=False).agg(
        {**{column: first_nonmissing for column in quality_columns}, "age_group": "first"}
    )

    rows: list[dict[str, object]] = []
    for variable in quality_columns:
        for age_group, group in subject_rows.groupby("age_group"):
            if variable.endswith("_available"):
                available = pd.to_numeric(group[variable], errors="coerce").fillna(0).eq(1)
            else:
                available = pd.to_numeric(group[variable], errors="coerce").notna()
            rows.append({
                "variable": variable,
                "age_group": age_group,
                "n_subjects": len(group),
                "n_available": int(available.sum()),
                "proportion_available": float(available.mean()) if len(group) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["variable", "age_group"]).reset_index(drop=True)


def build_comparisons(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns = list(joined.select_dtypes(include=[np.number, "boolean"]).columns)

    comparison_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    grouped = joined.groupby("subject_id", sort=False)

    for variable in numeric_columns:
        if is_non_outcome(variable):
            excluded_rows.append({
                "variable": variable,
                "reason": "non_outcome_or_quality_control",
            })
            continue
        if joined[variable].notna().sum() == 0:
            excluded_rows.append({"variable": variable, "reason": "all_missing"})
            continue
        if joined[variable].nunique(dropna=True) <= 1:
            excluded_rows.append({"variable": variable, "reason": "constant"})
            continue

        max_within_subject_unique = grouped[variable].nunique(dropna=True).max()
        if max_within_subject_unique <= 1:
            subject_frame = grouped.agg({
                variable: first_nonmissing,
                "age_group": "first",
            }).reset_index()
            comparison_rows.append(compare_variable(subject_frame, variable, "subject_level"))
        else:
            for session, session_frame in joined.groupby("session", dropna=False):
                label = f"session_{int(session)}" if pd.notna(session) else "session_missing"
                comparison_rows.append(compare_variable(session_frame, variable, label))

    comparisons = pd.DataFrame(comparison_rows)
    comparisons["p_value_fdr_bh"] = benjamini_hochberg(comparisons["p_value"])
    comparisons["significant_fdr_0_05"] = comparisons["p_value_fdr_bh"].lt(0.05)
    comparisons = comparisons.sort_values(
        ["p_value_fdr_bh", "p_value", "scope", "variable"], na_position="last"
    ).reset_index(drop=True)
    excluded = pd.DataFrame(excluded_rows).sort_values(["reason", "variable"]).reset_index(drop=True)
    return comparisons, excluded


def write_report(
    path: Path,
    groups: pd.DataFrame,
    joined: pd.DataFrame,
    comparisons: pd.DataFrame,
    excluded: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:
    analyzed_subjects = joined.drop_duplicates("subject_id")
    analyzed_counts = analyzed_subjects["age_group"].value_counts()
    all_counts = groups["age_group"].value_counts()
    missing_from_data = groups.loc[
        ~groups["subject_id"].isin(joined["subject_id"]), "subject_id"
    ].tolist()
    unmatched_data = joined.loc[joined["age_group"].eq("unclassified"), "subject_id"].unique().tolist()
    tested = comparisons["p_value"].notna().sum()
    significant = comparisons["significant_fdr_0_05"].sum()
    age_discordance = groups.loc[
        groups["reported_age"].notna()
        & (
            (groups["age_group"].eq("minor") & groups["reported_age"].ge(18))
            | (groups["age_group"].eq("older_than_minor") & groups["reported_age"].lt(18))
        ),
        "subject_id",
    ].tolist()

    lines = [
        "MINOR VS OLDER SUBJECT COMPARISON",
        "=================================",
        "",
        "Classification rule",
        "-------------------",
        "A subject is classified as minor only when the first workbook column,",
        "after trimming whitespace and ignoring case, equals 'N/A Minor'.",
        "All other workbook-matched subjects are classified as older_than_minor.",
        "",
        "Cohort coverage",
        "---------------",
        f"Workbook minor-coded subjects: {all_counts.get('minor', 0)}",
        f"Workbook older subjects: {all_counts.get('older_than_minor', 0)}",
        f"Analyzed distinct minor subjects: {analyzed_counts.get('minor', 0)}",
        f"Analyzed distinct older subjects: {analyzed_counts.get('older_than_minor', 0)}",
        f"Analyzed subject-session rows: {len(joined)}",
        f"Workbook subjects without multimodal rows: {', '.join(missing_from_data) or 'none'}",
        f"Multimodal subjects without workbook classification: {', '.join(unmatched_data) or 'none'}",
        f"Workbook label/recorded-age discordances: {', '.join(age_discordance) or 'none'}",
        "",
        "Statistical audit",
        "-----------------",
        f"Valid hypothesis tests: {tested}",
        f"Comparisons significant after Benjamini-Hochberg FDR correction: {significant}",
        f"Non-outcome, quality-control, all-missing, or constant variables excluded: {len(excluded)}",
        "Continuous outcomes use Welch t-tests and Hedges' g (minor minus older).",
        "Binary outcomes use Fisher exact tests and odds ratios.",
        "Variables repeated unchanged across sessions are analyzed once per subject;",
        "variables that change within subject are analyzed separately by session.",
        "EEG/heart-rate row and sample counts and availability flags are audited",
        f"descriptively in a separate {len(availability)}-row availability table.",
        "",
        "Interpretation cautions",
        "-----------------------",
        "This is an exploratory observational comparison with a small minor cohort.",
        "A non-significant result does not establish equivalence, and effect estimates",
        "may be unstable. The workbook's first-column label is the source of truth even",
        "when reported age is missing. Subject 028 has no multimodal row and therefore",
        "does not duplicate subject 029 in the analysis.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not MULTIMODAL_CSV.exists():
        raise FileNotFoundError(
            f"Missing {MULTIMODAL_CSV}. Run build_multimodal_dataset.py first."
        )

    groups = load_age_groups(SUBJECT_WORKBOOK)
    multimodal = pd.read_csv(
        MULTIMODAL_CSV,
        dtype={"subject_id": str, "subjectId": str},
    )
    multimodal["subject_id"] = multimodal["subject_id"].astype(str).str.zfill(3)
    if multimodal.duplicated(["subject_id", "session"]).any():
        raise ValueError("Multimodal input contains duplicate subject/session rows")

    joined = multimodal.merge(groups, on="subject_id", how="left", validate="many_to_one")
    joined["age_group"] = joined["age_group"].fillna("unclassified")

    analyzed_subjects = joined.drop_duplicates("subject_id")
    analyzable_minors = analyzed_subjects["age_group"].eq("minor").sum()
    if analyzable_minors < MINIMUM_ANALYZABLE_MINORS:
        raise ValueError(
            f"Expected at least {MINIMUM_ANALYZABLE_MINORS} analyzable minor subjects; "
            f"found {analyzable_minors}."
        )
    if analyzed_subjects["age_group"].eq("unclassified").any():
        missing = analyzed_subjects.loc[
            analyzed_subjects["age_group"].eq("unclassified"), "subject_id"
        ].tolist()
        raise ValueError(f"Subjects missing workbook classification: {missing}")

    comparisons, excluded = build_comparisons(joined)
    availability = build_availability_summary(joined)

    MINOR_COMPARISON_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MINOR_COMPARISON_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    groups = groups.copy()
    groups["included_in_multimodal"] = groups["subject_id"].isin(joined["subject_id"])
    groups.to_csv(
        MINOR_COMPARISON_ANALYSIS_DATA_DIR / "subject_age_groups.csv", index=False
    )
    joined.to_csv(
        MINOR_COMPARISON_ANALYSIS_DATA_DIR / "minor_vs_older_joined.csv", index=False
    )
    comparisons.to_csv(
        MINOR_COMPARISON_OUTPUTS_DIR / "minor_vs_older_numeric_comparisons.csv", index=False
    )
    excluded.to_csv(
        MINOR_COMPARISON_OUTPUTS_DIR / "minor_vs_older_excluded_variables.csv", index=False
    )
    availability.to_csv(
        MINOR_COMPARISON_OUTPUTS_DIR / "minor_vs_older_data_availability.csv", index=False
    )
    report_path = MINOR_COMPARISON_OUTPUTS_DIR / "minor_vs_older_analysis_report.txt"
    write_report(report_path, groups, joined, comparisons, excluded, availability)

    counts = analyzed_subjects["age_group"].value_counts()
    print(f"Minor subjects analyzed: {counts.get('minor', 0)}")
    print(f"Older subjects analyzed: {counts.get('older_than_minor', 0)}")
    print(f"Comparison rows: {len(comparisons)}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
