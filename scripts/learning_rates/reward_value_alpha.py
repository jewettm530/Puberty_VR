"""Fit asymmetric reinforcement-learning models for current and previous CPLearn data.

This script follows the instructor-provided R model:
    RPE = reward - Q_chosen
    Q_new = Q_chosen + alpha_plus * RPE   when RPE >= 0
    Q_new = Q_chosen + alpha_minus * RPE  when RPE < 0

Choice probability is modeled with a logistic softmax over Q_face - Q_house,
and each experimental block starts with Q_face = Q_house = 0.5.

Supported input naming conventions
----------------------------------
Current study files:
    005_plearning_1.csv
    005_plearning_2.csv

Previous comparison files:
    5004_behav_data.csv
    5006_behav_data.csv

Practice trials are removed before fitting. Files with a ``blocktype`` column are
restricted to rows where blocktype == "experiment". If a file instead contains a
``practice`` column, rows marked as practice are removed.

Output
------
A single combined CSV is written to:
    outputs/learning_rates/tables/RL_parameter_results_combined.csv

The output includes an age column. Ages are read from:
    data/metadata/rl_subject_ages.csv

Missing ages are exported as NA. This lookup file can be filled in later and the
script rerun without changing the analysis code.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# -----------------------------------------------------------------------------
# Make shared project paths importable when this file is run directly.
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_paths import (  # noqa: E402
    METADATA_DIR,
    PLEARNING_DIR,
    PREVIOUS_SUBJECTS_DIR,
    TABLES_DIR,
)


# -----------------------------------------------------------------------------
# Input/output locations
# -----------------------------------------------------------------------------

CURRENT_DATA_DIR = PLEARNING_DIR
PREVIOUS_DATA_DIR = PREVIOUS_SUBJECTS_DIR
AGE_LOOKUP_FILE = METADATA_DIR / "rl_subject_ages.csv"
OUTPUT_FILE = TABLES_DIR / "RL_parameter_results_combined.csv"

CURRENT_FILENAME_PATTERN = re.compile(
    r"^(?P<subject>\d+)_plearning_(?P<session>\d+)\.csv$",
    flags=re.IGNORECASE,
)

PREVIOUS_FILENAME_PATTERN = re.compile(
    r"^(?P<subject>\d+)_behav_data\.csv$",
    flags=re.IGNORECASE,
)

STARTING_VALUES = [
    (0.20, 0.20, 2),
    (0.50, 0.50, 5),
    (0.80, 0.20, 5),
    (0.20, 0.80, 5),
    (0.80, 0.80, 10),
]

BOUNDS = [
    (0.001, 0.999),  # alpha_plus
    (0.001, 0.999),  # alpha_minus
    (0.01, 20.0),    # beta
]


# -----------------------------------------------------------------------------
# Reinforcement-learning model
# -----------------------------------------------------------------------------


def prepare_rl_blocks(data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert a prepared dataframe into ordered block arrays for fitting."""

    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for block_number in data["blockNumber"].unique():
        block_data = (
            data.loc[data["blockNumber"] == block_number]
            .sort_values("trialNumber")
        )
        choices = (block_data["chosen"].to_numpy() == "face").astype(np.int8)
        rewards = block_data["reward"].to_numpy(dtype=float)
        blocks.append((choices, rewards))
    return blocks


def rl_nll(par: np.ndarray, blocks: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Return negative log likelihood for the asymmetric RL model."""

    alpha_plus, alpha_minus, beta = par
    nll = 0.0

    # Each tuple contains one experimental block in trial order.
    for choices, rewards in blocks:
        # Reset expected values at the beginning of every block.
        Q_face = 0.5
        Q_house = 0.5

        for choice_is_face, reward_t in zip(choices, rewards):
            # Probability of choosing face.
            p_face = 1.0 / (1.0 + np.exp(-beta * (Q_face - Q_house)))
            p_face = float(np.clip(p_face, 1e-10, 1.0 - 1e-10))

            # Likelihood of the participant's actual choice.
            if choice_is_face:
                nll -= np.log(p_face)
                Q_chosen = Q_face
            else:
                nll -= np.log(1.0 - p_face)
                Q_chosen = Q_house

            # Reward prediction error: observed reward - expected reward.
            rpe = reward_t - Q_chosen

            # Asymmetric learning-rate update.
            if rpe >= 0:
                Q_new = Q_chosen + alpha_plus * rpe
            else:
                Q_new = Q_chosen + alpha_minus * rpe

            # Only the chosen option is updated.
            if choice_is_face:
                Q_face = Q_new
            else:
                Q_house = Q_new

    return float(nll)


# -----------------------------------------------------------------------------
# File parsing and trial filtering
# -----------------------------------------------------------------------------


def parse_input_file(file: Path, dataset: str) -> tuple[str, int | None]:
    """Extract subject and session from either supported filename convention."""

    if dataset == "current":
        match = CURRENT_FILENAME_PATTERN.match(file.name)
        if match is None:
            raise ValueError(f"Unexpected current-study filename: {file.name}")
        return match.group("subject"), int(match.group("session"))

    if dataset == "previous":
        match = PREVIOUS_FILENAME_PATTERN.match(file.name)
        if match is None:
            raise ValueError(f"Unexpected previous-study filename: {file.name}")
        return match.group("subject"), None

    raise ValueError(f"Unknown dataset label: {dataset}")


def remove_practice_trials(data: pd.DataFrame) -> pd.DataFrame:
    """Remove practice trials using the practice indicator present in the file."""

    data = data.copy()

    # The newly supplied previous-subject files use blocktype, and this is also
    # supported if future current files contain the same field.
    if "blocktype" in data.columns:
        blocktype = data["blocktype"].astype(str).str.strip().str.lower()
        return data.loc[blocktype == "experiment"].copy()

    # Preserve the instructor's alternate practice-column logic when blocktype
    # is unavailable. Handle common bool/numeric/string encodings of FALSE/0.
    if "practice" in data.columns:
        practice = data["practice"]
        practice_text = practice.astype(str).str.strip().str.lower()
        not_practice = (
            practice.isna()
            | practice.eq(False)
            | practice.eq(0)
            | practice_text.isin({"false", "0", "no", "n"})
        )
        return data.loc[not_practice].copy()

    # If neither indicator exists, leave the data unchanged, matching the
    # original R script's behavior when no practice variable is available.
    return data


def prepare_behavior_data(file: Path) -> pd.DataFrame:
    """Read one behavioral CSV and prepare usable experimental trials."""

    data = pd.read_csv(file)

    required_columns = {"blockNumber", "trialNumber", "chosen", "feedback"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            f"{file.name} is missing required columns: {missing_columns}"
        )

    # Remove practice before checking choice/feedback validity so practice rows
    # never contribute to the fitted model.
    data = remove_practice_trials(data)

    # Keep valid choices and feedback, as specified in the instructor's R code.
    data = data.loc[data["chosen"].isin(["face", "house"])].copy()
    data = data.loc[data["feedback"].isin(["gain", "loss"])].copy()

    # gain = 1, loss = 0
    data["reward"] = np.where(data["feedback"] == "gain", 1, 0)

    return (
        data.sort_values(["blockNumber", "trialNumber"])
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# Fit one subject/session file
# -----------------------------------------------------------------------------


def fit_one_file(file: Path, dataset: str) -> dict[str, object]:
    """Fit one subject/session file and return model parameters/statistics."""

    subject, session = parse_input_file(file, dataset)
    data = prepare_behavior_data(file)
    blocks = prepare_rl_blocks(data)

    fits = []
    for starting_point in STARTING_VALUES:
        try:
            fit = minimize(
                fun=rl_nll,
                x0=np.asarray(starting_point, dtype=float),
                args=(blocks,),
                method="L-BFGS-B",
                bounds=BOUNDS,
            )
            fits.append(fit)
        except Exception:
            # Same intent as tryCatch(..., error = function(e) NULL) in R.
            continue

    if not fits:
        return {
            "subject": subject,
            "dataset": dataset,
            "session": session,
            "usable_trials": len(data),
            "alpha_plus": np.nan,
            "alpha_minus": np.nan,
            "beta": np.nan,
            "NLL": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "convergence": np.nan,
            "alpha_plus_boundary": pd.NA,
            "alpha_minus_boundary": pd.NA,
            "beta_boundary": pd.NA,
            "any_boundary": pd.NA,
        }

    # Select the solution with the lowest negative log likelihood.
    best_fit = min(fits, key=lambda fit: fit.fun)

    alpha_plus, alpha_minus, beta = best_fit.x
    nll = float(best_fit.fun)
    n_trials = len(data)
    number_parameters = 3

    aic = 2 * number_parameters + 2 * nll
    bic = number_parameters * np.log(n_trials) + 2 * nll

    # Re-check boundaries from the final selected parameters using the same
    # thresholds as the instructor-provided R script.
    alpha_plus_boundary = bool(
        alpha_plus <= 0.0011 or alpha_plus >= 0.9989
    )
    alpha_minus_boundary = bool(
        alpha_minus <= 0.0011 or alpha_minus >= 0.9989
    )
    beta_boundary = bool(beta <= 0.011 or beta >= 19.99)
    any_boundary = bool(
        alpha_plus_boundary or alpha_minus_boundary or beta_boundary
    )

    return {
        "subject": subject,
        "dataset": dataset,
        "session": session,
        "usable_trials": n_trials,
        "alpha_plus": float(alpha_plus),
        "alpha_minus": float(alpha_minus),
        "beta": float(beta),
        "NLL": nll,
        "AIC": float(aic),
        "BIC": float(bic),
        # scipy.optimize status 0 corresponds to successful convergence.
        "convergence": int(best_fit.status),
        "alpha_plus_boundary": alpha_plus_boundary,
        "alpha_minus_boundary": alpha_minus_boundary,
        "beta_boundary": beta_boundary,
        "any_boundary": any_boundary,
    }


# -----------------------------------------------------------------------------
# Ages
# -----------------------------------------------------------------------------


def load_age_lookup() -> dict[str, float]:
    """Load subject ages; missing/unknown ages are intentionally left absent."""

    if not AGE_LOOKUP_FILE.exists():
        print(
            f"Warning: age lookup not found at {AGE_LOOKUP_FILE}. "
            "All ages will be NA."
        )
        return {}

    age_data = pd.read_csv(
        AGE_LOOKUP_FILE,
        dtype={"subject": str},
        keep_default_na=True,
    )

    required = {"subject", "age"}
    missing = sorted(required.difference(age_data.columns))
    if missing:
        raise ValueError(
            f"{AGE_LOOKUP_FILE.name} is missing required columns: {missing}"
        )

    age_data["subject"] = age_data["subject"].str.strip()
    age_data["age"] = pd.to_numeric(age_data["age"], errors="coerce")

    return {
        row.subject: float(row.age)
        for row in age_data.itertuples(index=False)
        if pd.notna(row.age)
    }


# -----------------------------------------------------------------------------
# Discover files
# -----------------------------------------------------------------------------


def discover_input_files() -> list[tuple[Path, str]]:
    """Find all supported current and previous behavioral files."""

    files: list[tuple[Path, str]] = []

    if CURRENT_DATA_DIR.exists():
        for file in CURRENT_DATA_DIR.glob("*.csv"):
            if CURRENT_FILENAME_PATTERN.match(file.name):
                files.append((file, "current"))

    if PREVIOUS_DATA_DIR.exists():
        for file in PREVIOUS_DATA_DIR.glob("*.csv"):
            if PREVIOUS_FILENAME_PATTERN.match(file.name):
                files.append((file, "previous"))

    def sort_key(item: tuple[Path, str]) -> tuple[int, int, int]:
        file, dataset = item
        subject, session = parse_input_file(file, dataset)
        dataset_order = 0 if dataset == "current" else 1
        session_order = int(session) if pd.notna(session) else 0
        return dataset_order, int(subject), session_order

    return sorted(files, key=sort_key)


# -----------------------------------------------------------------------------
# Boundary summary for the requested re-check
# -----------------------------------------------------------------------------


def print_boundary_summary(results: pd.DataFrame) -> None:
    """Print compact boundary counts for each dataset and the combined sample."""

    print("\nBoundary re-check:")

    for label, subset in [
        ("current", results.loc[results["dataset"] == "current"]),
        ("previous", results.loc[results["dataset"] == "previous"]),
        ("combined", results),
    ]:
        n = len(subset)
        if n == 0:
            continue

        alpha_plus_n = int(subset["alpha_plus_boundary"].fillna(False).sum())
        alpha_minus_n = int(subset["alpha_minus_boundary"].fillna(False).sum())
        beta_n = int(subset["beta_boundary"].fillna(False).sum())
        any_n = int(subset["any_boundary"].fillna(False).sum())

        print(
            f"  {label:8s}: n={n:2d} | "
            f"alpha+ {alpha_plus_n}/{n} | "
            f"alpha- {alpha_minus_n}/{n} | "
            f"beta {beta_n}/{n} | "
            f"any {any_n}/{n}"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    files = discover_input_files()

    if not files:
        raise FileNotFoundError(
            "No supported behavioral files were found.\n"
            f"Current data directory: {CURRENT_DATA_DIR}\n"
            f"Previous data directory: {PREVIOUS_DATA_DIR}\n\n"
            "Expected current filenames such as 005_plearning_1.csv and "
            "previous filenames such as 5004_behav_data.csv."
        )

    current_count = sum(dataset == "current" for _, dataset in files)
    previous_count = sum(dataset == "previous" for _, dataset in files)

    print(f"Found {len(files)} behavioral files:")
    print(f"  current:  {current_count}")
    print(f"  previous: {previous_count}")

    results = []
    for file, dataset in files:
        print(f"Fitting [{dataset}] {file.name}...")
        results.append(fit_one_file(file, dataset))

    rl_results = pd.DataFrame(results)

    # Preserve integer sessions while allowing NA for previous-study files.
    rl_results["session"] = pd.array(rl_results["session"], dtype="Int64")

    # Add known ages; unknown ages remain missing and are exported as NA.
    age_lookup = load_age_lookup()
    rl_results["age"] = rl_results["subject"].map(age_lookup)

    # Put age alongside the subject/session identifiers.
    column_order = [
        "subject",
        "dataset",
        "session",
        "age",
        "usable_trials",
        "alpha_plus",
        "alpha_minus",
        "beta",
        "NLL",
        "AIC",
        "BIC",
        "convergence",
        "alpha_plus_boundary",
        "alpha_minus_boundary",
        "beta_boundary",
        "any_boundary",
    ]
    rl_results = rl_results[column_order]

    # Sort current study first, then previous study, by numeric subject/session.
    rl_results["_dataset_order"] = rl_results["dataset"].map(
        {"current": 0, "previous": 1}
    )
    rl_results["_subject_numeric"] = pd.to_numeric(
        rl_results["subject"], errors="coerce"
    )
    rl_results = (
        rl_results.sort_values(
            ["_dataset_order", "_subject_numeric", "session"],
            na_position="last",
        )
        .drop(columns=["_dataset_order", "_subject_numeric"])
        .reset_index(drop=True)
    )

    print("\nRL parameter results:")
    print(rl_results.to_string(index=False))

    print_boundary_summary(rl_results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rl_results.to_csv(OUTPUT_FILE, index=False, na_rep="NA")

    print(f"\nCombined CSV saved to:\n{OUTPUT_FILE}")
    print("\nRL model fitting complete.")


if __name__ == "__main__":
    main()
