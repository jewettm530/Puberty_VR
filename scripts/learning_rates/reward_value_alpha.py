# ============================================================
# RL MODEL: alpha_plus, alpha_minus, beta
# Fits one asymmetric reinforcement-learning model
# per subject per session
#
# Python translation of instructor-provided R script
# ============================================================

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ------------------------------------------------------------
# 1. Set folder containing participant CSV files
# ------------------------------------------------------------

DATA_FOLDER = Path(
    "/Users/maddiemac/Puberty_VR/data/processed/plearning"
)

# ------------------------------------------------------------
# 2. Negative log-likelihood function
# ------------------------------------------------------------

def rl_nll(par, data):
    """
    Calculate negative log likelihood for the asymmetric
    reinforcement-learning model.

    Parameters
    ----------
    par : array-like
        par[0] = alpha_plus
        par[1] = alpha_minus
        par[2] = beta

    data : pandas.DataFrame
        Must contain:
        blockNumber
        trialNumber
        chosen
        reward

    Returns
    -------
    float
        Negative log likelihood.
    """

    alpha_plus = par[0]
    alpha_minus = par[1]
    beta = par[2]

    nll = 0.0

    # Fit each experimental block separately
    blocks = data["blockNumber"].unique()

    for b in blocks:

        block_data = (
            data[data["blockNumber"] == b]
            .sort_values("trialNumber")
        )

        # Reset expected values at beginning of each block
        Q_face = 0.5
        Q_house = 0.5

        for _, row in block_data.iterrows():

            choice_t = row["chosen"]
            reward_t = row["reward"]

            # ------------------------------------------
            # Choice probability
            # ------------------------------------------

            p_face = 1 / (
                1 + np.exp(
                    -beta * (Q_face - Q_house)
                )
            )

            # Avoid log(0)
            p_face = min(
                max(p_face, 1e-10),
                1 - 1e-10
            )

            # ------------------------------------------
            # Likelihood of actual choice
            # ------------------------------------------

            if choice_t == "face":

                nll = nll - np.log(p_face)
                Q_chosen = Q_face

            elif choice_t == "house":

                nll = nll - np.log(
                    1 - p_face
                )
                Q_chosen = Q_house

            else:

                continue

            # ------------------------------------------
            # Reward Prediction Error
            #
            # delta_t =
            # reward outcome - expected reward
            # ------------------------------------------

            RPE = reward_t - Q_chosen

            # ------------------------------------------
            # Update chosen value
            # ------------------------------------------

            if RPE >= 0:

                Q_new = (
                    Q_chosen
                    + alpha_plus * RPE
                )

            else:

                Q_new = (
                    Q_chosen
                    + alpha_minus * RPE
                )

            # ------------------------------------------
            # Store updated value
            # ------------------------------------------

            if choice_t == "face":

                Q_face = Q_new

            else:

                Q_house = Q_new

    return nll


# ------------------------------------------------------------
# 3. Function to fit ONE participant/session file
# ------------------------------------------------------------

def fit_one_subject(file):
    """
    Fit one RL model to one subject/session CSV file.

    Expected filename format:
        005_plearning_1.csv
        005_plearning_2.csv

    Returns
    -------
    dict
        Estimated RL parameters and model-fit statistics.
    """

    file = Path(file)

    # --------------------------------------------------------
    # Read participant file
    # --------------------------------------------------------

    dat = pd.read_csv(file)

    # --------------------------------------------------------
    # Get participant ID and session from filename
    #
    # Example:
    # 005_plearning_1.csv
    #
    # subject = 005
    # session = 1
    # --------------------------------------------------------

    match = re.match(
        r"^(\d+)_plearning_(\d+)\.csv$",
        file.name,
        flags=re.IGNORECASE
    )

    if match is None:
        raise ValueError(
            f"Filename does not match expected format: "
            f"{file.name}"
        )

    subject_id = match.group(1)
    session = int(match.group(2))

    # --------------------------------------------------------
    # Check required variables
    # --------------------------------------------------------

    required_columns = [
        "blockNumber",
        "trialNumber",
        "chosen",
        "feedback"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in dat.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{file.name} is missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Remove practice trials
    # --------------------------------------------------------
    #
    # The provided behavioral files contain the variable
    # "blocktype", with experimental trials coded as
    # "experiment".
    #
    # If a "practice" variable exists instead, follow the
    # logic from the original R script.
    # --------------------------------------------------------

    if "practice" in dat.columns:

        dat = dat[
            (dat["practice"] == False)
            | (dat["practice"] == 0)
        ].copy()

    elif "blocktype" in dat.columns:

        dat = dat[
            dat["blocktype"]
            .astype(str)
            .str.lower()
            == "experiment"
        ].copy()

    # --------------------------------------------------------
    # Remove unusable trials
    # --------------------------------------------------------

    # Keep valid choices
    dat = dat[
        dat["chosen"].isin(
            ["face", "house"]
        )
    ].copy()

    # Keep valid feedback
    dat = dat[
        dat["feedback"].isin(
            ["gain", "loss"]
        )
    ].copy()

    # --------------------------------------------------------
    # Code reward
    #
    # gain = 1
    # loss = 0
    # --------------------------------------------------------

    dat["reward"] = np.where(
        dat["feedback"] == "gain",
        1,
        0
    )

    dat = dat.sort_values(
        ["blockNumber", "trialNumber"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Run model from several starting points
    #
    # Multiple starting values reduce the chance of obtaining
    # a local minimum.
    # --------------------------------------------------------

    starting_values = [

        [0.20, 0.20, 2],

        [0.50, 0.50, 5],

        [0.80, 0.20, 5],

        [0.20, 0.80, 5],

        [0.80, 0.80, 10]

    ]

    # Parameter bounds:
    #
    # alpha_plus:
    #   0.001 to 0.999
    #
    # alpha_minus:
    #   0.001 to 0.999
    #
    # beta:
    #   0.01 to 20

    bounds = [

        (0.001, 0.999),

        (0.001, 0.999),

        (0.01, 20)

    ]

    fits = []

    for starting_point in starting_values:

        try:

            fit = minimize(

                fun=rl_nll,

                x0=np.array(
                    starting_point,
                    dtype=float
                ),

                args=(dat,),

                method="L-BFGS-B",

                bounds=bounds

            )

            fits.append(fit)

        except Exception:

            # Equivalent to the R tryCatch behavior:
            # failed fits are ignored.
            continue

    # --------------------------------------------------------
    # If every optimization attempt failed
    # --------------------------------------------------------

    if len(fits) == 0:

        return {

            "subject": subject_id,

            "session": session,

            "usable_trials": len(dat),

            "alpha_plus": np.nan,

            "alpha_minus": np.nan,

            "beta": np.nan,

            "NLL": np.nan,

            "AIC": np.nan,

            "BIC": np.nan,

            "convergence": np.nan,

            "alpha_plus_boundary": np.nan,

            "alpha_minus_boundary": np.nan,

            "beta_boundary": np.nan,

            "any_boundary": np.nan

        }

    # --------------------------------------------------------
    # Select model with lowest negative log likelihood
    # --------------------------------------------------------

    best_fit = min(
        fits,
        key=lambda fit: fit.fun
    )

    # --------------------------------------------------------
    # Extract parameters
    # --------------------------------------------------------

    alpha_plus = best_fit.x[0]

    alpha_minus = best_fit.x[1]

    beta = best_fit.x[2]

    NLL = best_fit.fun

    n_trials = len(dat)

    number_parameters = 3

    # --------------------------------------------------------
    # AIC
    #
    # AIC = 2k + 2*NLL
    # --------------------------------------------------------

    AIC_value = (
        2 * number_parameters
        + 2 * NLL
    )

    # --------------------------------------------------------
    # BIC
    #
    # BIC = k * ln(n) + 2*NLL
    # --------------------------------------------------------

    BIC_value = (
        number_parameters
        * np.log(n_trials)
        + 2 * NLL
    )

    # --------------------------------------------------------
    # Optimizer convergence
    #
    # SciPy status = 0 indicates successful convergence,
    # corresponding to convergence = 0 in R optim().
    # --------------------------------------------------------

    convergence = best_fit.status

    # --------------------------------------------------------
    # 5. Identify boundary estimates
    #
    # Same thresholds as instructor's R script
    # --------------------------------------------------------

    alpha_plus_boundary = (
        alpha_plus <= 0.0011
        or alpha_plus >= 0.9989
    )

    alpha_minus_boundary = (
        alpha_minus <= 0.0011
        or alpha_minus >= 0.9989
    )

    beta_boundary = (
        beta <= 0.011
        or beta >= 19.99
    )

    any_boundary = (
        alpha_plus_boundary
        or alpha_minus_boundary
        or beta_boundary
    )

    # --------------------------------------------------------
    # Return participant/session-level result
    # --------------------------------------------------------

    return {

        "subject": subject_id,

        "session": session,

        "usable_trials": n_trials,

        "alpha_plus": alpha_plus,

        "alpha_minus": alpha_minus,

        "beta": beta,

        "NLL": NLL,

        "AIC": AIC_value,

        "BIC": BIC_value,

        "convergence": convergence,

        "alpha_plus_boundary":
            alpha_plus_boundary,

        "alpha_minus_boundary":
            alpha_minus_boundary,

        "beta_boundary":
            beta_boundary,

        "any_boundary":
            any_boundary

    }


# ============================================================
# 6. Find all behavioral CSV files
# ============================================================

all_csv_files = list(
    DATA_FOLDER.glob(
        "*_plearning_*.csv"
    )
)

filename_pattern = re.compile(
    r"^\d+_plearning_\d+\.csv$",
    flags=re.IGNORECASE
)

files = [
    file
    for file in all_csv_files
    if filename_pattern.match(
        file.name
    )
]


def file_sort_key(file):

    parts = re.match(
        r"^(\d+)_plearning_(\d+)\.csv$",
        file.name,
        flags=re.IGNORECASE
    )

    return (
        int(parts.group(1)),
        int(parts.group(2))
    )


files = sorted(
    files,
    key=file_sort_key
)


if len(files) == 0:
    raise FileNotFoundError(
        f"No behavioral files were found in:\n"
        f"{DATA_FOLDER.resolve()}\n\n"
        f"Expected filenames like:\n"
        f"005_plearning_1.csv\n"
        f"005_plearning_2.csv"
    )


print(
    f"Found {len(files)} behavioral files."
)

for file in files:
    print(file.name)

# ============================================================
# 7. Fit every participant/session
# ============================================================

results = []

for file in files:

    print(
        f"Fitting {file.name}..."
    )

    result = fit_one_subject(file)

    results.append(result)


RL_results = pd.DataFrame(results)


# ============================================================
# 8. Sort by participant number and session
# ============================================================

if not RL_results.empty:

    RL_results[
        "subject_numeric"
    ] = pd.to_numeric(
        RL_results["subject"],
        errors="coerce"
    )

    RL_results = (
        RL_results
        .sort_values(
            [
                "subject_numeric",
                "session"
            ]
        )
        .drop(
            columns=[
                "subject_numeric"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 9. View results
# ============================================================

print("\nRL parameter results:")
print(RL_results)


# ============================================================
# 10. Export participant/session parameters to CSV
# ============================================================

csv_output = (
    DATA_FOLDER
    / "RL_parameter_results.csv"
)

RL_results.to_csv(
    csv_output,
    index=False
)

print(
    f"\nCSV saved to:\n{csv_output}"
)


# ============================================================
# 11. Export participant/session parameters to Excel
# ============================================================

excel_output = (
    DATA_FOLDER
    / "RL_parameter_results.xlsx"
)

RL_results.to_excel(
    excel_output,
    index=False
)

print(
    f"\nExcel file saved to:\n{excel_output}"
)


# ============================================================
# DONE
# ============================================================

print("\nRL model fitting complete.")