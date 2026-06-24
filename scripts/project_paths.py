from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------------
# Top-level folders
# -------------------------
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCUMENTATION_DIR = PROJECT_ROOT / "documentation"

# -------------------------
# Data folders
# -------------------------
RAW_DIR = DATA_DIR / "raw"
RAW_SUBJECT_DIR = RAW_DIR / "subject_data"

PROCESSED_DIR = DATA_DIR / "processed"
PLEARNING_DIR = PROCESSED_DIR / "plearning"

ANALYSIS_DATA_DIR = DATA_DIR / "analysis_data"
LEARNING_ANALYSIS_DATA_DIR = ANALYSIS_DATA_DIR / "learning_rates"

METADATA_DIR = DATA_DIR / "metadata"

# -------------------------
# Output folders
# -------------------------
LEARNING_OUTPUTS_DIR = OUTPUTS_DIR / "learning_rates"
FIGURES_DIR = LEARNING_OUTPUTS_DIR / "figures"
TABLES_DIR = LEARNING_OUTPUTS_DIR / "tables"
REPORTS_DIR = LEARNING_OUTPUTS_DIR / "reports"

# -------------------------
# Make sure folders exist
# -------------------------
for folder in [
    RAW_SUBJECT_DIR,
    PLEARNING_DIR,
    LEARNING_ANALYSIS_DATA_DIR,
    METADATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    DOCUMENTATION_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)