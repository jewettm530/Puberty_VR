"""Shared, portable paths for the Puberty & VR project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Top-level folders
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCUMENTATION_DIR = PROJECT_ROOT / "documentation"

# Raw and processed data
RAW_DIR = DATA_DIR / "raw"
RAW_SUBJECT_DIR = RAW_DIR / "subject_data"

PROCESSED_DIR = DATA_DIR / "processed"
PLEARNING_DIR = PROCESSED_DIR / "plearning"
HEART_RATE_PROCESSED_DIR = PROCESSED_DIR / "heart_rate"
EEG_PROCESSED_DIR = PROCESSED_DIR / "eeg"
SALIVA_PROCESSED_DIR = PROCESSED_DIR / "saliva"
MULTIMODAL_PROCESSED_DIR = PROCESSED_DIR / "multimodal"

# Reusable analysis data
ANALYSIS_DATA_DIR = DATA_DIR / "analysis_data"
LEARNING_ANALYSIS_DATA_DIR = ANALYSIS_DATA_DIR / "learning_rates"
HEART_RATE_ANALYSIS_DATA_DIR = ANALYSIS_DATA_DIR / "heart_rate"
EEG_ANALYSIS_DATA_DIR = ANALYSIS_DATA_DIR / "eeg"
SALIVA_ANALYSIS_DATA_DIR = ANALYSIS_DATA_DIR / "saliva"
MULTIMODAL_ANALYSIS_DATA_DIR = ANALYSIS_DATA_DIR / "multimodal"
MINOR_COMPARISON_ANALYSIS_DATA_DIR = ANALYSIS_DATA_DIR / "minor_comparison"

METADATA_DIR = DATA_DIR / "metadata"

# Generated outputs
LEARNING_OUTPUTS_DIR = OUTPUTS_DIR / "learning_rates"
FIGURES_DIR = LEARNING_OUTPUTS_DIR / "figures"
TABLES_DIR = LEARNING_OUTPUTS_DIR / "tables"
REPORTS_DIR = LEARNING_OUTPUTS_DIR / "reports"
HEART_RATE_OUTPUTS_DIR = OUTPUTS_DIR / "heart_rate"
EEG_OUTPUTS_DIR = OUTPUTS_DIR / "eeg"
SALIVA_OUTPUTS_DIR = OUTPUTS_DIR / "saliva"
MULTIMODAL_OUTPUTS_DIR = OUTPUTS_DIR / "multimodal"
MINOR_COMPARISON_OUTPUTS_DIR = OUTPUTS_DIR / "minor_comparison"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path stored relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_project_directories() -> None:
    """Create the standard directory structure without deleting existing data."""
    folders = [
        RAW_SUBJECT_DIR,
        PLEARNING_DIR,
        HEART_RATE_PROCESSED_DIR,
        EEG_PROCESSED_DIR,
        SALIVA_PROCESSED_DIR,
        MULTIMODAL_PROCESSED_DIR,
        LEARNING_ANALYSIS_DATA_DIR,
        HEART_RATE_ANALYSIS_DATA_DIR,
        EEG_ANALYSIS_DATA_DIR,
        SALIVA_ANALYSIS_DATA_DIR,
        MULTIMODAL_ANALYSIS_DATA_DIR,
        MINOR_COMPARISON_ANALYSIS_DATA_DIR,
        METADATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        REPORTS_DIR,
        HEART_RATE_OUTPUTS_DIR,
        EEG_OUTPUTS_DIR,
        SALIVA_OUTPUTS_DIR,
        MULTIMODAL_OUTPUTS_DIR,
        MINOR_COMPARISON_OUTPUTS_DIR,
        DOCUMENTATION_DIR,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


ensure_project_directories()
