from pathlib import Path

# Project root is the parent of the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset root relative to the project root
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"

# Specific dataset files
RAW_TRAIN_DATA = DATASET_ROOT / "certificate_applications_raw_50000.csv"
JURY_DATA = DATASET_ROOT / "unseen_jury_applications_5000.csv"
SCHEMA_DICTIONARY = DATASET_ROOT / "schema_dictionary.csv"
FEATURE_WHITELIST = DATASET_ROOT / "feature_whitelist.csv"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
