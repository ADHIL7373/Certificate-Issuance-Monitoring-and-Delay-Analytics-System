from pathlib import Path
import pandas as pd
import numpy as np
import joblib

print("=" * 70)
print("FINAL JURY APPLICATION PREDICTION")
print("=" * 70)

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(r"D:\Datascience Hackathon")

DATA_DIR = PROJECT_ROOT / "dataset"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "certificate-delay-intelligence"
    / "outputs"
    / "round2"
)

JURY_FILE = DATA_DIR / "unseen_jury_applications_5000.csv"

DELAY_MODEL_FILE = (
    OUTPUT_DIR /
    "random_forest_delay_model.joblib"
)

TAT_MODEL_FILE = (
    OUTPUT_DIR /
    "random_forest_remaining_tat_model.joblib"
)

# ==========================================================
# LOAD
# ==========================================================

print("\nLoading jury dataset...")

jury = pd.read_csv(
    JURY_FILE,
    low_memory=False
)

print("Jury dataset:", jury.shape)

# ==========================================================
# FEATURES
# ==========================================================

FEATURE_COLUMNS = [
    "service_code",
    "certificate_type",
    "district",
    "taluk",
    "submission_channel",
    "submission_month",
    "submission_day_of_week",
    "peak_period_flag",
    "current_stage",
    "application_age_days",
    "stage_age_days",
    "days_to_target",
    "target_tat_days",
    "required_document_count",
    "submitted_document_count",
    "document_completeness_pct",
    "missing_document_count",
    "verification_complexity",
    "resubmission_count",
    "queue_length",
    "pending_cases_department",
    "department_workload_index",
    "staff_availability_pct",
    "recent_backlog_growth_pct",
    "historical_stage_median_days",
    "historical_service_median_days",
    "target_tat_basis"
]

X_jury = jury[FEATURE_COLUMNS].copy()

# ==========================================================
# NUMERIC CLEANING
# ==========================================================

categorical_features = [
    "service_code",
    "certificate_type",
    "district",
    "taluk",
    "submission_channel",
    "submission_day_of_week",
    "current_stage",
    "target_tat_basis"
]

numeric_features = [
    col
    for col in FEATURE_COLUMNS
    if col not in categorical_features
]

for col in numeric_features:
    X_jury[col] = pd.to_numeric(
        X_jury[col],
        errors="coerce"
    )

# ==========================================================
# LOAD MODELS
# ==========================================================

print("\nLoading trained models...")

delay_model = joblib.load(
    DELAY_MODEL_FILE
)

tat_model = joblib.load(
    TAT_MODEL_FILE
)

print("Models loaded successfully.")

# ==========================================================
# DELAY PREDICTION
# ==========================================================

print("\nPredicting delay risk...")

delay_probability = (
    delay_model.predict_proba(X_jury)[:, 1]
)

delay_prediction = (
    delay_model.predict(X_jury)
)

# ==========================================================
# RISK LEVEL
# ==========================================================

def risk_level(probability):

    if probability >= 0.70:
        return "HIGH"

    elif probability >= 0.40:
        return "MEDIUM"

    else:
        return "LOW"


risk = [
    risk_level(p)
    for p in delay_probability
]

# ==========================================================
# TAT PREDICTION
# ==========================================================

print("Predicting remaining TAT...")

predicted_remaining_tat = (
    tat_model.predict(X_jury)
)

# Prevent negative predictions

predicted_remaining_tat = np.maximum(
    predicted_remaining_tat,
    0
)

predicted_total_tat = (
    jury["application_age_days"].fillna(0).values
    + predicted_remaining_tat
)

# ==========================================================
# FINAL RESULT
# ==========================================================

final = pd.DataFrame({

    "application_id":
        jury["application_id"],

    "service_code":
        jury["service_code"],

    "certificate_type":
        jury["certificate_type"],

    "district":
        jury["district"],

    "current_stage":
        jury["current_stage"],

    "application_age_days":
        jury["application_age_days"],

    "delay_probability":
        delay_probability,

    "risk_level":
        risk,

    "predicted_remaining_tat_days":
        predicted_remaining_tat,

    "predicted_total_tat_days":
        predicted_total_tat
})

# ==========================================================
# PRIORITY RANK
# ==========================================================

final = final.sort_values(
    "delay_probability",
    ascending=False
).reset_index(drop=True)

final["priority_rank"] = (
    np.arange(1, len(final) + 1)
)

# ==========================================================
# SAVE
# ==========================================================

FINAL_FILE = (
    OUTPUT_DIR /
    "FINAL_JURY_APPLICATION_RANKING.csv"
)

final.to_csv(
    FINAL_FILE,
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("FINAL JURY RESULTS")
print("=" * 70)

print("\nTotal applications:", len(final))

print("\nRisk distribution:")
print(
    final["risk_level"].value_counts()
)

print("\nAverage predicted remaining TAT:")
print(
    round(
        final["predicted_remaining_tat_days"].mean(),
        2
    ),
    "days"
)

print("\nTOP 20 PRIORITY APPLICATIONS")
print("=" * 70)

print(
    final.head(20).to_string(
        index=False
    )
)

print("\nFinal file saved:")
print(FINAL_FILE)

print("\n" + "=" * 70)
print("FINAL JURY PREDICTION COMPLETE")
print("=" * 70)