from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


print("=" * 60)
print("ROUND 2 - BOTTLENECK STAGE MODEL")
print("=" * 60)

PROJECT_ROOT = Path(r"D:\Datascience Hackathon")
DATA_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "certificate-delay-intelligence"
    / "outputs"
    / "round2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = DATA_DIR / "certificate_applications_raw_50000.csv"

# ----------------------------------------------------------
# Load
# ----------------------------------------------------------

df = pd.read_csv(RAW_FILE, low_memory=False)
df = df.drop_duplicates().copy()

print("\nDataset:", df.shape)

TARGET = "actual_bottleneck_stage"

# ----------------------------------------------------------
# Features
# ----------------------------------------------------------

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

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET].copy()

# ----------------------------------------------------------
# Feature types
# ----------------------------------------------------------

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
    col for col in FEATURE_COLUMNS
    if col not in categorical_features
]

# Convert numeric values
for col in numeric_features:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

# Remove missing target rows
valid = y.notna()

X = X.loc[valid].copy()
y = y.loc[valid].copy()

print("\nBottleneck classes:")
print(y.value_counts())

# ----------------------------------------------------------
# Train / Test split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining:", len(X_train))
print("Testing :", len(X_test))

# ----------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])

# ----------------------------------------------------------
# Random Forest
# ----------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=250,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\nTraining bottleneck model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n" + "=" * 60)
print("BOTTLENECK MODEL RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# ----------------------------------------------------------
# Save predictions
# ----------------------------------------------------------

results = X_test.copy()

results["actual_bottleneck_stage"] = y_test.values
results["predicted_bottleneck_stage"] = y_pred

output_file = (
    OUTPUT_DIR /
    "bottleneck_predictions.csv"
)

results.to_csv(
    output_file,
    index=False
)

# ----------------------------------------------------------
# Save model
# ----------------------------------------------------------

import joblib

model_file = (
    OUTPUT_DIR /
    "random_forest_bottleneck_model.joblib"
)

joblib.dump(
    pipeline,
    model_file
)

print("\nPrediction file:")
print(output_file)

print("\nModel file:")
print(model_file)

print("\nPredicted bottleneck distribution:")
print(
    pd.Series(y_pred)
    .value_counts()
)

print("\n" + "=" * 60)
print("BOTTLENECK MODEL COMPLETE")
print("=" * 60) 