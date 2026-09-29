from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 60)
print("ROUND 2 - RANDOM FOREST REMAINING TAT MODEL")
print("=" * 60)


# ==========================================================
# 1. PATHS
# ==========================================================

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


# ==========================================================
# 2. LOAD DATA
# ==========================================================

print("\nLoading dataset...")

df = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

print("Original dataset shape:", df.shape)


# Remove exact duplicates

duplicate_rows = df.duplicated().sum()

df = df.drop_duplicates().copy()

print("Exact duplicate rows removed:", duplicate_rows)

print("Dataset after deduplication:", df.shape)


# ==========================================================
# 3. TARGET
# ==========================================================

TARGET = "actual_remaining_tat_days"

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

print("\nTarget:", TARGET)

print("\nTarget statistics:")

print(
    y.describe()
)


# ==========================================================
# 4. FEATURES
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


X = df[FEATURE_COLUMNS].copy()


print("\nNumber of features:", len(FEATURE_COLUMNS))


# ==========================================================
# 5. FEATURE TYPES
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


print("Categorical features:", len(categorical_features))

print("Numeric features:", len(numeric_features))


# ==========================================================
# 6. NUMERIC CONVERSION
# ==========================================================

print("\nConverting numeric columns...")

for col in numeric_features:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# Target conversion

y = pd.to_numeric(
    y,
    errors="coerce"
)


# Remove rows where target is missing

valid_rows = y.notna()

X = X.loc[valid_rows].copy()

y = y.loc[valid_rows].copy()


print(
    "\nRows available for TAT modeling:",
    len(X)
)


# ==========================================================
# 7. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))

print("Testing samples :", len(X_test))


# ==========================================================
# 8. PREPROCESSING
# ==========================================================

numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[

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
    ]
)


# ==========================================================
# 9. RANDOM FOREST REGRESSOR
# ==========================================================

model = RandomForestRegressor(

    n_estimators=300,

    max_depth=None,

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)


# ==========================================================
# 10. COMPLETE PIPELINE
# ==========================================================

pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)


# ==========================================================
# 11. TRAIN
# ==========================================================

print("\nTraining Random Forest Regressor...")

print("Please wait...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ==========================================================
# 12. PREDICTION
# ==========================================================

print("\nGenerating TAT predictions...")

y_pred = pipeline.predict(
    X_test
)


# ==========================================================
# 13. EVALUATION
# ==========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("REMAINING TAT PREDICTION RESULTS")
print("=" * 60)

print(
    f"\nMAE  : {mae:.4f} days"
)

print(
    f"RMSE : {rmse:.4f} days"
)

print(
    f"R²   : {r2:.4f}"
)


# ==========================================================
# 14. PREDICTION OUTPUT
# ==========================================================

results = X_test.copy()

results["actual_remaining_tat_days"] = (
    y_test.values
)

results["predicted_remaining_tat_days"] = (
    y_pred
)

results["prediction_error_days"] = (
    results["predicted_remaining_tat_days"]
    - results["actual_remaining_tat_days"]
)


# Predicted total TAT

results["predicted_total_tat_days"] = (
    results["application_age_days"]
    + results["predicted_remaining_tat_days"]
)


# ==========================================================
# 15. SAVE RESULTS
# ==========================================================

prediction_file = (
    OUTPUT_DIR
    / "remaining_tat_predictions.csv"
)

results.to_csv(
    prediction_file,
    index=False
)


print("\nPrediction file saved:")

print(
    prediction_file
)


# ==========================================================
# 16. SAVE MODEL
# ==========================================================

model_file = (
    OUTPUT_DIR
    / "random_forest_remaining_tat_model.joblib"
)

joblib.dump(
    pipeline,
    model_file
)


print("\nModel saved:")

print(
    model_file
)


# ==========================================================
# 17. SAMPLE PREDICTIONS
# ==========================================================

print("\n" + "=" * 60)
print("SAMPLE TAT PREDICTIONS")
print("=" * 60)

print(
    results[
        [
            "current_stage",
            "application_age_days",
            "actual_remaining_tat_days",
            "predicted_remaining_tat_days",
            "predicted_total_tat_days"
        ]
    ].head(10).to_string(index=False)
)


# ==========================================================
# COMPLETE
# ==========================================================

print("\n" + "=" * 60)
print("ROUND 2 REMAINING TAT MODEL COMPLETE")
print("=" * 60)