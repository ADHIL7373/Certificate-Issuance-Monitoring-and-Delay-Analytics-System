from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("ROUND 2 - RANDOM FOREST DELAY RISK MODEL")
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
# 2. LOAD DATASET
# ==========================================================

print("\nLoading dataset...")

df = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

print(f"Original dataset shape: {df.shape}")

# Remove exact duplicate rows
duplicate_rows = df.duplicated().sum()

df = df.drop_duplicates().copy()

print(f"Exact duplicate rows removed: {duplicate_rows}")
print(f"Dataset after deduplication: {df.shape}")

# ==========================================================
# 3. TARGET
# ==========================================================

TARGET = "missed_target"

y = df[TARGET].copy()

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ==========================================================
# 4. FEATURE SELECTION
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

print(f"\nNumber of features: {len(FEATURE_COLUMNS)}")

# ==========================================================
# 5. CATEGORICAL FEATURES
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

# ==========================================================
# 6. NUMERIC FEATURES
# ==========================================================

numeric_features = [
    col
    for col in FEATURE_COLUMNS
    if col not in categorical_features
]

print(f"\nCategorical features: {len(categorical_features)}")
print(f"Numeric features    : {len(numeric_features)}")

# ==========================================================
# 7. CLEAN NUMERIC FEATURES
# ==========================================================

print("\nConverting numeric columns...")

for col in numeric_features:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

invalid_numeric_values = (
    X[numeric_features]
    .isna()
    .sum()
    .sum()
)

print(
    f"Invalid/missing numeric values converted to NaN: "
    f"{invalid_numeric_values}"
)

# ==========================================================
# 8. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# ==========================================================
# 9. PREPROCESSING
# ==========================================================

print("\nCreating preprocessing pipeline...")

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
# 10. RANDOM FOREST CLASSIFIER
# ==========================================================

print("\nCreating Random Forest model...")

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=None,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)

# ==========================================================
# 11. COMPLETE PIPELINE
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
# 12. TRAIN MODEL
# ==========================================================

print("\nTraining Random Forest...")
print("Please wait...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully.")

# ==========================================================
# 13. PREDICTION
# ==========================================================

print("\nGenerating predictions...")

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]

# ==========================================================
# 14. EVALUATION
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CLASSIFICATION RESULTS")
print("=" * 60)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==========================================================
# 15. RISK LEVEL
# ==========================================================

def risk_level(probability):

    if probability >= 0.70:

        return "HIGH"

    elif probability >= 0.40:

        return "MEDIUM"

    else:

        return "LOW"


# ==========================================================
# 16. CREATE RESULTS
# ==========================================================

results = X_test.copy()

results["actual_missed_target"] = (
    y_test.values
)

results["predicted_missed_target"] = (
    y_pred
)

results["delay_probability"] = (
    y_probability
)

results["risk_level"] = (
    results["delay_probability"]
    .apply(risk_level)
)

# Highest risk first

results = results.sort_values(
    "delay_probability",
    ascending=False
)

# Add priority rank

results["priority_rank"] = (
    np.arange(1, len(results) + 1)
)

# ==========================================================
# 17. SAVE PREDICTIONS
# ==========================================================

prediction_file = (
    OUTPUT_DIR
    / "delay_risk_predictions.csv"
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
# 18. SAVE MODEL
# ==========================================================

model_file = (
    OUTPUT_DIR
    / "random_forest_delay_model.joblib"
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
# 19. RISK SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("RISK LEVEL SUMMARY")
print("=" * 60)

print(
    results["risk_level"]
    .value_counts()
)

# ==========================================================
# 20. TOP 10 HIGH-RISK APPLICATIONS
# ==========================================================

print("\n" + "=" * 60)
print("TOP 10 HIGH-RISK APPLICATIONS")
print("=" * 60)

top10 = results[
    [
        "application_id"
        if "application_id" in results.columns
        else "service_code",
        "current_stage",
        "delay_probability",
        "risk_level",
        "priority_rank"
    ]
].head(10)

print(top10)

# ==========================================================
# COMPLETE
# ==========================================================

print("\n" + "=" * 60)
print("ROUND 2 DELAY RISK MODEL COMPLETE")
print("=" * 60)

# ==========================================================
# 21. FEATURE IMPORTANCE
# ==========================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE - MAJOR DELAY FACTORS")
print("=" * 60)

# Get transformed feature names
feature_names = (
    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

# Get Random Forest
rf_model = pipeline.named_steps["model"]

# Get importance values
importance_values = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

# Sort highest first
feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

# Save complete importance
importance_file = (
    OUTPUT_DIR
    / "feature_importance.csv"
)

feature_importance.to_csv(
    importance_file,
    index=False
)

# Show top 20
print("\nTOP 20 IMPORTANT FEATURES:")

print(
    feature_importance.head(20).to_string(
        index=False
    )
)

print("\nFeature importance saved:")
print(importance_file)

# ==========================================================
# 22. TOP 10 MAJOR DELAY FACTORS
# ==========================================================

top10_features = feature_importance.head(10)

print("\n" + "=" * 60)
print("TOP 10 MAJOR DELAY FACTORS")
print("=" * 60)

for rank, (_, row) in enumerate(
    top10_features.iterrows(),
    start=1
):

    print(
        f"{rank}. {row['feature']} "
        f"-> {row['importance']:.4f}"
    )