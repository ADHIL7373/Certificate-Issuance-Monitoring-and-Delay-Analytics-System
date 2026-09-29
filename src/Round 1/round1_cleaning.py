from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RAW_DATA = DATASET_ROOT / "certificate_applications_raw_50000.csv"
CLEAN_DATA = OUTPUT_DIR / "round1_cleaned_applications.csv"

CLEANING_REPORT = REPORT_DIR / "ROUND1_CLEANING_REPORT.md"


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("ROUND 1 — DATA CLEANING")
print("=" * 80)

df = pd.read_csv(
    RAW_DATA,
    low_memory=False
)

print(f"\nRaw rows    : {len(df):,}")
print(f"Raw columns : {len(df.columns)}")

original_rows = len(df)


# ============================================================
# 1. EXACT DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("1. DUPLICATE ANALYSIS")
print("=" * 80)

exact_duplicate_mask = df.duplicated(keep="first")
exact_duplicates = exact_duplicate_mask.sum()

print(f"Exact duplicate rows : {exact_duplicates:,}")

# Remove exact duplicates only.
df = df.loc[~exact_duplicate_mask].copy()

print(f"Rows after exact deduplication : {len(df):,}")


# ============================================================
# 2. DUPLICATE APPLICATION ID ANALYSIS
# ============================================================

duplicate_id_mask = df["application_id"].duplicated(keep=False)

duplicate_id_count = duplicate_id_mask.sum()
duplicate_id_values = (
    df.loc[duplicate_id_mask, "application_id"]
    .drop_duplicates()
    .tolist()
)

print(f"Rows belonging to duplicate application IDs : {duplicate_id_count}")
print(f"Duplicate application IDs                  : {len(duplicate_id_values)}")

if duplicate_id_values:

    duplicate_id_report = (
        df[df["application_id"].isin(duplicate_id_values)]
        .sort_values("application_id")
    )

    duplicate_id_report.to_csv(
        OUTPUT_DIR / "round1_duplicate_application_ids.csv",
        index=False
    )

    print(
        "\nIMPORTANT: Non-identical duplicate application IDs were preserved."
    )
    print(
        "Detailed report:",
        OUTPUT_DIR / "round1_duplicate_application_ids.csv"
    )


# ============================================================
# 3. CATEGORICAL NORMALIZATION
# ============================================================

print("\n" + "=" * 80)
print("2. CATEGORICAL NORMALIZATION")
print("=" * 80)

categorical_columns = [
    "service_code",
    "certificate_type",
    "district",
    "taluk",
    "submission_channel",
    "submission_day_of_week",
    "current_stage",
    "target_tat_basis"
]

categorical_report = []

for col in categorical_columns:

    before_unique = df[col].nunique(dropna=True)

    df[col] = df[col].apply(
        lambda x: x.strip().lower() if isinstance(x, str) else x
    )

    after_unique = df[col].nunique(dropna=True)

    categorical_report.append({
        "column": col,
        "unique_before": before_unique,
        "unique_after": after_unique
    })

    print(
        f"{col:35s}: "
        f"{before_unique:3d} → {after_unique:3d}"
    )


pd.DataFrame(categorical_report).to_csv(
    OUTPUT_DIR / "round1_categorical_cleaning_report.csv",
    index=False
)


# ============================================================
# 4. DATE NORMALIZATION
# ============================================================

print("\n" + "=" * 80)
print("3. DATE NORMALIZATION")
print("=" * 80)

# The raw dataset intentionally contains mixed date formats:
#   DD-MM-YYYY
#   YYYY/MM/DD
#
# Parse each known format explicitly instead of relying on
# automatic date inference.

original_submission_dates = df["submission_date"].astype("string")

parsed_dates = pd.Series(
    pd.NaT,
    index=df.index,
    dtype="datetime64[ns]"
)

# Format 1: DD-MM-YYYY
mask_dmy = original_submission_dates.str.match(
    r"^\d{2}-\d{2}-\d{4}$",
    na=False
)

parsed_dates.loc[mask_dmy] = pd.to_datetime(
    original_submission_dates.loc[mask_dmy],
    format="%d-%m-%Y",
    errors="coerce"
)

# Format 2: YYYY/MM/DD
mask_ymd_slash = original_submission_dates.str.match(
    r"^\d{4}/\d{2}/\d{2}$",
    na=False
)

parsed_dates.loc[mask_ymd_slash] = pd.to_datetime(
    original_submission_dates.loc[mask_ymd_slash],
    format="%Y/%m/%d",
    errors="coerce"
)

# Fallback parser for any other legitimate format.
remaining_mask = (
    original_submission_dates.notna()
    & parsed_dates.isna()
)

if remaining_mask.any():

    parsed_dates.loc[remaining_mask] = pd.to_datetime(
        original_submission_dates.loc[remaining_mask],
        errors="coerce"
    )

df["submission_date"] = parsed_dates

remaining_invalid_dates = df["submission_date"].isna().sum()

print(
    "DD-MM-YYYY parsed:",
    mask_dmy.sum()
)

print(
    "YYYY/MM/DD parsed:",
    mask_ymd_slash.sum()
)

print(
    "Remaining invalid dates:",
    remaining_invalid_dates
)

if remaining_invalid_dates > 0:

    invalid_dates_report = pd.DataFrame({
        "application_id": df.loc[
            df["submission_date"].isna(),
            "application_id"
        ],
        "original_submission_date": original_submission_dates.loc[
            df["submission_date"].isna()
        ]
    })

    invalid_dates_report.to_csv(
        OUTPUT_DIR / "round1_invalid_dates.csv",
        index=False
    )

    print(
        "Invalid-date report:",
        OUTPUT_DIR / "round1_invalid_dates.csv"
    )


# ============================================================
# 5. NUMERIC CONVERSION
# ============================================================

print("\n" + "=" * 80)
print("4. NUMERIC CONVERSION")
print("=" * 80)

numeric_columns = [
    "submission_month",
    "peak_period_flag",
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
    "historical_service_median_days"
]

numeric_conversion_report = []

for col in numeric_columns:

    before_missing = df[col].isna().sum()

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    after_missing = df[col].isna().sum()

    newly_invalid = max(
        after_missing - before_missing,
        0
    )

    numeric_conversion_report.append({
        "column": col,
        "original_missing": before_missing,
        "new_invalid_numeric": newly_invalid,
        "missing_after_conversion": after_missing
    })

    if newly_invalid > 0:
        print(
            f"{col:35s}: "
            f"{newly_invalid} invalid values → NaN"
        )


pd.DataFrame(numeric_conversion_report).to_csv(
    OUTPUT_DIR / "round1_numeric_conversion_report.csv",
    index=False
)


# ============================================================
# 6. RANGE VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("5. RANGE VALIDATION")
print("=" * 80)

range_rules = {
    "document_completeness_pct": (0, 100),
    "staff_availability_pct": (0, 100),
    "department_workload_index": (0, 100),
    "verification_complexity": (1, 5),
    "peak_period_flag": (0, 1)
}

range_report = []

for col, (lower, upper) in range_rules.items():

    invalid_mask = (
        df[col].notna()
        & (
            (df[col] < lower)
            | (df[col] > upper)
        )
    )

    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        df.loc[invalid_mask, col] = np.nan

    range_report.append({
        "column": col,
        "lower_bound": lower,
        "upper_bound": upper,
        "invalid_values_converted_to_nan": invalid_count
    })

    print(
        f"{col:35s}: {invalid_count} invalid values"
    )


pd.DataFrame(range_report).to_csv(
    OUTPUT_DIR / "round1_range_correction_report.csv",
    index=False
)


# ============================================================
# 7. NON-NEGATIVE VALIDATION
# ============================================================

non_negative_columns = [
    "required_document_count",
    "submitted_document_count",
    "missing_document_count",
    "resubmission_count",
    "queue_length",
    "pending_cases_department"
]

non_negative_report = []

for col in non_negative_columns:

    invalid_mask = (
        df[col].notna()
        & (df[col] < 0)
    )

    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        df.loc[invalid_mask, col] = np.nan

    non_negative_report.append({
        "column": col,
        "negative_values_converted_to_nan": invalid_count
    })


pd.DataFrame(non_negative_report).to_csv(
    OUTPUT_DIR / "round1_non_negative_report.csv",
    index=False
)


# ============================================================
# 8. LOGICAL CONSISTENCY
# ============================================================

print("\n" + "=" * 80)
print("6. LOGICAL CONSISTENCY")
print("=" * 80)

invalid_submitted = (
    df["submitted_document_count"].notna()
    & df["required_document_count"].notna()
    & (
        df["submitted_document_count"]
        > df["required_document_count"]
    )
)

print(
    "submitted > required:",
    invalid_submitted.sum()
)

if invalid_submitted.sum() > 0:
    df.loc[
        invalid_submitted,
        "submitted_document_count"
    ] = np.nan


invalid_missing_docs = (
    df["missing_document_count"].notna()
    & (
        df["missing_document_count"] < 0
    )
)

print(
    "negative missing documents:",
    invalid_missing_docs.sum()
)

if invalid_missing_docs.sum() > 0:
    df.loc[
        invalid_missing_docs,
        "missing_document_count"
    ] = np.nan


# ============================================================
# 9. TRACK MISSING VALUES BEFORE IMPUTATION
# ============================================================

before_imputation_missing = df.isna().copy()


# ============================================================
# 10. NUMERIC IMPUTATION
# ============================================================

print("\n" + "=" * 80)
print("7. NUMERIC IMPUTATION")
print("=" * 80)

imputation_columns = [
    "application_age_days",
    "stage_age_days",
    "days_to_target",
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
    "historical_service_median_days"
]

group_columns = [
    "certificate_type",
    "current_stage"
]

imputation_report = []

for col in imputation_columns:

    before = df[col].isna().sum()

    group_median = df.groupby(
        group_columns,
        dropna=False
    )[col].transform("median")

    df[col] = df[col].fillna(group_median)

    global_median = df[col].median()

    df[col] = df[col].fillna(global_median)

    after = df[col].isna().sum()

    imputed = before - after

    imputation_report.append({
        "column": col,
        "missing_before": before,
        "imputed_count": imputed,
        "missing_after": after,
        "global_median_fallback": global_median
    })

    print(
        f"{col:35s}: {imputed:4d} imputed"
    )


pd.DataFrame(imputation_report).to_csv(
    OUTPUT_DIR / "round1_imputation_report.csv",
    index=False
)


# ============================================================
# 11. CATEGORICAL IMPUTATION
# ============================================================

print("\nCategorical missing-value handling:")

categorical_imputation_report = []

for col in categorical_columns:

    before = df[col].isna().sum()

    if before > 0:

        mode_values = df[col].mode(dropna=True)

        if len(mode_values) > 0:
            fill_value = mode_values.iloc[0]
        else:
            fill_value = "unknown"

        df[col] = df[col].fillna(fill_value)

    after = df[col].isna().sum()

    categorical_imputation_report.append({
        "column": col,
        "missing_before": before,
        "missing_after": after
    })

    print(
        f"{col:35s}: {before} → {after}"
    )


pd.DataFrame(categorical_imputation_report).to_csv(
    OUTPUT_DIR / "round1_categorical_imputation_report.csv",
    index=False
)


# ============================================================
# 12. DATA QUALITY FLAG
# ============================================================

# 1 = at least one input field had missing data before imputation.
# 0 = no missing input fields before imputation.

tracking_columns = (
    imputation_columns
    + categorical_columns
)

df["data_quality_flag"] = (
    before_imputation_missing[tracking_columns]
    .any(axis=1)
    .astype(int)
)

print("\nData quality flags:")
print(
    df["data_quality_flag"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 13. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("8. FINAL VALIDATION")
print("=" * 80)

remaining_missing = df.isna().sum()
remaining_missing = remaining_missing[
    remaining_missing > 0
].sort_values(ascending=False)

print("\nRemaining missing values:")

if len(remaining_missing) == 0:
    print("NONE")
else:
    print(remaining_missing.to_string())


print(
    "\nExact duplicate rows remaining:",
    df.duplicated().sum()
)

print(
    "Duplicate application IDs remaining:",
    df["application_id"].duplicated().sum()
)


# ============================================================
# 14. SAVE
# ============================================================

df.to_csv(
    CLEAN_DATA,
    index=False
)


# ============================================================
# 15. REPORT
# ============================================================

report_lines = [
    "# Round 1 — Data Cleaning Report",
    "",
    "## Dataset Summary",
    "",
    f"- Original rows: {original_rows:,}",
    f"- Exact duplicate rows removed: {exact_duplicates:,}",
    f"- Rows after exact deduplication: {len(df):,}",
    f"- Final columns: {len(df.columns):,}",
    "",
    "## Duplicate Application IDs",
    "",
    f"- Duplicate application IDs remaining: {len(duplicate_id_values)}",
    "- Non-identical duplicate IDs were preserved for investigation.",
    "",
    "## Date Handling",
    "",
    f"- Remaining invalid dates: {remaining_invalid_dates}",
    "",
    "## Cleaning Operations",
    "",
    "- Removed exact duplicate rows.",
    "- Preserved non-identical duplicate application IDs for investigation.",
    "- Normalized categorical values.",
    "- Parsed mixed submission-date formats.",
    "- Converted invalid numeric tokens to NaN.",
    "- Validated domain ranges.",
    "- Validated non-negative operational counts.",
    "- Checked document-count logical consistency.",
    "- Imputed numeric values using certificate type + current stage medians.",
    "- Used global median as fallback.",
    "- Imputed categorical values using mode.",
    "- Added data_quality_flag.",
    "",
    "## Important Modeling Rules",
    "",
    "- Negative days_to_target values were retained.",
    "- Operationally extreme TAT values were not removed.",
    "- Outcome/leakage columns must not be used as model features.",
    "",
    "## Output",
    "",
    f"- Clean dataset: `{CLEAN_DATA}`",
    f"- Duplicate-ID investigation: `{OUTPUT_DIR / 'round1_duplicate_application_ids.csv'}`"
]

CLEANING_REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("DATA CLEANING COMPLETE")
print("=" * 80)

print(f"Clean dataset : {CLEAN_DATA}")
print(f"Final rows    : {len(df):,}")
print(f"Final columns : {len(df.columns):,}")