from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DATA = DATASET_ROOT / "certificate_applications_raw_50000.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "application_id",
    "service_code",
    "certificate_type",
    "district",
    "taluk",
    "submission_channel",
    "submission_date",
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
    "actual_initial_scrutiny_days",
    "actual_document_verification_days",
    "actual_department_review_days",
    "actual_approval_days",
    "actual_certificate_issuance_days",
    "actual_total_tat_days",
    "actual_remaining_tat_days",
    "missed_target",
    "delay_overrun_days",
    "actual_bottleneck_stage",
    "target_tat_basis",
]


# ============================================================
# LEAKAGE / TARGET COLUMNS
# ============================================================

LEAKAGE_COLUMNS = [
    "application_id",
    "actual_initial_scrutiny_days",
    "actual_document_verification_days",
    "actual_department_review_days",
    "actual_approval_days",
    "actual_certificate_issuance_days",
    "actual_total_tat_days",
    "actual_remaining_tat_days",
    "missed_target",
    "delay_overrun_days",
    "actual_bottleneck_stage",
]


# ============================================================
# NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
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
    "historical_service_median_days",
]


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

CATEGORICAL_COLUMNS = [
    "service_code",
    "certificate_type",
    "district",
    "taluk",
    "submission_channel",
    "submission_day_of_week",
    "current_stage",
    "target_tat_basis",
]


# ============================================================
# MAIN AUDIT
# ============================================================

def run_quality_audit():

    print("=" * 80)
    print("ROUND 1 — DATA QUALITY AUDIT")
    print("=" * 80)

    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        RAW_DATA,
        low_memory=False
    )

    print("\nDataset loaded successfully.")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")


    # --------------------------------------------------------
    # COLUMN AUDIT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("1. COLUMN AUDIT")
    print("=" * 80)

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    extra_columns = [
        col for col in df.columns
        if col not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        print("\nMissing expected columns:")
        for col in missing_columns:
            print(f"  - {col}")
    else:
        print("All expected columns are present.")

    if extra_columns:
        print("\nUnexpected extra columns:")
        for col in extra_columns:
            print(f"  - {col}")
    else:
        print("No unexpected columns found.")


    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("2. DATA TYPE AUDIT")
    print("=" * 80)

    dtype_table = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values
    })

    print(dtype_table.to_string(index=False))


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("3. MISSING VALUE AUDIT")
    print("=" * 80)

    missing_count = df.isna().sum()

    missing_table = pd.DataFrame({
        "column": missing_count.index,
        "missing_count": missing_count.values,
        "missing_percentage": (
            missing_count.values / len(df) * 100
        ).round(2)
    })

    missing_table = missing_table[
        missing_table["missing_count"] > 0
    ].sort_values(
        "missing_count",
        ascending=False
    )

    if len(missing_table) == 0:
        print("No missing values detected.")
    else:
        print(
            missing_table.to_string(index=False)
        )

    missing_table.to_csv(
        OUTPUT_DIR / "round1_missing_values.csv",
        index=False
    )


    # --------------------------------------------------------
    # DUPLICATE AUDIT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("4. DUPLICATE AUDIT")
    print("=" * 80)

    exact_duplicates = int(
        df.duplicated().sum()
    )

    duplicate_ids = int(
        df["application_id"].duplicated().sum()
    )

    unique_ids = int(
        df["application_id"].nunique()
    )

    print(f"Exact duplicate rows      : {exact_duplicates:,}")
    print(f"Duplicate application IDs  : {duplicate_ids:,}")
    print(f"Unique application IDs     : {unique_ids:,}")


    # --------------------------------------------------------
    # INVALID / CORRUPTED NUMERIC VALUES
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("5. NUMERIC TYPE AUDIT")
    print("=" * 80)

    numeric_audit = []

    for column in NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        original = df[column]

        converted = pd.to_numeric(
            original,
            errors="coerce"
        )

        invalid_mask = (
            original.notna()
            & converted.isna()
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        numeric_audit.append({
            "column": column,
            "original_dtype": str(original.dtype),
            "invalid_numeric_values": invalid_count,
            "invalid_percentage": round(
                invalid_count / len(df) * 100,
                4
            )
        })

        if invalid_count > 0:
            print(
                f"{column}: "
                f"{invalid_count:,} invalid numeric values"
            )

    numeric_audit_df = pd.DataFrame(
        numeric_audit
    )

    numeric_audit_df.to_csv(
        OUTPUT_DIR / "round1_numeric_type_audit.csv",
        index=False
    )


    # --------------------------------------------------------
    # CATEGORICAL CONSISTENCY AUDIT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("6. CATEGORICAL CONSISTENCY AUDIT")
    print("=" * 80)

    categorical_audit = []

    for column in CATEGORICAL_COLUMNS:

        if column not in df.columns:
            continue

        raw_unique = (
            df[column]
            .dropna()
            .astype(str)
            .nunique()
        )

        normalized_unique = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .nunique()
        )

        categorical_audit.append({
            "column": column,
            "raw_unique_values": raw_unique,
            "normalized_unique_values": normalized_unique,
            "possible_inconsistency": (
                raw_unique > normalized_unique
            )
        })

        print(
            f"{column}: "
            f"{raw_unique} raw → "
            f"{normalized_unique} normalized"
        )

    categorical_audit_df = pd.DataFrame(
        categorical_audit
    )

    categorical_audit_df.to_csv(
        OUTPUT_DIR / "round1_categorical_audit.csv",
        index=False
    )


    # --------------------------------------------------------
    # DATE AUDIT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("7. DATE AUDIT")
    print("=" * 80)

    date_series = df["submission_date"]

    parsed_dates = pd.to_datetime(
        date_series,
        errors="coerce",
        format="mixed"
    )

    invalid_dates = int(
        (
            date_series.notna()
            & parsed_dates.isna()
        ).sum()
    )

    print(
        f"Invalid/unparseable dates: "
        f"{invalid_dates:,}"
    )

    print(
        f"Valid dates: "
        f"{parsed_dates.notna().sum():,}"
    )


    # --------------------------------------------------------
    # TARGET AUDIT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("8. TARGET AUDIT")
    print("=" * 80)

    if "missed_target" in df.columns:

        target_counts = (
            df["missed_target"]
            .value_counts()
            .sort_index()
        )

        target_percent = (
            df["missed_target"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )

        target_table = pd.DataFrame({
            "count": target_counts,
            "percentage": target_percent.round(2)
        })

        print(
            target_table.to_string()
        )

        target_table.to_csv(
            OUTPUT_DIR / "round1_target_distribution.csv"
        )


    # --------------------------------------------------------
    # LEAKAGE INVENTORY
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("9. LEAKAGE INVENTORY")
    print("=" * 80)

    leakage_found = [
        col for col in LEAKAGE_COLUMNS
        if col in df.columns
    ]

    print(
        "The following columns exist in the raw data "
        "but MUST NOT be used as model input:"
    )

    for column in leakage_found:
        print(f"  - {column}")


    # --------------------------------------------------------
    # BASIC RANGE AUDIT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("10. DOMAIN RANGE AUDIT")
    print("=" * 80)

    range_checks = {}

    def check_range(column, minimum=None, maximum=None):

        if column not in df.columns:
            return

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid = pd.Series(
            False,
            index=df.index
        )

        if minimum is not None:
            invalid |= values < minimum

        if maximum is not None:
            invalid |= values > maximum

        count = int(
            invalid.sum()
        )

        range_checks[column] = count

        print(
            f"{column}: "
            f"{count:,} outside expected range"
        )

    check_range(
        "document_completeness_pct",
        0,
        100
    )

    check_range(
        "staff_availability_pct",
        0,
        100
    )

    check_range(
        "department_workload_index",
        0,
        100
    )

    check_range(
        "verification_complexity",
        1,
        5
    )

    check_range(
        "peak_period_flag",
        0,
        1
    )

    check_range(
        "submitted_document_count",
        0,
        None
    )

    check_range(
        "required_document_count",
        0,
        None
    )

    check_range(
        "missing_document_count",
        0,
        None
    )

    check_range(
        "resubmission_count",
        0,
        None
    )

    check_range(
        "queue_length",
        0,
        None
    )

    check_range(
        "pending_cases_department",
        0,
        None
    )


    # --------------------------------------------------------
    # LOGICAL CONSISTENCY
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("11. LOGICAL CONSISTENCY AUDIT")
    print("=" * 80)

    required = pd.to_numeric(
        df["required_document_count"],
        errors="coerce"
    )

    submitted = pd.to_numeric(
        df["submitted_document_count"],
        errors="coerce"
    )

    missing_docs = pd.to_numeric(
        df["missing_document_count"],
        errors="coerce"
    )

    completeness = pd.to_numeric(
        df["document_completeness_pct"],
        errors="coerce"
    )

    checks = {
        "submitted > required":
            (submitted > required).sum(),

        "negative missing documents":
            (missing_docs < 0).sum(),

        "completeness outside 0-100":
            (
                (completeness < 0)
                | (completeness > 100)
            ).sum(),
    }

    for name, count in checks.items():
        print(
            f"{name}: {int(count):,}"
        )


    # --------------------------------------------------------
    # SAVE COMPLETE AUDIT
    # --------------------------------------------------------

    audit_summary = pd.DataFrame({
        "metric": [
            "raw_rows",
            "raw_columns",
            "exact_duplicate_rows",
            "duplicate_application_ids",
            "unique_application_ids",
            "total_missing_cells",
            "invalid_submission_dates",
            "leakage_columns_found",
        ],
        "value": [
            len(df),
            len(df.columns),
            exact_duplicates,
            duplicate_ids,
            unique_ids,
            int(df.isna().sum().sum()),
            invalid_dates,
            len(leakage_found),
        ]
    })

    audit_summary.to_csv(
        OUTPUT_DIR / "round1_quality_audit_summary.csv",
        index=False
    )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "ROUND1_DATA_QUALITY_AUDIT.md"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "# Round 1 — Data Quality Audit\n\n"
        )

        report.write(
            "## Dataset\n\n"
        )

        report.write(
            f"- Raw rows: {len(df):,}\n"
        )

        report.write(
            f"- Columns: {len(df.columns)}\n"
        )

        report.write(
            f"- Unique applications: {unique_ids:,}\n"
        )

        report.write(
            f"- Exact duplicate rows: "
            f"{exact_duplicates:,}\n"
        )

        report.write(
            f"- Duplicate application IDs: "
            f"{duplicate_ids:,}\n"
        )

        report.write(
            f"- Total missing cells: "
            f"{int(df.isna().sum().sum()):,}\n"
        )

        report.write(
            f"- Invalid dates: "
            f"{invalid_dates:,}\n"
        )

        report.write(
            "\n## Leakage Columns\n\n"
        )

        for column in leakage_found:
            report.write(
                f"- `{column}`\n"
            )

        report.write(
            "\nThese columns must be excluded from "
            "the predictive feature matrix.\n"
        )


    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DATA QUALITY AUDIT COMPLETE")
    print("=" * 80)

    print(
        f"Audit report: {report_path}"
    )

    print(
        f"Audit outputs saved in: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    run_quality_audit()