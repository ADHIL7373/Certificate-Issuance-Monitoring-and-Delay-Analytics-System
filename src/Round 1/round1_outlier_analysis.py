from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

CLEAN_DATA = OUTPUT_DIR / "round1_cleaned_applications.csv"

OUTLIER_REPORT = OUTPUT_DIR / "round1_outlier_report.csv"
OUTLIER_SUMMARY = REPORT_DIR / "ROUND1_OUTLIER_REPORT.md"


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("ROUND 1 — OUTLIER ANALYSIS")
print("=" * 80)

df = pd.read_csv(
    CLEAN_DATA,
    low_memory=False
)

print(f"\nRows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# ANALYSIS COLUMNS
# ============================================================

analysis_columns = [
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
    "delay_overrun_days"
]


# ============================================================
# 1. IQR OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 80)
print("1. IQR OUTLIER DETECTION")
print("=" * 80)

results = []

for col in analysis_columns:

    if col not in df.columns:
        continue

    series = pd.to_numeric(
        df[col],
        errors="coerce"
    ).dropna()

    if len(series) == 0:
        continue

    q1 = series.quantile(0.25)
    median = series.quantile(0.50)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    extreme_lower = q1 - 3 * iqr
    extreme_upper = q3 + 3 * iqr

    outlier_mask = (
        (series < lower_bound)
        | (series > upper_bound)
    )

    extreme_mask = (
        (series < extreme_lower)
        | (series > extreme_upper)
    )

    outlier_count = int(outlier_mask.sum())
    extreme_count = int(extreme_mask.sum())

    outlier_percentage = (
        outlier_count / len(series) * 100
    )

    results.append({
        "column": col,
        "count": len(series),
        "min": series.min(),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": series.max(),
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "iqr_outlier_count": outlier_count,
        "iqr_outlier_percentage": outlier_percentage,
        "extreme_outlier_count": extreme_count
    })

    print(
        f"{col:40s} "
        f"outliers={outlier_count:6,d} "
        f"({outlier_percentage:6.2f}%) "
        f"extreme={extreme_count:6,d}"
    )


outlier_df = pd.DataFrame(results)

outlier_df.to_csv(
    OUTLIER_REPORT,
    index=False
)


# ============================================================
# 2. DOMAIN-SPECIFIC CHECKS
# ============================================================

print("\n" + "=" * 80)
print("2. DOMAIN-SPECIFIC CHECKS")
print("=" * 80)

domain_checks = []


def add_check(
    name,
    mask,
    interpretation
):
    count = int(mask.sum())

    percentage = (
        count / len(df) * 100
    )

    domain_checks.append({
        "check": name,
        "count": count,
        "percentage": percentage,
        "interpretation": interpretation
    })

    print(
        f"{name:45s}: "
        f"{count:6,d} "
        f"({percentage:6.2f}%)"
    )


# Negative application age is suspicious.
add_check(
    "Negative application_age_days",
    df["application_age_days"] < 0,
    "Potential data-quality issue"
)


# Negative stage age is suspicious.
add_check(
    "Negative stage_age_days",
    df["stage_age_days"] < 0,
    "Potential data-quality issue"
)


# Negative days_to_target is valid.
add_check(
    "Negative days_to_target",
    df["days_to_target"] < 0,
    "Operationally valid: target date has been crossed"
)


# Application age should normally be >= stage age.
add_check(
    "stage_age_days > application_age_days",
    df["stage_age_days"] > df["application_age_days"],
    "Potential logical inconsistency"
)


# Submitted documents should not exceed required documents.
add_check(
    "Submitted documents > required documents",
    df["submitted_document_count"]
    > df["required_document_count"],
    "Potential data-quality issue"
)


# Missing documents cannot exceed required documents.
add_check(
    "Missing documents > required documents",
    df["missing_document_count"]
    > df["required_document_count"],
    "Potential data-quality issue"
)


# Completeness should be between 0 and 100.
add_check(
    "Completeness outside 0-100",
    (
        (df["document_completeness_pct"] < 0)
        | (df["document_completeness_pct"] > 100)
    ),
    "Potential data-quality issue"
)


# Staff availability should be between 0 and 100.
add_check(
    "Staff availability outside 0-100",
    (
        (df["staff_availability_pct"] < 0)
        | (df["staff_availability_pct"] > 100)
    ),
    "Potential data-quality issue"
)


# Workload should be between 0 and 100.
add_check(
    "Department workload outside 0-100",
    (
        (df["department_workload_index"] < 0)
        | (df["department_workload_index"] > 100)
    ),
    "Potential data-quality issue"
)


# Verification complexity should be 1-5.
add_check(
    "Verification complexity outside 1-5",
    (
        (df["verification_complexity"] < 1)
        | (df["verification_complexity"] > 5)
    ),
    "Potential data-quality issue"
)


# Peak period flag should be binary.
add_check(
    "Invalid peak period flag",
    ~df["peak_period_flag"].isin([0, 1]),
    "Potential data-quality issue"
)


domain_df = pd.DataFrame(domain_checks)

domain_df.to_csv(
    OUTPUT_DIR / "round1_domain_outlier_checks.csv",
    index=False
)


# ============================================================
# 3. HIGH-TAT OPERATIONAL EXTREMES
# ============================================================

print("\n" + "=" * 80)
print("3. OPERATIONAL TAT EXTREMES")
print("=" * 80)

tat_columns = [
    "actual_initial_scrutiny_days",
    "actual_document_verification_days",
    "actual_department_review_days",
    "actual_approval_days",
    "actual_certificate_issuance_days",
    "actual_total_tat_days",
    "actual_remaining_tat_days",
    "delay_overrun_days"
]

tat_summary = []

for col in tat_columns:

    series = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    negative_count = int(
        (series < 0).sum()
    )

    zero_count = int(
        (series == 0).sum()
    )

    p95 = series.quantile(0.95)
    p99 = series.quantile(0.99)

    high_extreme_count = int(
        (series > p99).sum()
    )

    tat_summary.append({
        "column": col,
        "min": series.min(),
        "median": series.median(),
        "p95": p95,
        "p99": p99,
        "max": series.max(),
        "negative_count": negative_count,
        "zero_count": zero_count,
        "above_p99_count": high_extreme_count
    })

    print(
        f"{col:40s} "
        f"median={series.median():8.2f} "
        f"p95={p95:8.2f} "
        f"p99={p99:8.2f} "
        f"max={series.max():8.2f}"
    )


tat_df = pd.DataFrame(tat_summary)

tat_df.to_csv(
    OUTPUT_DIR / "round1_tat_extreme_analysis.csv",
    index=False
)


# ============================================================
# 4. TOP EXTREME APPLICATIONS
# ============================================================

print("\n" + "=" * 80)
print("4. TOP OPERATIONAL EXTREMES")
print("=" * 80)

top_tat = (
    df[
        [
            "application_id",
            "certificate_type",
            "district",
            "current_stage",
            "actual_total_tat_days",
            "target_tat_days",
            "missed_target",
            "actual_bottleneck_stage"
        ]
    ]
    .sort_values(
        "actual_total_tat_days",
        ascending=False
    )
    .head(20)
)

top_tat.to_csv(
    OUTPUT_DIR / "round1_top_20_tat_extremes.csv",
    index=False
)

print(
    top_tat.to_string(index=False)
)


# ============================================================
# 5. OUTLIER DECISION
# ============================================================

print("\n" + "=" * 80)
print("5. OUTLIER DECISION")
print("=" * 80)

print(
    """
Outlier policy for this project:

1. IQR outliers are DETECTED, not automatically deleted.
2. Negative days_to_target values are retained.
3. High turnaround-time values are retained when they are
   operationally plausible.
4. Impossible domain values are treated as data-quality issues.
5. Genuine severe delays are valuable because the prediction
   system is specifically designed to identify delay risk.
"""
)


# ============================================================
# 6. REPORT
# ============================================================

report_lines = [
    "# Round 1 — Outlier Analysis Report",
    "",
    "## Approach",
    "",
    "IQR-based statistical outlier detection was applied to numerical variables.",
    "Additional domain-specific validation rules were applied to identify impossible or logically inconsistent values.",
    "",
    "## Outlier Policy",
    "",
    "- Statistical outliers were detected but not automatically removed.",
    "- Negative `days_to_target` values were retained because they indicate that an application's target date has been crossed.",
    "- Extreme turnaround times were retained when operationally plausible.",
    "- Domain-invalid values are treated as data-quality issues.",
    "- Severe delays are important to the project's prediction objective and should not be removed merely because they are statistically extreme.",
    "",
    "## Generated Files",
    "",
    f"- `{OUTLIER_REPORT}`",
    f"- `{OUTPUT_DIR / 'round1_domain_outlier_checks.csv'}`",
    f"- `{OUTPUT_DIR / 'round1_tat_extreme_analysis.csv'}`",
    f"- `{OUTPUT_DIR / 'round1_top_20_tat_extremes.csv'}`"
]

OUTLIER_SUMMARY.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("OUTLIER ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Outlier report : {OUTLIER_REPORT}"
)

print(
    f"Summary report : {OUTLIER_SUMMARY}"
)