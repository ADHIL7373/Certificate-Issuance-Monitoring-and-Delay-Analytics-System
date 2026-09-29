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

DESCRIPTIVE_OUTPUT = OUTPUT_DIR / "round1_descriptive_statistics.csv"
TARGET_OUTPUT = OUTPUT_DIR / "round1_target_analysis.csv"

REPORT = REPORT_DIR / "ROUND1_DESCRIPTIVE_ANALYSIS.md"


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("ROUND 1 — DESCRIPTIVE STATISTICS & TARGET ANALYSIS")
print("=" * 80)

df = pd.read_csv(
    CLEAN_DATA,
    low_memory=False
)

print(f"\nRows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 1. NUMERIC DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("1. NUMERIC DESCRIPTIVE STATISTICS")
print("=" * 80)

numeric_columns = [
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
    "actual_total_tat_days",
    "actual_remaining_tat_days",
    "delay_overrun_days"
]

descriptive_rows = []

for col in numeric_columns:

    if col not in df.columns:
        continue

    series = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    descriptive_rows.append({
        "feature": col,
        "count": series.count(),
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "median": series.median(),
        "q3": series.quantile(0.75),
        "p95": series.quantile(0.95),
        "p99": series.quantile(0.99),
        "max": series.max()
    })

    print(
        f"{col:40s} "
        f"mean={series.mean():8.2f} "
        f"median={series.median():8.2f} "
        f"p95={series.quantile(0.95):8.2f} "
        f"max={series.max():8.2f}"
    )


descriptive_df = pd.DataFrame(descriptive_rows)

descriptive_df.to_csv(
    DESCRIPTIVE_OUTPUT,
    index=False
)


# ============================================================
# 2. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("2. TARGET DISTRIBUTION")
print("=" * 80)

target_counts = (
    df["missed_target"]
    .value_counts()
    .sort_index()
)

target_percentages = (
    df["missed_target"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

target_rows = []

for target_value in [0, 1]:

    count = int(
        target_counts.get(target_value, 0)
    )

    percentage = float(
        target_percentages.get(target_value, 0)
    )

    label = (
        "On Time"
        if target_value == 0
        else "Missed Target"
    )

    target_rows.append({
        "missed_target": target_value,
        "label": label,
        "count": count,
        "percentage": percentage
    })

    print(
        f"{label:20s}: "
        f"{count:6,d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 3. TAT BY TARGET CLASS
# ============================================================

print("\n" + "=" * 80)
print("3. TAT BY TARGET CLASS")
print("=" * 80)

tat_by_target = (
    df.groupby("missed_target")[
        [
            "actual_total_tat_days",
            "actual_remaining_tat_days",
            "delay_overrun_days"
        ]
    ]
    .agg([
        "count",
        "mean",
        "median",
        "min",
        "max"
    ])
)

print(tat_by_target.to_string())

tat_by_target.to_csv(
    OUTPUT_DIR / "round1_tat_by_target.csv"
)


# ============================================================
# 4. DELAY RATE BY CERTIFICATE TYPE
# ============================================================

print("\n" + "=" * 80)
print("4. DELAY RATE BY CERTIFICATE TYPE")
print("=" * 80)

certificate_analysis = (
    df.groupby("certificate_type")
    .agg(
        applications=("application_id", "count"),
        delayed=("missed_target", "sum"),
        delay_rate=("missed_target", "mean"),
        median_total_tat=("actual_total_tat_days", "median"),
        mean_total_tat=("actual_total_tat_days", "mean")
    )
    .reset_index()
)

certificate_analysis["delay_rate_pct"] = (
    certificate_analysis["delay_rate"] * 100
)

certificate_analysis = certificate_analysis.sort_values(
    "delay_rate_pct",
    ascending=False
)

print(
    certificate_analysis.to_string(index=False)
)

certificate_analysis.to_csv(
    OUTPUT_DIR / "round1_delay_by_certificate_type.csv",
    index=False
)


# ============================================================
# 5. DELAY RATE BY CURRENT STAGE
# ============================================================

print("\n" + "=" * 80)
print("5. DELAY RATE BY CURRENT STAGE")
print("=" * 80)

stage_analysis = (
    df.groupby("current_stage")
    .agg(
        applications=("application_id", "count"),
        delayed=("missed_target", "sum"),
        delay_rate=("missed_target", "mean"),
        median_stage_age=("stage_age_days", "median"),
        median_queue=("queue_length", "median"),
        median_workload=("department_workload_index", "median"),
        median_staff_availability=("staff_availability_pct", "median")
    )
    .reset_index()
)

stage_analysis["delay_rate_pct"] = (
    stage_analysis["delay_rate"] * 100
)

stage_analysis = stage_analysis.sort_values(
    "delay_rate_pct",
    ascending=False
)

print(
    stage_analysis.to_string(index=False)
)

stage_analysis.to_csv(
    OUTPUT_DIR / "round1_delay_by_current_stage.csv",
    index=False
)


# ============================================================
# 6. DELAY RATE BY SUBMISSION CHANNEL
# ============================================================

print("\n" + "=" * 80)
print("6. DELAY RATE BY SUBMISSION CHANNEL")
print("=" * 80)

channel_analysis = (
    df.groupby("submission_channel")
    .agg(
        applications=("application_id", "count"),
        delayed=("missed_target", "sum"),
        delay_rate=("missed_target", "mean"),
        median_total_tat=("actual_total_tat_days", "median")
    )
    .reset_index()
)

channel_analysis["delay_rate_pct"] = (
    channel_analysis["delay_rate"] * 100
)

print(
    channel_analysis.to_string(index=False)
)

channel_analysis.to_csv(
    OUTPUT_DIR / "round1_delay_by_submission_channel.csv",
    index=False
)


# ============================================================
# 7. DELAY RATE BY PEAK PERIOD
# ============================================================

print("\n" + "=" * 80)
print("7. DELAY RATE BY PEAK PERIOD")
print("=" * 80)

peak_analysis = (
    df.groupby("peak_period_flag")
    .agg(
        applications=("application_id", "count"),
        delayed=("missed_target", "sum"),
        delay_rate=("missed_target", "mean"),
        median_total_tat=("actual_total_tat_days", "median"),
        median_queue=("queue_length", "median")
    )
    .reset_index()
)

peak_analysis["delay_rate_pct"] = (
    peak_analysis["delay_rate"] * 100
)

print(
    peak_analysis.to_string(index=False)
)

peak_analysis.to_csv(
    OUTPUT_DIR / "round1_delay_by_peak_period.csv",
    index=False
)


# ============================================================
# 8. DELAY RATE BY DISTRICT
# ============================================================

print("\n" + "=" * 80)
print("8. DELAY RATE BY DISTRICT")
print("=" * 80)

district_analysis = (
    df.groupby("district")
    .agg(
        applications=("application_id", "count"),
        delayed=("missed_target", "sum"),
        delay_rate=("missed_target", "mean"),
        median_total_tat=("actual_total_tat_days", "median")
    )
    .reset_index()
)

district_analysis["delay_rate_pct"] = (
    district_analysis["delay_rate"] * 100
)

district_analysis = district_analysis.sort_values(
    "delay_rate_pct",
    ascending=False
)

print(
    district_analysis.head(15).to_string(index=False)
)

district_analysis.to_csv(
    OUTPUT_DIR / "round1_delay_by_district.csv",
    index=False
)


# ============================================================
# 9. TARGET-CLASS FEATURE COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("9. FEATURE COMPARISON: ON-TIME VS DELAYED")
print("=" * 80)

comparison_features = [
    "application_age_days",
    "stage_age_days",
    "days_to_target",
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

comparison_rows = []

for feature in comparison_features:

    on_time = df.loc[
        df["missed_target"] == 0,
        feature
    ]

    delayed = df.loc[
        df["missed_target"] == 1,
        feature
    ]

    comparison_rows.append({
        "feature": feature,

        "on_time_mean": on_time.mean(),
        "on_time_median": on_time.median(),

        "delayed_mean": delayed.mean(),
        "delayed_median": delayed.median(),

        "mean_difference": (
            delayed.mean() - on_time.mean()
        ),

        "median_difference": (
            delayed.median() - on_time.median()
        )
    })


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    OUTPUT_DIR / "round1_feature_target_comparison.csv",
    index=False
)

print(
    comparison_df.to_string(index=False)
)


# ============================================================
# 10. REPORT
# ============================================================

report_lines = [
    "# Round 1 — Descriptive Statistics & Target Analysis",
    "",
    "## Dataset",
    "",
    f"- Applications: {len(df):,}",
    "",
    "## Target",
    "",
    f"- On-time applications: {target_rows[0]['count']:,} ({target_rows[0]['percentage']:.2f}%)",
    f"- Missed-target applications: {target_rows[1]['count']:,} ({target_rows[1]['percentage']:.2f}%)",
    "",
    "## Analysis Performed",
    "",
    "- Numerical descriptive statistics",
    "- Target distribution",
    "- TAT comparison by target class",
    "- Delay rate by certificate type",
    "- Delay rate by current stage",
    "- Delay rate by submission channel",
    "- Delay rate by peak period",
    "- Delay rate by district",
    "- Feature comparison between on-time and delayed applications",
    "",
    "## Important Modeling Note",
    "",
    "Actual turnaround and outcome columns are used here for descriptive analysis only.",
    "They must not be used as predictive inputs in the final model.",
    "",
    "## Generated Outputs",
    "",
    f"- `{DESCRIPTIVE_OUTPUT}`",
    f"- `{TARGET_OUTPUT}`",
    "- `round1_tat_by_target.csv`",
    "- `round1_delay_by_certificate_type.csv`",
    "- `round1_delay_by_current_stage.csv`",
    "- `round1_delay_by_submission_channel.csv`",
    "- `round1_delay_by_peak_period.csv`",
    "- `round1_delay_by_district.csv`",
    "- `round1_feature_target_comparison.csv`"
]

REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("DESCRIPTIVE ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Descriptive statistics : {DESCRIPTIVE_OUTPUT}"
)

print(
    f"Target analysis        : {TARGET_OUTPUT}"
)