from pathlib import Path
import pandas as pd
import numpy as np

from scipy.stats import pointbiserialr, mannwhitneyu, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

CLEAN_DATA = OUTPUT_DIR / "round1_cleaned_applications.csv"

OUTPUT_FILE = OUTPUT_DIR / "round1_feature_relevance.csv"
CANDIDATE_FILE = OUTPUT_DIR / "round1_feature_candidates.csv"

REPORT_FILE = REPORT_DIR / "ROUND1_FEATURE_RELEVANCE_REPORT.md"


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("ROUND 1 — FEATURE RELEVANCE ANALYSIS")
print("=" * 80)

df = pd.read_csv(
    CLEAN_DATA,
    low_memory=False
)

print(f"\nRows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# TARGET
# ============================================================

y = df["missed_target"].astype(int)


# ============================================================
# CANDIDATE FEATURES
# ============================================================

numeric_features = [
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
    "data_quality_flag"
]

categorical_features = [
    "service_code",
    "certificate_type",
    "district",
    "submission_channel",
    "submission_day_of_week",
    "current_stage",
    "target_tat_basis"
]


# ============================================================
# LEAKAGE / IDENTIFIER EXCLUSIONS
# ============================================================

excluded_columns = {
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
    "taluk"
}


# ============================================================
# 1. NUMERIC FEATURE RELEVANCE
# ============================================================

print("\n" + "=" * 80)
print("1. NUMERIC FEATURE RELEVANCE")
print("=" * 80)

numeric_results = []

for feature in numeric_features:

    if feature in excluded_columns:
        continue

    if feature not in df.columns:
        continue

    x = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    valid = x.notna() & y.notna()

    x_valid = x.loc[valid]
    y_valid = y.loc[valid]

    if x_valid.nunique() <= 1:
        continue

    # Point-biserial correlation
    corr, corr_p = pointbiserialr(
        y_valid,
        x_valid
    )

    # Mann-Whitney U
    group_0 = x_valid[y_valid == 0]
    group_1 = x_valid[y_valid == 1]

    try:
        u_stat, mw_p = mannwhitneyu(
            group_0,
            group_1,
            alternative="two-sided"
        )
    except Exception:
        u_stat = np.nan
        mw_p = np.nan

    median_0 = group_0.median()
    median_1 = group_1.median()

    mean_0 = group_0.mean()
    mean_1 = group_1.mean()

    median_difference = median_1 - median_0
    mean_difference = mean_1 - mean_0

    numeric_results.append({
        "feature": feature,
        "feature_type": "numeric",
        "correlation": corr,
        "correlation_p_value": corr_p,
        "mann_whitney_p_value": mw_p,
        "on_time_median": median_0,
        "delayed_median": median_1,
        "median_difference": median_difference,
        "on_time_mean": mean_0,
        "delayed_mean": mean_1,
        "mean_difference": mean_difference
    })

    print(
        f"{feature:40s} "
        f"corr={corr:8.4f} "
        f"p={corr_p:.3e}"
    )


# ============================================================
# 2. CATEGORICAL FEATURE RELEVANCE
# ============================================================

print("\n" + "=" * 80)
print("2. CATEGORICAL FEATURE RELEVANCE")
print("=" * 80)

categorical_results = []

for feature in categorical_features:

    if feature in excluded_columns:
        continue

    if feature not in df.columns:
        continue

    contingency = pd.crosstab(
        df[feature],
        y
    )

    if contingency.shape[0] < 2:
        continue

    chi2, p_value, dof, expected = chi2_contingency(
        contingency
    )

    n = contingency.to_numpy().sum()

    min_dimension = min(
        contingency.shape[0] - 1,
        contingency.shape[1] - 1
    )

    if min_dimension > 0:
        cramers_v = np.sqrt(
            (chi2 / n) / min_dimension
        )
    else:
        cramers_v = np.nan

    categorical_results.append({
        "feature": feature,
        "feature_type": "categorical",
        "chi_square": chi2,
        "chi_square_p_value": p_value,
        "cramers_v": cramers_v,
        "category_count": df[feature].nunique()
    })

    print(
        f"{feature:40s} "
        f"Cramer's V={cramers_v:8.4f} "
        f"p={p_value:.3e}"
    )


# ============================================================
# 3. COMBINE
# ============================================================

results_df = pd.concat(
    [
        pd.DataFrame(numeric_results),
        pd.DataFrame(categorical_results)
    ],
    ignore_index=True
)


# ============================================================
# 4. MUTUAL INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("3. MUTUAL INFORMATION")
print("=" * 80)

mi_features = []

# Numeric features
for feature in numeric_features:

    if feature in excluded_columns:
        continue

    if feature not in df.columns:
        continue

    x = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    x = x.fillna(x.median())

    try:
        mi = mutual_info_classif(
            x.to_numpy().reshape(-1, 1),
            y.to_numpy(),
            random_state=42
        )[0]
    except Exception:
        mi = np.nan

    mi_features.append({
        "feature": feature,
        "mutual_information": mi
    })


# Categorical features
for feature in categorical_features:

    if feature in excluded_columns:
        continue

    if feature not in df.columns:
        continue

    encoded = LabelEncoder().fit_transform(
        df[feature].astype(str)
    )

    try:
        mi = mutual_info_classif(
            encoded.reshape(-1, 1),
            y.to_numpy(),
            discrete_features=True,
            random_state=42
        )[0]
    except Exception:
        mi = np.nan

    mi_features.append({
        "feature": feature,
        "mutual_information": mi
    })


mi_df = pd.DataFrame(mi_features)

results_df = results_df.merge(
    mi_df,
    on="feature",
    how="left"
)


# ============================================================
# 5. RELEVANCE DECISION
# ============================================================

print("\n" + "=" * 80)
print("4. FEATURE DECISION")
print("=" * 80)


def decide_numeric(row):

    corr = abs(row.get("correlation", np.nan))
    p = row.get("correlation_p_value", np.nan)
    mi = row.get("mutual_information", np.nan)

    if (
        (pd.notna(p) and p < 0.001 and corr >= 0.10)
        or
        (pd.notna(mi) and mi >= 0.02)
    ):
        return "KEEP"

    if (
        (pd.notna(p) and p < 0.05)
        or
        (pd.notna(mi) and mi >= 0.005)
    ):
        return "REVIEW"

    return "REVIEW"


def decide_categorical(row):

    v = row.get("cramers_v", np.nan)
    p = row.get("chi_square_p_value", np.nan)
    mi = row.get("mutual_information", np.nan)

    if (
        (pd.notna(p) and p < 0.001 and v >= 0.05)
        or
        (pd.notna(mi) and mi >= 0.02)
    ):
        return "KEEP"

    if (
        (pd.notna(p) and p < 0.05)
        or
        (pd.notna(mi) and mi >= 0.005)
    ):
        return "REVIEW"

    return "REVIEW"


decisions = []

for _, row in results_df.iterrows():

    if row["feature_type"] == "numeric":
        decision = decide_numeric(row)
    else:
        decision = decide_categorical(row)

    decisions.append(decision)


results_df["decision"] = decisions


# ============================================================
# 6. MODEL AVAILABILITY / DOMAIN NOTES
# ============================================================

domain_notes = {

    "submission_month":
        "Potential temporal/seasonal feature.",

    "peak_period_flag":
        "Operational load-period indicator.",

    "application_age_days":
        "Available at prediction time; represents current age.",

    "stage_age_days":
        "Available at prediction time; represents time in current stage.",

    "days_to_target":
        "Highly relevant time-pressure feature.",

    "target_tat_days":
        "Service-specific target available at prediction time.",

    "required_document_count":
        "Document burden available at prediction time.",

    "submitted_document_count":
        "Document progress available at prediction time.",

    "document_completeness_pct":
        "Document completeness available at prediction time.",

    "missing_document_count":
        "Missing-document burden available at prediction time.",

    "verification_complexity":
        "Operational verification complexity.",

    "resubmission_count":
        "Prior correction/resubmission burden.",

    "queue_length":
        "Current operational queue pressure.",

    "pending_cases_department":
        "Department-level workload indicator.",

    "department_workload_index":
        "Department workload pressure.",

    "staff_availability_pct":
        "Available staffing level.",

    "recent_backlog_growth_pct":
        "Recent backlog trend.",

    "historical_stage_median_days":
        "Historical stage-level processing time.",

    "historical_service_median_days":
        "Historical service-level processing time.",

    "data_quality_flag":
        "Data-quality indicator; use cautiously and validate against leakage.",

    "service_code":
        "Service identifier; potentially redundant with certificate type.",

    "certificate_type":
        "Service/category information.",

    "district":
        "Geographic operational context.",

    "submission_channel":
        "Submission route.",

    "submission_day_of_week":
        "Potential weekly operational pattern.",

    "current_stage":
        "Current workflow position.",

    "target_tat_basis":
        "Indicates basis of target SLA."
}


results_df["domain_note"] = (
    results_df["feature"]
    .map(domain_notes)
    .fillna("")
)


# ============================================================
# 7. SORT
# ============================================================

results_df["abs_correlation"] = (
    results_df["correlation"].abs()
)

results_df["abs_correlation"] = (
    results_df["abs_correlation"]
    .fillna(0)
)

results_df["cramers_v"] = (
    results_df["cramers_v"]
    .fillna(0)
)

results_df["mutual_information"] = (
    results_df["mutual_information"]
    .fillna(0)
)

results_df["relevance_score"] = (
    results_df[
        [
            "abs_correlation",
            "cramers_v",
            "mutual_information"
        ]
    ]
    .max(axis=1)
)

results_df = results_df.sort_values(
    "relevance_score",
    ascending=False
)


# ============================================================
# 8. SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


candidate_df = results_df[
    results_df["decision"].isin(
        ["KEEP", "REVIEW"]
    )
].copy()

candidate_df.to_csv(
    CANDIDATE_FILE,
    index=False
)


# ============================================================
# 9. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FEATURE RELEVANCE SUMMARY")
print("=" * 80)

print(
    results_df[
        [
            "feature",
            "feature_type",
            "correlation",
            "cramers_v",
            "mutual_information",
            "decision"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. REPORT
# ============================================================

report_lines = [
    "# Round 1 — Feature Relevance Report",
    "",
    "## Objective",
    "",
    "Assess the relationship between candidate input features and the `missed_target` classification target before model training.",
    "",
    "## Methods",
    "",
    "- Point-biserial correlation for numerical features.",
    "- Mann-Whitney U comparison for numerical features.",
    "- Chi-square test for categorical features.",
    "- Cramér's V for categorical association strength.",
    "- Mutual information as supplementary nonlinear relevance evidence.",
    "",
    "## Leakage Policy",
    "",
    "Actual processing durations, actual total TAT, actual remaining TAT, delay overrun, actual bottleneck stage, application ID and target itself are excluded from predictive inputs.",
    "",
    "## Important Note",
    "",
    "Statistical association does not establish causation.",
    "Final model feature selection will also consider prediction-time availability, domain meaning, redundancy and leakage risk.",
    "",
    "## Outputs",
    "",
    f"- `{OUTPUT_FILE}`",
    f"- `{CANDIDATE_FILE}`"
]

REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("FEATURE RELEVANCE ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Full relevance report : {OUTPUT_FILE}"
)

print(
    f"Candidate features    : {CANDIDATE_FILE}"
)