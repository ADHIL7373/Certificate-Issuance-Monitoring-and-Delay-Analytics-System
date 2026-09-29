from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Certificate Delay Intelligence",
    page_icon="📋",
    layout="wide"
)


# ==========================================================
# PATHS
# ==========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_DIR = PROJECT_DIR / "outputs" / "round2"
DATA_DIR = PROJECT_DIR.parent / "dataset"

DELAY_MODEL_FILE = (
    MODEL_DIR / "random_forest_delay_model.joblib"
)

TAT_MODEL_FILE = (
    MODEL_DIR / "random_forest_remaining_tat_model.joblib"
)

RAW_DATA_FILE = (
    DATA_DIR / "certificate_applications_raw_50000.csv"
)


# ==========================================================
# HEADER
# ==========================================================

st.title("Certificate Issuance Monitoring and Delay Analytics")

st.write(
    "Predict certificate processing delay risk and "
    "estimate remaining turnaround time."
)

st.divider()


# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    delay_model = joblib.load(
        DELAY_MODEL_FILE
    )

    tat_model = joblib.load(
        TAT_MODEL_FILE
    )

    return delay_model, tat_model


@st.cache_data
def load_reference_data():

    df = pd.read_csv(
        RAW_DATA_FILE,
        low_memory=False
    )

    return df


try:

    delay_model, tat_model = load_models()
    reference_df = load_reference_data()

except Exception as e:

    st.error("Unable to load models or reference data.")
    st.code(str(e))
    st.stop()


# ==========================================================
# MODEL STATUS
# ==========================================================

st.success("Machine Learning Models Loaded Successfully")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Delay Risk Model",
        "Random Forest"
    )

with col2:

    st.metric(
        "TAT Prediction Model",
        "Random Forest"
    )


st.divider()


# ==========================================================
# INPUT SECTION
# ==========================================================

st.header("New Application Prediction")

st.write(
    "Enter the current application details to "
    "predict delay risk and remaining processing time."
)


# ==========================================================
# OPTIONS FROM DATASET
# ==========================================================

certificate_options = sorted(
    reference_df["certificate_type"]
    .dropna()
    .astype(str)
    .unique()
)

service_options = sorted(
    reference_df["service_code"]
    .dropna()
    .astype(str)
    .unique()
)

district_options = sorted(
    reference_df["district"]
    .dropna()
    .astype(str)
    .unique()
)

taluk_options = sorted(
    reference_df["taluk"]
    .dropna()
    .astype(str)
    .unique()
)

channel_options = sorted(
    reference_df["submission_channel"]
    .dropna()
    .astype(str)
    .unique()
)

day_options = sorted(
    reference_df["submission_day_of_week"]
    .dropna()
    .astype(str)
    .unique()
)

stage_options = sorted(
    reference_df["current_stage"]
    .dropna()
    .astype(str)
    .unique()
)

basis_options = sorted(
    reference_df["target_tat_basis"]
    .dropna()
    .astype(str)
    .unique()
)


# ==========================================================
# BASIC INFORMATION
# ==========================================================

st.subheader("Application Information")

col1, col2, col3 = st.columns(3)

with col1:

    service_code = st.selectbox(
        "Service Code",
        service_options
    )

with col2:

    certificate_type = st.selectbox(
        "Certificate Type",
        certificate_options
    )

with col3:

    district = st.selectbox(
        "District",
        district_options
    )


col1, col2, col3 = st.columns(3)

with col1:

    taluk = st.selectbox(
        "Taluk",
        taluk_options
    )

with col2:

    submission_channel = st.selectbox(
        "Submission Channel",
        channel_options
    )

with col3:

    submission_day_of_week = st.selectbox(
        "Submission Day",
        day_options
    )


# ==========================================================
# PROCESSING INFORMATION
# ==========================================================

st.subheader("Processing Information")

col1, col2, col3 = st.columns(3)

with col1:

    current_stage = st.selectbox(
        "Current Stage",
        stage_options
    )

with col2:

    submission_month = st.number_input(
        "Submission Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1
    )

with col3:

    peak_period_flag = st.selectbox(
        "Peak Period",
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )


col1, col2, col3 = st.columns(3)

with col1:

    application_age_days = st.number_input(
        "Application Age (days)",
        min_value=0.0,
        max_value=100.0,
        value=3.0,
        step=0.1
    )

with col2:

    stage_age_days = st.number_input(
        "Current Stage Age (days)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1
    )

with col3:

    days_to_target = st.number_input(
        "Days to Target",
        min_value=-100.0,
        max_value=100.0,
        value=7.0,
        step=0.1
    )


# ==========================================================
# DOCUMENT INFORMATION
# ==========================================================

st.subheader("Document & Verification Information")

col1, col2, col3 = st.columns(3)

with col1:

    target_tat_days = st.number_input(
        "Target TAT (days)",
        min_value=1.0,
        max_value=100.0,
        value=15.0,
        step=1.0
    )

with col2:

    required_document_count = st.number_input(
        "Required Documents",
        min_value=0,
        max_value=50,
        value=6,
        step=1
    )

with col3:

    submitted_document_count = st.number_input(
        "Submitted Documents",
        min_value=0,
        max_value=50,
        value=6,
        step=1
    )


col1, col2, col3 = st.columns(3)

with col1:

    document_completeness_pct = st.number_input(
        "Document Completeness (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0
    )

with col2:

    missing_document_count = st.number_input(
        "Missing Documents",
        min_value=0,
        max_value=50,
        value=0,
        step=1
    )

with col3:

    verification_complexity = st.number_input(
        "Verification Complexity",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1
    )


col1, col2 = st.columns(2)

with col1:

    resubmission_count = st.number_input(
        "Resubmission Count",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

with col2:

    target_tat_basis = st.selectbox(
        "Target TAT Basis",
        basis_options
    )


# ==========================================================
# OPERATIONAL INFORMATION
# ==========================================================

st.subheader("Department Operational Information")

col1, col2, col3 = st.columns(3)

with col1:

    queue_length = st.number_input(
        "Queue Length",
        min_value=0.0,
        max_value=500.0,
        value=40.0,
        step=1.0
    )

with col2:

    pending_cases_department = st.number_input(
        "Pending Department Cases",
        min_value=0.0,
        max_value=500.0,
        value=60.0,
        step=1.0
    )

with col3:

    department_workload_index = st.number_input(
        "Department Workload Index",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=1.0
    )


col1, col2, col3 = st.columns(3)

with col1:

    staff_availability_pct = st.number_input(
        "Staff Availability (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

with col2:

    recent_backlog_growth_pct = st.number_input(
        "Recent Backlog Growth (%)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1
    )

with col3:

    historical_stage_median_days = st.number_input(
        "Historical Stage Median (days)",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.1
    )


historical_service_median_days = st.number_input(
    "Historical Service Median (days)",
    min_value=0.0,
    max_value=100.0,
    value=9.0,
    step=0.1
)


# ==========================================================
# PREDICTION
# ==========================================================

st.divider()

predict_button = st.button(
    "Predict Application Risk",
    type="primary",
    use_container_width=True
)


if predict_button:

    # ------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # ------------------------------------------------------

    input_data = pd.DataFrame({

        "service_code": [service_code],

        "certificate_type": [certificate_type],

        "district": [district],

        "taluk": [taluk],

        "submission_channel": [submission_channel],

        "submission_month": [submission_month],

        "submission_day_of_week": [
            submission_day_of_week
        ],

        "peak_period_flag": [
            peak_period_flag
        ],

        "current_stage": [
            current_stage
        ],

        "application_age_days": [
            application_age_days
        ],

        "stage_age_days": [
            stage_age_days
        ],

        "days_to_target": [
            days_to_target
        ],

        "target_tat_days": [
            target_tat_days
        ],

        "required_document_count": [
            required_document_count
        ],

        "submitted_document_count": [
            submitted_document_count
        ],

        "document_completeness_pct": [
            document_completeness_pct
        ],

        "missing_document_count": [
            missing_document_count
        ],

        "verification_complexity": [
            verification_complexity
        ],

        "resubmission_count": [
            resubmission_count
        ],

        "queue_length": [
            queue_length
        ],

        "pending_cases_department": [
            pending_cases_department
        ],

        "department_workload_index": [
            department_workload_index
        ],

        "staff_availability_pct": [
            staff_availability_pct
        ],

        "recent_backlog_growth_pct": [
            recent_backlog_growth_pct
        ],

        "historical_stage_median_days": [
            historical_stage_median_days
        ],

        "historical_service_median_days": [
            historical_service_median_days
        ],

        "target_tat_basis": [
            target_tat_basis
        ]
    })


    # ------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------

    delay_probability = float(
        delay_model.predict_proba(
            input_data
        )[0][1]
    )

    predicted_remaining_tat = float(
        tat_model.predict(
            input_data
        )[0]
    )

    predicted_remaining_tat = max(
        predicted_remaining_tat,
        0
    )

    predicted_total_tat = (
        application_age_days
        + predicted_remaining_tat
    )


    # ------------------------------------------------------
    # RISK LEVEL
    # ------------------------------------------------------

    if delay_probability >= 0.70:

        risk = "HIGH"

    elif delay_probability >= 0.40:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    # ------------------------------------------------------
    # RESULT
    # ------------------------------------------------------

    st.divider()

    st.header("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Delay Probability",
            f"{delay_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Level",
            risk
        )

    with col3:

        st.metric(
            "Remaining TAT",
            f"{predicted_remaining_tat:.2f} days"
        )


    st.metric(
        "Predicted Total TAT",
        f"{predicted_total_tat:.2f} days"
    )


    # ------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------

    if risk == "HIGH":

        st.error(
            "HIGH RISK: This application has a high "
            "predicted probability of missing its target."
        )

    elif risk == "MEDIUM":

        st.warning(
            "MEDIUM RISK: This application requires "
            "additional monitoring."
        )

    else:

        st.success(
            "LOW RISK: This application has a lower "
            "predicted probability of missing its target."
        )


    # ------------------------------------------------------
    # IMPORTANT INPUT FACTORS
    # ------------------------------------------------------

    st.subheader(
        "Important Operational Factors"
    )

    factors = pd.DataFrame({

        "Factor": [
            "Department Workload",
            "Queue Length",
            "Application Age",
            "Document Completeness",
            "Missing Documents",
            "Stage Age",
            "Staff Availability",
            "Resubmissions"
        ],

        "Current Value": [
            f"{department_workload_index:.1f}",
            f"{queue_length:.1f}",
            f"{application_age_days:.2f} days",
            f"{document_completeness_pct:.1f}%",
            f"{missing_document_count}",
            f"{stage_age_days:.2f} days",
            f"{staff_availability_pct:.1f}%",
            f"{resubmission_count}"
        ]
    })

    st.dataframe(
        factors,
        use_container_width=True,
        hide_index=True
    )


    # ==========================================================
# ROUND 3 - ANALYTICS DASHBOARD
# ==========================================================

st.divider()

st.header("Application Monitoring Dashboard")

FINAL_FILE = (
    PROJECT_DIR
    / "outputs"
    / "round2"
    / "FINAL_JURY_APPLICATION_RANKING.csv"
)

if FINAL_FILE.exists():

    dashboard_df = pd.read_csv(FINAL_FILE)

    # ------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------

    total_apps = len(dashboard_df)

    high_risk = (
        dashboard_df["risk_level"]
        .eq("HIGH")
        .sum()
    )

    medium_risk = (
        dashboard_df["risk_level"]
        .eq("MEDIUM")
        .sum()
    )

    low_risk = (
        dashboard_df["risk_level"]
        .eq("LOW")
        .sum()
    )

    avg_remaining_tat = (
        dashboard_df["predicted_remaining_tat_days"]
        .mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Applications",
            f"{total_apps:,}"
        )

    with col2:
        st.metric(
            "High Risk",
            f"{high_risk:,}"
        )

    with col3:
        st.metric(
            "Medium Risk",
            f"{medium_risk:,}"
        )

    with col4:
        st.metric(
            "Avg Remaining TAT",
            f"{avg_remaining_tat:.2f} days"
        )

    # ------------------------------------------------------
    # RISK DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("Risk Distribution")

    risk_chart = (
        dashboard_df["risk_level"]
        .value_counts()
        .reindex(
            ["LOW", "MEDIUM", "HIGH"],
            fill_value=0
        )
    )

    st.bar_chart(risk_chart)

    # ------------------------------------------------------
    # CURRENT STAGE DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("Applications by Current Stage")

    stage_chart = (
        dashboard_df["current_stage"]
        .value_counts()
    )

    st.bar_chart(stage_chart)

    # ------------------------------------------------------
    # CERTIFICATE TYPE
    # ------------------------------------------------------

    st.subheader("Applications by Certificate Type")

    certificate_chart = (
        dashboard_df["certificate_type"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(certificate_chart)

    # ------------------------------------------------------
    # HIGH RISK APPLICATIONS
    # ------------------------------------------------------

    st.subheader("Top High-Risk Applications")

    high_risk_table = (
        dashboard_df[
            dashboard_df["risk_level"] == "HIGH"
        ]
        .sort_values(
            "delay_probability",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        high_risk_table[
            [
                "priority_rank",
                "application_id",
                "certificate_type",
                "district",
                "current_stage",
                "delay_probability",
                "risk_level",
                "predicted_remaining_tat_days",
                "predicted_total_tat_days"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------
    # DELAY PROBABILITY DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("Delay Probability")

    probability_chart = (
        dashboard_df[
            "delay_probability"
        ]
        .round(1)
        .value_counts()
        .sort_index()
    )

    st.line_chart(
        probability_chart
    )

    # ------------------------------------------------------
    # TAT ANALYSIS
    # ------------------------------------------------------

    st.subheader("Predicted Remaining TAT")

    tat_chart = (
        dashboard_df[
            "predicted_remaining_tat_days"
        ]
        .round(0)
        .value_counts()
        .sort_index()
        .head(20)
    )

    st.bar_chart(
        tat_chart
    )

else:

    st.warning(
        "Final jury prediction file not found."
    )

# ==========================================================
# MODEL FEATURE IMPORTANCE
# ==========================================================

st.divider()

st.header("Major Delay Factors")

try:

    rf_model = delay_model.named_steps["model"]
    preprocessor = delay_model.named_steps["preprocessor"]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.caption(
        "These are the features the Random Forest relied on most "
        "for delay-risk prediction. Feature importance indicates "
        "model reliance, not causation."
    )

except Exception as e:

    st.warning(
        f"Feature importance could not be displayed: {e}"
    )