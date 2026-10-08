"""Customer Churn Prediction view: Individual customer assessment and explainability.

Engineered with a warm light enterprise visual system and non-jargon factor attributions.
"""

import streamlit as st

from dashboard.components.cards import (
    render_driver_card,
    render_metric_card,
    render_risk_badge,
)
from dashboard.components.tokens import (
    COLOR_BORDER,
    COLOR_PRIMARY_SURFACE,
    COLOR_PRIMARY_TEXT,
    COLOR_SECONDARY_TEXT,
    FONT_FAMILY,
)
from src.telco_churn.inference.predictor import (
    CustomerPredictionRequest,
    get_predictor,
)


def render_customer_prediction() -> None:
    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 1.75rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 4px; font-family: {FONT_FAMILY};">
                Customer Churn Prediction
            </h1>
            <p style="color: {COLOR_SECONDARY_TEXT}; font-size: 0.95rem; margin-bottom: 14px; font-family: {FONT_FAMILY};">
                Estimate churn risk for an individual customer and understand which factors influence the prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compact 3-Step Workflow Banner
    st.markdown(
        f"""
        <div style="
            background: {COLOR_PRIMARY_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.82rem;
            color: {COLOR_SECONDARY_TEXT};
            font-family: {FONT_FAMILY};
        ">
            <div><span style="font-weight: 700; color: {COLOR_PRIMARY_TEXT};">1.</span> Enter customer details</div>
            <div style="color: {COLOR_BORDER};">→</div>
            <div><span style="font-weight: 700; color: {COLOR_PRIMARY_TEXT};">2.</span> Run prediction</div>
            <div style="color: {COLOR_BORDER};">→</div>
            <div><span style="font-weight: 700; color: {COLOR_PRIMARY_TEXT};">3.</span> Review risk and contributing factors</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        predictor = get_predictor()
    except Exception:
        st.error(
            "The prediction service is temporarily unavailable. Please verify that the model artifact is loaded."
        )
        return

    col_input, col_result = st.columns([1.1, 0.9])

    with col_input:
        with st.form("customer_prediction_form"):
            st.markdown(
                f"<div style='font-size: 0.96rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 8px;'>Customer Profile</div>",
                unsafe_allow_html=True,
            )
            customer_id = st.text_input(
                "Customer ID",
                value="CUST-7590-VH",
                help="Account or customer reference identifier.",
            )

            p1, p2 = st.columns(2)
            with p1:
                gender = st.selectbox("Gender", options=["Female", "Male"])
                senior = st.selectbox(
                    "Senior Citizen",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                )
            with p2:
                partner = st.selectbox("Partner", options=["No", "Yes"])
                dependents = st.selectbox("Dependents", options=["No", "Yes"])

            st.markdown(
                f"<div style='font-size: 0.96rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin: 16px 0 8px 0;'>Account & Billing</div>",
                unsafe_allow_html=True,
            )
            a1, a2 = st.columns(2)
            with a1:
                tenure = st.slider(
                    "Tenure (Months)",
                    min_value=0,
                    max_value=72,
                    value=4,
                    help="Number of months customer has been subscribed.",
                )
                contract = st.selectbox(
                    "Contract Term",
                    options=["Month-to-month", "One year", "Two year"],
                    index=0,
                )
                paperless = st.selectbox("Paperless Billing", options=["Yes", "No"], index=0)
            with a2:
                payment = st.selectbox(
                    "Payment Method",
                    options=[
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    index=0,
                )
                monthly = st.number_input(
                    "Monthly Charges ($)",
                    min_value=18.0,
                    max_value=130.0,
                    value=79.85,
                    step=1.0,
                )
                default_total = round(monthly * max(tenure, 1), 2)
                total = st.number_input(
                    "Total Charges to Date ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=default_total,
                    step=10.0,
                )

            st.markdown(
                f"<div style='font-size: 0.96rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin: 16px 0 8px 0;'>Subscribed Services</div>",
                unsafe_allow_html=True,
            )
            s1, s2, s3 = st.columns(3)
            with s1:
                phone = st.selectbox("Phone Service", options=["Yes", "No"], index=0)
                multiple = st.selectbox(
                    "Multiple Lines", options=["No", "Yes", "No phone service"], index=0
                )
                internet = st.selectbox(
                    "Internet Service", options=["Fiber optic", "DSL", "No"], index=0
                )
            with s2:
                security = st.selectbox(
                    "Online Security", options=["No", "Yes", "No internet service"], index=0
                )
                backup = st.selectbox(
                    "Online Backup", options=["No", "Yes", "No internet service"], index=0
                )
                protection = st.selectbox(
                    "Device Protection", options=["No", "Yes", "No internet service"], index=0
                )
            with s3:
                support = st.selectbox(
                    "Technical Support", options=["No", "Yes", "No internet service"], index=0
                )
                tv = st.selectbox(
                    "Streaming TV", options=["Yes", "No", "No internet service"], index=0
                )
                movies = st.selectbox(
                    "Streaming Movies", options=["Yes", "No", "No internet service"], index=0
                )

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.form_submit_button("Predict Churn Risk", width="stretch")

    with col_result:
        st.markdown(
            f"<div style='font-size: 1.05rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 8px;'>Model Assessment</div>",
            unsafe_allow_html=True,
        )

        req = CustomerPredictionRequest(
            customer_id=customer_id or "CUST-ANON",
            gender=gender,
            SeniorCitizen=senior,
            Partner=partner,
            Dependents=dependents,
            tenure=tenure,
            PhoneService=phone,
            MultipleLines=multiple,
            InternetService=internet,
            OnlineSecurity=security,
            OnlineBackup=backup,
            DeviceProtection=protection,
            TechSupport=support,
            StreamingTV=tv,
            StreamingMovies=movies,
            Contract=contract,
            PaperlessBilling=paperless,
            PaymentMethod=payment,
            MonthlyCharges=monthly,
            TotalCharges=total,
        )

        try:
            res = predictor.predict_single(req)
        except Exception:
            st.error("Unable to evaluate this customer profile. Please check the entered values.")
            return

        # Accessible Soft Risk Banner
        render_risk_badge(res.risk_level, res.churn_probability)

        # Supporting Value Metrics
        m1, m2 = st.columns(2)
        with m1:
            render_metric_card(
                title="Customer Value (CLV)",
                value=f"${res.clv:,.2f}",
                subtext="Monthly Charges × Tenure",
                color_class="olive",
            )
        with m2:
            render_metric_card(
                title="Retention Priority",
                value=f"{res.retention_priority_score:,.1f}",
                subtext=f"Expected exposure: ₹{res.retention_priority_inr:,.0f}",
                color_class="terracotta" if res.risk_level in ["CRITICAL", "HIGH"] else "sage",
            )

        # Contributing Factors Section
        st.markdown(
            f"<div style='font-size: 0.96rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin: 16px 0 2px 0;'>Why this prediction?</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size: 0.80rem; color: {COLOR_SECONDARY_TEXT}; margin-bottom: 10px;'>Top factors influencing this customer's predicted risk:</div>",
            unsafe_allow_html=True,
        )

        for idx, driver in enumerate(res.top_drivers, start=1):
            render_driver_card(driver.feature, driver.direction, driver.impact, idx)

        # Expandable Methodology & Model Decisions
        with st.expander("How did the model decide?", expanded=False):
            st.markdown(
                """
                Each prediction can be explained using feature-level contribution values.
                These indicate which customer characteristics pushed the model's prediction toward
                higher or lower churn risk.
                
                * **Baseline comparison:** The model evaluates customer attributes against historical retention patterns across 7,043 subscriber profiles.
                * **Operational threshold:** Accounts with churn probabilities above **0.23** are classified as higher risk to ensure timely review by the retention team.
                * **Impact direction:** Factors either increase predicted risk (e.g. Month-to-month contracts, short tenure) or provide protective stability (e.g. Multi-year commitments, established tenure).
                """
            )

        # Neutral Suggested Review
        st.markdown(
            f"<div style='font-size: 0.96rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin: 16px 0 4px 0;'>Suggested Review</div>",
            unsafe_allow_html=True,
        )
        if res.risk_level in ["CRITICAL", "HIGH"]:
            if contract == "Month-to-month":
                action = "Review contract flexibility and explore annual commitment options before contacting the customer."
            elif internet == "Fiber optic" and support == "No":
                action = "Review broadband service stability and verify if technical assistance could reduce service friction."
            else:
                action = "Flag account for priority review by retention specialists prior to next billing cycle."
            st.info(f"**Suggested Review:** {action}")
        else:
            st.success(
                "**Suggested Review:** Account signals indicate normal engagement. Continue standard customer care."
            )
