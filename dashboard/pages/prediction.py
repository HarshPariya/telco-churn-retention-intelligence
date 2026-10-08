"""Single Customer Prediction & Explainability page."""

import streamlit as st

from dashboard.components.risk_card import (
    render_driver_card,
    render_metric_card,
    render_risk_badge,
)
from src.telco_churn.inference.predictor import (
    CustomerPredictionRequest,
    get_predictor,
)


def render_prediction_page() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px;">
                Single Customer Churn Diagnostic
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem;">
                Enter a customer profile to receive real-time churn probability, risk classification, Customer Lifetime Value (CLV), and exact top-3 SHAP drivers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Could not load inference engine: {e}")
        return

    # Two column layout: Input Form on left, Prediction Results on right
    col_input, col_result = st.columns([1.1, 0.9])

    with col_input:
        st.markdown("### 📋 Customer Profile Parameters")
        with st.form("customer_prediction_form"):
            customer_id = st.text_input("Customer Account ID", value="CUST-7590-VH")

            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                gender = st.selectbox("Gender", options=["Female", "Male"])
                senior = st.selectbox(
                    "Senior Citizen (≥ 65)",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                )
                partner = st.selectbox("Partner", options=["No", "Yes"])
                dependents = st.selectbox("Dependents", options=["No", "Yes"])
                tenure = st.slider("Tenure with Telco (Months)", min_value=0, max_value=72, value=4)

            with c_sub2:
                contract = st.selectbox(
                    "Contract Term",
                    options=["Month-to-month", "One year", "Two year"],
                    index=0,
                )
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
                paperless = st.selectbox("Paperless Billing", options=["Yes", "No"], index=0)
                monthly = st.number_input(
                    "Monthly Charges ($)", min_value=18.0, max_value=130.0, value=79.85, step=1.0
                )
                # Compute default TotalCharges
                default_total = round(monthly * max(tenure, 1), 2)
                total = st.number_input(
                    "Total Charges ($)",
                    min_value=0.0,
                    max_value=9000.0,
                    value=default_total,
                    step=10.0,
                )

            st.markdown("#### 🌐 Subscribed Services")
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                phone = st.selectbox("Phone Service", options=["Yes", "No"], index=0)
                multiple = st.selectbox(
                    "Multiple Lines", options=["No", "Yes", "No phone service"], index=0
                )
                internet = st.selectbox(
                    "Internet Service", options=["Fiber optic", "DSL", "No"], index=0
                )

            with s_col2:
                security = st.selectbox(
                    "Online Security", options=["No", "Yes", "No internet service"], index=0
                )
                backup = st.selectbox(
                    "Online Backup", options=["No", "Yes", "No internet service"], index=0
                )
                protection = st.selectbox(
                    "Device Protection", options=["No", "Yes", "No internet service"], index=0
                )

            with s_col3:
                support = st.selectbox(
                    "Tech Support", options=["No", "Yes", "No internet service"], index=0
                )
                tv = st.selectbox(
                    "Streaming TV", options=["Yes", "No", "No internet service"], index=0
                )
                movies = st.selectbox(
                    "Streaming Movies", options=["Yes", "No", "No internet service"], index=0
                )

            st.form_submit_button(
                "⚡ Evaluate Churn Risk & Explain", use_container_width=True
            )

    with col_result:
        st.markdown("### 🎯 Diagnostic & Retention Output")
        # Run default prediction or on submit
        req = CustomerPredictionRequest(
            customer_id=customer_id,
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

        with st.spinner("Executing calibrated inference pipeline & SHAP explainer..."):
            res = predictor.predict_single(req)

        # Risk Badge
        render_risk_badge(res.risk_level, res.churn_probability)

        # Metrics Row
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            render_metric_card(
                title="Customer Lifetime Value",
                value=f"${res.clv:,.2f}",
                subtext=f"₹{res.clv_inr:,.0f} Total Revenue",
                color_class="indigo",
            )
        with m_col2:
            render_metric_card(
                title="Retention Priority",
                value=f"{res.retention_priority_score:,.1f}",
                subtext=f"Expected Loss: ₹{res.retention_priority_inr:,.0f}",
                color_class="rose" if res.risk_level in ["CRITICAL", "HIGH"] else "emerald",
            )

        st.markdown("#### 🔍 Top 3 Explanatory Risk Drivers (SHAP)")
        st.caption("Derived from tree Shapley values mapping exact feature contribution:")
        for idx, driver in enumerate(res.top_drivers, start=1):
            render_driver_card(driver.feature, driver.direction, driver.impact, idx)

        # Recommended Action
        st.markdown("#### 🎯 Prescriptive Retention Action")
        if res.risk_level in ["CRITICAL", "HIGH"]:
            if contract == "Month-to-month":
                action = "Offer 1-year contract extension with a 10% loyalty discount & free Tech Support onboarding."
            elif internet == "Fiber optic" and support == "No":
                action = "Complimentary Tech Support bundle and home Wi-Fi optimization check."
            else:
                action = (
                    "Priority outreach by senior account manager with customized retention bundle."
                )
            st.warning(f"**Recommended Intervention:** {action}")
        else:
            st.success(
                "**Customer is Stable:** Maintain standard service engagement. No discount required."
            )
