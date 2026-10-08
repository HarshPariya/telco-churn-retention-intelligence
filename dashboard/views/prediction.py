"""Customer Churn Prediction view with real-time SHAP explanation."""

import streamlit as st

from dashboard.components.risk_card import (
    render_metric_card,
    render_risk_badge,
)
from src.telco_churn.inference.predictor import (
    CustomerPredictionRequest,
    get_predictor,
)


def render_customer_prediction() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px;">
                Customer Churn Prediction
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem; margin-bottom: 12px;">
                Enter the customer's current profile to estimate churn risk and understand the main factors influencing the prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # User Guidance Workflow
    st.markdown(
        """
        <div style="
            background: rgba(30, 41, 59, 0.6);
            border-left: 4px solid #6366f1;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 24px;
            font-size: 0.88rem;
            color: #cbd5e1;
        ">
            <b>Workflow Guide:</b>
            <span style="margin-left: 8px;">1. Enter customer details</span> →
            <span style="margin-left: 4px;">2. Click <b>"Predict Churn Risk"</b></span> →
            <span style="margin-left: 4px;">3. Review risk & customer value</span> →
            <span style="margin-left: 4px;">4. Inspect main risk drivers</span> →
            <span style="margin-left: 4px;">5. Prioritize retention review</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        predictor = get_predictor()
    except Exception:
        st.error(
            "The prediction model is currently unavailable. Please verify the model service or run `python scripts/train_model.py`."
        )
        return

    col_input, col_result = st.columns([1.1, 0.9])

    with col_input:
        with st.form("customer_prediction_form"):
            st.markdown("### 1. Customer Profile")
            customer_id = st.text_input(
                "Customer Account ID",
                value="CUST-7590-VH",
                help="Unique identifier for the subscriber account.",
            )

            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", options=["Female", "Male"])
                senior = st.selectbox(
                    "Senior Citizen (Age ≥ 65)",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Indicates whether the customer is 65 years or older.",
                )
            with c2:
                partner = st.selectbox(
                    "Partner",
                    options=["No", "Yes"],
                    help="Indicates if the customer has a partner.",
                )
                dependents = st.selectbox(
                    "Dependents",
                    options=["No", "Yes"],
                    help="Indicates if the customer lives with dependents.",
                )

            st.markdown("### 2. Account & Billing")
            a1, a2 = st.columns(2)
            with a1:
                tenure = st.slider(
                    "Customer Tenure (Months)",
                    min_value=0,
                    max_value=72,
                    value=4,
                    help="Total number of months the customer has stayed with the company.",
                )
                contract = st.selectbox(
                    "Contract Type",
                    options=["Month-to-month", "One year", "Two year"],
                    index=0,
                    help="Contract commitment term.",
                )
                paperless = st.selectbox(
                    "Paperless Billing",
                    options=["Yes", "No"],
                    index=0,
                    help="Whether paperless digital billing is enabled.",
                )
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
                    help="Primary billing method used by the account.",
                )
                monthly = st.number_input(
                    "Monthly Charges ($)",
                    min_value=18.0,
                    max_value=130.0,
                    value=79.85,
                    step=1.0,
                    help="Current monthly subscription cost.",
                )
                calc_total = round(monthly * max(tenure, 1), 2)
                total = st.number_input(
                    "Total Charges ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=calc_total,
                    step=10.0,
                    help="Cumulative lifetime billing charges to date.",
                )

            st.markdown("### 3. Services")
            s1, s2, s3 = st.columns(3)
            with s1:
                phone = st.selectbox("Phone Service", options=["Yes", "No"], index=0)
                multiple = st.selectbox(
                    "Multiple Lines",
                    options=["No", "Yes", "No phone service"],
                    index=0,
                )
                internet = st.selectbox(
                    "Internet Service",
                    options=["Fiber optic", "DSL", "No"],
                    index=0,
                )

            with s2:
                security = st.selectbox(
                    "Online Security",
                    options=["No", "Yes", "No internet service"],
                    index=0,
                )
                backup = st.selectbox(
                    "Online Backup",
                    options=["No", "Yes", "No internet service"],
                    index=0,
                )
                protection = st.selectbox(
                    "Device Protection",
                    options=["No", "Yes", "No internet service"],
                    index=0,
                )

            with s3:
                support = st.selectbox(
                    "Tech Support",
                    options=["No", "Yes", "No internet service"],
                    index=0,
                )
                tv = st.selectbox(
                    "Streaming TV",
                    options=["Yes", "No", "No internet service"],
                    index=0,
                )
                movies = st.selectbox(
                    "Streaming Movies",
                    options=["Yes", "No", "No internet service"],
                    index=0,
                )

            st.form_submit_button(
                "⚡ Predict Churn Risk", width="stretch"
            )

    with col_result:
        st.markdown("### 🎯 Prediction & Diagnostic Output")

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
            with st.spinner("Calculating calibrated churn risk & SHAP attributions..."):
                res = predictor.predict_single(req)
        except Exception:
            st.error(
                "Prediction failed. Please ensure all customer profile fields have valid values."
            )
            return

        # Render Risk Badge
        render_risk_badge(res.risk_level, res.churn_probability)

        # Value & Priority Metrics
        m1, m2 = st.columns(2)
        with m1:
            render_metric_card(
                title="Customer Lifetime Value",
                value=f"${res.clv:,.2f}",
                subtext=f"≈ ₹{res.clv_inr:,.0f} Historical Spend",
                color_class="indigo",
            )
        with m2:
            render_metric_card(
                title="Retention Priority",
                value=f"{res.retention_priority_score:,.1f}",
                subtext=f"Expected Loss: ₹{res.retention_priority_inr:,.0f}",
                color_class="rose" if res.risk_level in ["CRITICAL", "HIGH"] else "emerald",
            )

        # Top 3 Drivers Section
        st.markdown("#### Why is this customer at risk?")
        st.caption("Top 3 influential drivers identified by the model:")

        for idx, driver in enumerate(res.top_drivers, start=1):
            is_risk = driver.direction == "INCREASES_CHURN"
            arrow = "🔺 Increases predicted churn risk" if is_risk else "🔻 Protects retention (reduces risk)"
            color = "#f43f5e" if is_risk else "#10b981"
            bg = "rgba(244, 63, 94, 0.08)" if is_risk else "rgba(16, 185, 129, 0.08)"

            st.markdown(
                f"""
                <div style="
                    background: {bg};
                    border-left: 4px solid {color};
                    border-radius: 6px;
                    padding: 10px 14px;
                    margin-bottom: 8px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">DRIVER #{idx}</div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">{driver.feature}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.8rem; font-weight: 600; color: {color};">{arrow}</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">Attribution Impact: +{driver.impact:.3f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Expandable simple explanation
        with st.expander("How did the model decide?", expanded=False):
            st.markdown(
                """
                **How Churn Risk is Calculated:**
                - Our machine learning model (Extreme Gradient Boosting) reviews historical account patterns across 7,043 customers.
                - Each factor (such as having a Month-to-month contract or a short tenure) pushes the predicted churn risk higher or lower.
                - We use **SHAP (Shapley Additive Explanations)**, a mathematical framework from cooperative game theory, to measure the exact contribution of each profile feature to the final prediction.
                - When features push the probability above our cost-optimized decision threshold (**0.23**), the account is flagged for proactive retention outreach.
                """
            )

        # Suggested Next Step
        st.markdown("#### Suggested Next Step")
        if res.risk_level in ["CRITICAL", "HIGH"]:
            if contract == "Month-to-month":
                action = "Offer an annual contract upgrade with a 10% loyalty incentive and complimentary onboarding."
            elif internet == "Fiber optic" and support == "No":
                action = "Provide a complimentary Tech Support add-on and conduct a remote line diagnostic check."
            else:
                action = "Schedule a proactive customer success call with a tailored retention bundle."
            st.warning(f"**Recommended Action:** {action}")
        else:
            st.success("**Account Healthy:** Maintain regular engagement. No promotional discount required.")
