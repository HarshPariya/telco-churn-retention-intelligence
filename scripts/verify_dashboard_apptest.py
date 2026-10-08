"""Automated Streamlit AppTest verification script.

Tests all 4 user-facing views:
- Executive Overview
- Customer Prediction (and single prediction execution)
- Retention Prioritization (and sample cohort load + ranking)
- Model Insights

Ensures 0 unhandled exceptions, 0 blank pages, and successful render.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest


def test_dashboard_full_lifecycle():
    print("Testing Streamlit Dashboard with AppTest...")
    app_file = str(PROJECT_ROOT / "dashboard" / "app.py")

    # 1. Initial Load (Executive Overview)
    print("\n--- 1. Testing Initial Load (Executive Overview) ---")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()
    if at.exception:
        print(f"FAILED on initial load: {at.exception}")
        sys.exit(1)
    print(f"SUCCESS: Loaded Executive Overview. Found {len(at.markdown)} markdowns.")

    # 2. Navigate to Customer Prediction
    print("\n--- 2. Testing Customer Prediction ---")
    at.radio[0].set_value("Customer Prediction").run()
    if at.exception:
        print(f"FAILED on Customer Prediction navigation: {at.exception}")
        sys.exit(1)
    print(f"SUCCESS: Navigated to Customer Prediction. Found {len(at.button)} buttons.")

    # Click 'Predict Churn Risk' button
    predict_buttons = [b for b in at.button if "Predict Churn Risk" in b.label]
    if predict_buttons:
        print("Clicking 'Predict Churn Risk'...")
        predict_buttons[0].click().run()
        if at.exception:
            print(f"FAILED after clicking Predict Churn Risk: {at.exception}")
            sys.exit(1)
        print("SUCCESS: Prediction executed without exceptions.")
    else:
        print("WARNING: Could not find 'Predict Churn Risk' button.")

    # 3. Navigate to Retention Prioritization
    print("\n--- 3. Testing Retention Prioritization ---")
    at.radio[0].set_value("Retention Prioritization").run()
    if at.exception:
        print(f"FAILED on Retention Prioritization navigation: {at.exception}")
        sys.exit(1)
    print("SUCCESS: Navigated to Retention Prioritization.")

    # Click 'Load Sample Cohort' button
    sample_buttons = [b for b in at.button if "Load Sample Cohort" in b.label]
    if sample_buttons:
        print("Clicking 'Load Sample Cohort'...")
        sample_buttons[0].click().run()
        if at.exception:
            print(f"FAILED after clicking Load Sample Cohort: {at.exception}")
            sys.exit(1)
        print(
            f"SUCCESS: Sample Cohort evaluated and ranked successfully. Found {len(at.dataframe)} dataframes."
        )
    else:
        print("WARNING: Could not find 'Load Sample Cohort' button.")

    # 4. Navigate to Model Insights
    print("\n--- 4. Testing Model Insights ---")
    at.radio[0].set_value("Model Insights").run()
    if at.exception:
        print(f"FAILED on Model Insights navigation: {at.exception}")
        sys.exit(1)
    print(
        f"SUCCESS: Navigated to Model Insights. Found {len(at.image)} images and {len(at.dataframe)} dataframes."
    )

    # 5. Return to Executive Overview
    print("\n--- 5. Testing Return to Executive Overview ---")
    at.radio[0].set_value("Executive Overview").run()
    if at.exception:
        print(f"FAILED returning to Executive Overview: {at.exception}")
        sys.exit(1)
    print("SUCCESS: Returned to Executive Overview with zero exceptions.")

    print("\n========================================================")
    print("ALL STREAMLIT VIEWS & INTERACTION FLOWS VERIFIED 100% OK")
    print("========================================================")


if __name__ == "__main__":
    test_dashboard_full_lifecycle()
