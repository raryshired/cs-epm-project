"""
Engine Condition Prediction - Streamlit Web Application
Loads trained Gradient Boosting model from Hugging Face Model Hub
Feature order loaded dynamically from model_metadata.json
"""
import os
import json
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Constants
HF_USER_NAME = "spac1ngcat"
HF_MODEL_REPO_NAME = "cs-pred-maintain-model"
MODEL_FILENAME = "best_model.joblib"
METADATA_FILENAME = "model_metadata.json"

# HF Token for authenticated downloads (optional for public repos, recommended for rate limits)
HF_TOKEN = os.getenv("HF_TOKEN")


@st.cache_resource
def load_model_and_metadata():
    """
    Load model and metadata from Hugging Face Model Hub with caching.
    Returns tuple of (model, metadata_dict)
    """
    repo_id = f"{HF_USER_NAME}/{HF_MODEL_REPO_NAME}"

    # Download model
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=MODEL_FILENAME,
        token=HF_TOKEN
    )
    model = joblib.load(model_path)

    # Download metadata for feature order
    metadata_path = hf_hub_download(
        repo_id=repo_id,
        filename=METADATA_FILENAME,
        token=HF_TOKEN
    )
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return model, metadata


# Load model and metadata
model, metadata = load_model_and_metadata()

# Extract configuration from metadata
FEATURE_COLUMNS = metadata.get('feature_columns', [])
CLASS_MAPPING = metadata.get('class_mapping', {'0': 'Normal', '1': 'Faulty'})
TEST_METRICS = metadata.get('test_metrics', {})

# Page configuration
st.set_page_config(
    page_title="Engine Condition Prediction",
    page_icon="🔧",
    layout="centered"
)

# App header
st.title("Engine Predictive Maintenance")
st.markdown("### Classify engine condition as Normal or Faulty")

# Input form with 6 sensor features
st.markdown("#### Enter Sensor Readings:")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        engine_rpm = st.number_input(
            "Engine RPM",
            min_value=0.0,
            max_value=5000.0,
            value=1500.0,
            step=50.0,
            help="Engine revolutions per minute"
        )
        lub_oil_pressure = st.number_input(
            "Lub Oil Pressure",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1,
            help="Lubricating oil pressure in bar"
        )
        fuel_pressure = st.number_input(
            "Fuel Pressure",
            min_value=0.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
            help="Fuel system pressure in bar"
        )

    with col2:
        coolant_pressure = st.number_input(
            "Coolant Pressure",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.1,
            help="Coolant system pressure in bar"
        )
        lub_oil_temp = st.number_input(
            "Lub Oil Temperature (°C)",
            min_value=50.0,
            max_value=150.0,
            value=80.0,
            step=1.0,
            help="Lubricating oil temperature"
        )
        coolant_temp = st.number_input(
            "Coolant Temperature (°C)",
            min_value=50.0,
            max_value=250.0,
            value=85.0,
            step=1.0,
            help="Engine coolant temperature"
        )

    submitted = st.form_submit_button("Predict Engine Condition", use_container_width=True)

if submitted:
    # Create input dataframe with feature order from metadata
    input_dict = {
        'Engine_RPM': engine_rpm,
        'Lub_Oil_Pressure': lub_oil_pressure,
        'Fuel_Pressure': fuel_pressure,
        'Coolant_Pressure': coolant_pressure,
        'Lub_Oil_Temperature': lub_oil_temp,
        'Coolant_Temperature': coolant_temp
    }
    # Ensure column order matches training
    input_data = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Display results
    st.markdown("#### Prediction Results:")

    col_result1, col_result2 = st.columns(2)

    with col_result1:
        if prediction == 1:
            st.error(f"### {CLASS_MAPPING.get('1', 'FAULTY')}")
            st.caption("Engine requires maintenance")
        else:
            st.success(f"### {CLASS_MAPPING.get('0', 'NORMAL')}")
            st.caption("Engine operating normally")

    with col_result2:
        st.metric(
            "Confidence",
            f"{max(probabilities) * 100:.1f}%"
        )

    # Probability breakdown
    st.markdown("##### Prediction Probabilities:")
    prob_df = pd.DataFrame({
        'Condition': [f"{CLASS_MAPPING.get('0', 'Normal')} (0)", f"{CLASS_MAPPING.get('1', 'Faulty')} (1)"],
        'Probability': [f"{probabilities[0]*100:.2f}%", f"{probabilities[1]*100:.2f}%"]
    })
    st.table(prob_df)

    # Model info with actual metrics from metadata
    with st.expander("About this model"):
        accuracy = TEST_METRICS.get('accuracy', 0) * 100
        f1 = TEST_METRICS.get('f1_score', 0) * 100
        recall = TEST_METRICS.get('recall', 0) * 100
        precision = TEST_METRICS.get('precision', 0) * 100

        st.markdown(f"""
        **Model Details:**
        - Algorithm: Gradient Boosting (sklearn)
        - Training samples: 13,674
        - Test Accuracy: {accuracy:.2f}%
        - Test F1-Score: {f1:.2f}%
        - Test Recall: {recall:.2f}%
        - Test Precision: {precision:.2f}%

        **Design Philosophy:**
        - High recall ({recall:.1f}%) for Faulty class - designed to catch potential issues
        - F1-Score optimized for class imbalance handling
        - Feature order validated against model_metadata.json
        """)
