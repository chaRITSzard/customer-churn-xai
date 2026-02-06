import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import shap

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import (
    load_data,
    clean_data,
    get_train_test_data,
    build_preprocessor
)

from model import train_xgboost
from shap_utils import (
    create_shap_explainer,
    explain_single_customer,
    get_top_features
)

from recommendation_engine import generate_precautions

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"

df = load_data(DATA_PATH)
df = clean_data(df)

X_train, X_test, y_train, y_test = get_train_test_data(df)
preprocessor = build_preprocessor(X_train)

xgb_pipeline = train_xgboost(preprocessor, X_train, y_train)

explainer, shap_values, X_train_processed = create_shap_explainer(xgb_pipeline, X_train)

feature_names = xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()

#STREAMLIT APP UI

st.title("📊Customer Churn Dashboard")
customer_index = st.slider(
    "Select Customer Index",
    0,
    len(X_test) - 1,
    0
)

customer = X_test.iloc[[customer_index]]
churn_prob = xgb_pipeline.predict_proba(customer)[0][1]

st.subheader("Churn Risk")
st.metric("Probability of Churn", f"{churn_prob:.2f}")
single_shap = explain_single_customer(
    xgb_pipeline,
    explainer,
    customer
)

top_risk_features = get_top_features(
    single_shap,
    feature_names,
    top_n=3
)

precautions = generate_precautions(top_risk_features)

st.subheader("Why this customer may churn")

fig, ax = plt.subplots()
shap.waterfall_plot(single_shap[0], show=False)
st.pyplot(fig)

st.subheader("Recommended Precautions")
for p in precautions:
    st.write(f"- {p}")
