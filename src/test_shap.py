from pathlib import Path
import shap
import matplotlib.pyplot as plt

from preprocessing import (
    load_data,
    clean_data,
    get_train_test_data,
    build_preprocessor
)
from model import train_xgboost
from shap_utils import create_shap_explainer, explain_single_customer, get_top_features
from recommendation_engine import generate_precautions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"

df = load_data(DATA_PATH)
df = clean_data(df)

X_train, X_test, y_train, y_test = get_train_test_data(df)
preprocessor = build_preprocessor(X_train)

xgb_pipeline = train_xgboost(preprocessor, X_train, y_train)

explainer, shap_values, X_train_processed = create_shap_explainer(
    xgb_pipeline,
    X_train
)

print("Showing global feature importance...")
shap.summary_plot(shap_values, X_train_processed)

single_customer = X_test.iloc[[0]]

single_shap = explain_single_customer(
    xgb_pipeline,
    explainer,
    single_customer
)

feature_names = xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()

top_risk_features = get_top_features(
    single_shap,
    feature_names,
    top_n=3
)

precautions = generate_precautions(top_risk_features)

print("\nTop Risk Drivers:")
for f in top_risk_features:
    print("-", f)

print("\nRecommended Precautions:")
for p in precautions:
    print("-", p)
