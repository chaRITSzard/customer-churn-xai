from pathlib import Path
import shap
import matplotlib.pyplot as plt

from preprocessing import(
    load_data,
    clean_data,
    get_train_test_data,
    build_preprocessor
)
from model import train_xgboost
from shap_utils import create_shap_explainer, explain_single_customer

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

print("Showing explanation for one customer...")
shap.waterfall_plot(single_shap[0])
