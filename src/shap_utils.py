import shap
import numpy as np
import pandas as pd

def create_shap_explainer(trained_pipeline, X_train):
    """
       Creates a SHAP TreeExplainer using the trained XGBoost model
       and preprocessed training data.
    """
    preprocessor = trained_pipeline.named_steps['preprocessor']
    model = trained_pipeline.named_steps['model']

    X_train_processed = preprocessor.transform(X_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train_processed)
    return explainer, shap_values, X_train_processed

def explain_single_customer(trained_pipeine, explainer, X_customer):
    """
    RETURNS SHAP values for a single customer
    """
    preprocessor = trained_pipeine.named_steps['preprocessor']
    model = trained_pipeine.named_steps['model']

    X_processed = preprocessor.transform(X_customer)
    shap_values = explainer(X_processed)
    return shap_values


def get_top_features(shap_values, feature_names, top_n = 3):
    shap_array = shap_values.values[0]
    feature_contri = list(zip(feature_names, shap_array))

    positive_feats = [
        (feat, val) for feat, val in feature_contri if val > 0
    ]
    positive_feats.sort(key=lambda x: x[1], reverse=True)

    return [f[0] for f in positive_feats[:top_n]]