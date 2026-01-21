import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

def train_logistic_regression(preprocessor, X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def train_xgboost(preprocessor, X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def eval_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)

    print("ROC-AUC:", round(roc_auc, 4))
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))