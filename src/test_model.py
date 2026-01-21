from pathlib import Path

from preprocessing import (
    load_data,
    clean_data,
    get_train_test_data,
    build_preprocessor
)

from model import (
    train_logistic_regression,
    train_xgboost,
    eval_model
)


def main():
    # ------------------------------------------------------------------
    # Resolve project root safely
    # ------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"

    # ------------------------------------------------------------------
    # Load & clean data
    # ------------------------------------------------------------------
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # ------------------------------------------------------------------
    # Train-test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = get_train_test_data(df)

    # ------------------------------------------------------------------
    # Build preprocessor -----(DO NOT fit it here)------
    # ------------------------------------------------------------------
    preprocessor = build_preprocessor(X_train)

    print("\n================ Logistic Regression =================")
    lr_pipeline = train_logistic_regression(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train
    )

    eval_model(
        model=lr_pipeline,
        X_test=X_test,
        y_test=y_test
    )

    print("\n================ XGBoost =================")
    xgb_pipeline = train_xgboost(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train
    )

    eval_model(
        model=xgb_pipeline,
        X_test=X_test,
        y_test=y_test
    )


if __name__ == "__main__":
    main()
