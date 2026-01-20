from pathlib import Path
from preprocessing import load_data, clean_data, get_train_test_data, build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"

df = load_data(DATA_PATH)
df = clean_data(df)

X_train, X_test, y_train, y_test = get_train_test_data(df)

preprocessor = build_preprocessor(X_train)
X_train_processed = preprocessor.fit_transform(X_train)

print("Original shape:", X_train.shape)
print("Processed shape:", X_train_processed.shape)
print("Churn rate:", y_train.mean())
