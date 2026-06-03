from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = Path("data/employee_attrition_dataset.csv")
MODEL_DIR = Path("models")

TARGET_COLUMN = "Attrition (Target)"
DROP_COLUMNS = ["Employee ID", TARGET_COLUMN]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Education Level",
    "Department",
    "Job Role",
    "Commute Method",
    "Marital Status",
]

NUMERICAL_COLUMNS = [
    "Age",
    "Salary (USD)",
    "Income (Euro)",
    "Years of Experience",
    "Performance Rating",
    "Working Hours",
    "Distance from Home",
]


def preprocess_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder], List[str]]:
    df = df.copy()

    for col in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in NUMERICAL_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    label_encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        label_encoders[col] = encoder

    target_encoder = LabelEncoder()
    y = pd.Series(target_encoder.fit_transform(df[TARGET_COLUMN].astype(str)), name=TARGET_COLUMN)

    X = df.drop(columns=DROP_COLUMNS)
    feature_columns = X.columns.tolist()

    return X, y, label_encoders, feature_columns


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X, y, label_encoders, feature_columns = preprocess_training_data(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    joblib.dump(model, MODEL_DIR / "employee_attrition_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(label_encoders, MODEL_DIR / "label_encoders.pkl")
    joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")
    joblib.dump(metrics, MODEL_DIR / "metrics.pkl")

    print("Model training complete. Artifacts saved in the models/ folder.\n")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
