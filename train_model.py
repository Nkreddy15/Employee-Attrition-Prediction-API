"""Demo training with fold-local preprocessing and profile-group splits."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from schema import CATEGORY_OPTIONS, CATEGORICAL_FIELDS, EXCLUDED_FIELDS, FEATURE_COLUMNS, NUMERIC_FIELDS, NUMERIC_RULES, SCHEMA_VERSION

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data/employee_attrition_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
TARGET_COLUMN = "Attrition (Target)"


def clean_training_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Apply deterministic checks only; learned imputations belong inside each fold."""
    if raw[TARGET_COLUMN].isna().any() or not raw[TARGET_COLUMN].isin(["Yes", "No"]).all():
        raise ValueError("Every target must be Yes or No; labels must never be imputed.")
    features = raw[FEATURE_COLUMNS].copy()
    invalid = {}
    for name, rule in NUMERIC_RULES.items():
        numeric = pd.to_numeric(features[name], errors="coerce").astype(float)
        valid = np.isfinite(numeric) & numeric.between(rule["min"], rule["max"])
        if name in {"Age", "Performance Rating"}:
            valid &= numeric.mod(1).eq(0)
        invalid[name] = int((features[name].notna() & ~valid).sum())
        features[name] = numeric.where(valid, np.nan)
    inconsistent = features["Years of Experience"] > features["Age"] - 18
    invalid["experience_exceeds_adult_age"] = int(inconsistent.sum())
    features.loc[inconsistent, "Years of Experience"] = np.nan
    for name, options in CATEGORY_OPTIONS.items():
        valid = features[name].isin(options)
        invalid[name] = int((features[name].notna() & ~valid).sum())
        features.loc[~valid, name] = np.nan
    audit = {
        "n_rows": len(raw),
        "raw_rows_with_missing_values": int(raw.isna().any(axis=1).sum()),
        "invalid_values_set_missing": invalid,
        "missing_after_checks": features.isna().sum().astype(int).to_dict(),
        "excluded_fields": EXCLUDED_FIELDS,
        "raw_csv_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "assumptions": ["Age >= 18; experience cannot exceed age minus 18 for this demo.",
                        "Salary period and distance units are unspecified by the source.",
                        "No income conversion or daily-to-weekly hours conversion is inferred."],
    }
    return features, raw[TARGET_COLUMN].map({"No": 0, "Yes": 1}), audit


def profile_groups(features: pd.DataFrame) -> np.ndarray:
    # Independent of targets and fitted imputers: identical observed profiles stay together.
    keys = features.astype(object).where(features.notna(), None).apply(
        lambda row: json.dumps(row.tolist(), separators=(",", ":")), axis=1
    )
    return pd.factorize(keys, sort=True)[0]


def build_pipeline(classifier=None) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessing = ColumnTransformer([("numeric", numeric, NUMERIC_FIELDS),
                                       ("categorical", categorical, CATEGORICAL_FIELDS)])
    if classifier is None:
        classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    return Pipeline([("preprocess", preprocessing), ("classifier", classifier)])


def evaluation_splits(features, target, groups):
    return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(features, target, groups))


def cross_validated_metrics(features, target, groups) -> tuple[dict, list]:
    splits = evaluation_splits(features, target, groups)
    scorers = {"accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0),
               "recall": make_scorer(recall_score, zero_division=0), "f1": make_scorer(f1_score, zero_division=0), "roc_auc": "roc_auc"}
    candidates = {
        "gradient_boosting": build_pipeline(),
        "logistic_regression": build_pipeline(LogisticRegression(max_iter=2000, random_state=42)),
        "majority_baseline": build_pipeline(DummyClassifier(strategy="most_frequent")),
    }
    results = {}
    for name, pipeline in candidates.items():
        scores = cross_validate(pipeline, features, target, cv=splits, scoring=scorers, error_score="raise")
        results[name] = {metric: {"mean": float(np.mean(scores[f"test_{metric}"])),
                                 "std": float(np.std(scores[f"test_{metric}"])),
                                 "fold_scores": scores[f"test_{metric}"].tolist()} for metric in scorers}
    split_audit = [{"fold": i + 1, "training_rows": len(train), "validation_rows": len(test),
                    "overlapping_profile_groups": len(set(groups[train]) & set(groups[test]))}
                   for i, (train, test) in enumerate(splits)]
    return results, split_audit


def build_reference_stats(features: pd.DataFrame, target: pd.Series, model, model_version: str) -> dict:
    # Preserve actual tied/discrete bin masses instead of assuming uniform quintiles.
    stats = {"n_rows": len(features), "model_version": model_version, "schema_version": SCHEMA_VERSION,
             "numeric": {}, "categorical": {}, "positive_rate": float(target.mean()),
             "reference_prediction_rate": float(model.predict(features).mean())}
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    numeric = pd.DataFrame(imputer.transform(features[NUMERIC_FIELDS]), columns=NUMERIC_FIELDS)
    for name in NUMERIC_FIELDS:
        values = numeric[name].to_numpy()
        edges = np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8]))
        counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
        stats["numeric"][name] = {
            "typical_min": float(values.min()), "typical_max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "bin_edges": edges.tolist(), "bin_probabilities": (counts / len(values)).tolist(),
            "imputation_value": float(imputer.statistics_[NUMERIC_FIELDS.index(name)]),
        }
    categorical_imputer = model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["impute"]
    categorical = pd.DataFrame(categorical_imputer.transform(features[CATEGORICAL_FIELDS]), columns=CATEGORICAL_FIELDS)
    for name in CATEGORICAL_FIELDS:
        stats["categorical"][name] = categorical[name].value_counts(normalize=True).to_dict()
    return stats


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    features, target, audit = clean_training_data(raw)
    groups = profile_groups(features)
    audit["n_unique_profile_groups"] = int(len(np.unique(groups)))
    audit["duplicate_profile_rows"] = len(features) - audit["n_unique_profile_groups"]
    audit["conflicting_label_profile_groups"] = int(pd.DataFrame({"group": groups, "target": target}).groupby("group")["target"].nunique().gt(1).sum())
    cv_metrics, splits = cross_validated_metrics(features, target, groups)
    model = build_pipeline()
    model.fit(features, target)
    seed = json.dumps({"data": audit["raw_csv_sha256"], "schema": SCHEMA_VERSION,
                       "sklearn": sklearn.__version__, "features": FEATURE_COLUMNS}, sort_keys=True)
    version = "v2-" + hashlib.sha256(seed.encode() + Path(__file__).read_bytes() + (BASE_DIR / "schema.py").read_bytes()).hexdigest()[:12]
    info = {
        "model_version": version, "schema_version": SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "n_training_rows": len(features), "n_features": len(FEATURE_COLUMNS), "features": FEATURE_COLUMNS,
        "evaluation": "5-fold StratifiedGroupKFold; identical cleaned observed profiles remain in one fold; fold-local imputation, encoding and scaling",
        "cross_validated_metrics": cv_metrics["gradient_boosting"], "model_comparison": cv_metrics,
        "folds": splits, "data_quality": audit, "probabilities_calibrated": False,
        "limitations": ["100 rows with repeated and conflicting profiles; provenance unverified.",
                        "Exact-profile grouping does not remove all similarity between near-duplicate profiles.",
                        "No independent external test set or validated subgroup fairness study.",
                        "Probabilities are uncalibrated model scores, not validated individual risk estimates."],
    }
    joblib.dump(model, MODEL_DIR / "employee_attrition_model.pkl")
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (MODEL_DIR / "reference_stats.json").write_text(json.dumps(build_reference_stats(features, target, model, version), indent=2) + "\n")
    (MODEL_DIR / "data_quality.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Trained {version}: {len(features)} rows, {audit['n_unique_profile_groups']} distinct profiles")
    for name, metrics in cv_metrics.items():
        print(f"{name}: accuracy {metrics['accuracy']['mean']:.3f} +/- {metrics['accuracy']['std']:.3f}; F1 {metrics['f1']['mean']:.3f}")


if __name__ == "__main__":
    main()
