"""Check evaluation separation and drift false alarms against the training distribution."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_report import build_report, main
from train_model import DATA_PATH, build_pipeline, clean_training_data, evaluation_splits, profile_groups

ROOT = Path(__file__).resolve().parents[1]


def test_identical_profiles_never_cross_evaluation_folds():
    features, target, audit = clean_training_data(pd.read_csv(DATA_PATH))
    groups = profile_groups(features)
    assert len(np.unique(groups)) < len(features)
    seen_validation = []
    for train, validation in evaluation_splits(features, target, groups):
        assert not set(groups[train]) & set(groups[validation])
        seen_validation.extend(validation)
        pipeline = build_pipeline().fit(features.iloc[train], target.iloc[train])
        imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        assert imputer.statistics_[0] == pytest.approx(features.iloc[train]["Age"].median())
    assert sorted(seen_validation) == list(range(len(features)))


def test_bad_source_values_are_missing_and_target_is_never_imputed():
    raw = pd.read_csv(DATA_PATH)
    features, target, audit = clean_training_data(raw)
    assert features.loc[21, "Performance Rating"] != features.loc[21, "Performance Rating"]
    assert pd.isna(features.loc[27, "Commute Method"])
    assert set(features) == set(json.loads((ROOT / "examples/request.json").read_text()))
    raw.loc[0, "Attrition (Target)"] = None
    with pytest.raises(ValueError, match="labels must never be imputed"):
        clean_training_data(raw)


def reference_rows():
    import joblib
    from schema import CATEGORICAL_FIELDS, NUMERIC_FIELDS
    features, target, _ = clean_training_data(pd.read_csv(DATA_PATH))
    pipeline = joblib.load(ROOT / "models/employee_attrition_model.pkl")
    reference = json.loads((ROOT / "models/reference_stats.json").read_text())
    preprocessor = pipeline.named_steps["preprocess"]
    features[NUMERIC_FIELDS] = preprocessor.named_transformers_["numeric"].named_steps["impute"].transform(features[NUMERIC_FIELDS])
    features[CATEGORICAL_FIELDS] = preprocessor.named_transformers_["categorical"].named_steps["impute"].transform(features[CATEGORICAL_FIELDS])
    predictions = pipeline.predict(features)
    rows = [{"inputs": data, "prediction": "Yes" if label else "No", "model_version": reference["model_version"],
             "schema_version": reference["schema_version"]} for data, label in zip(features.to_dict("records"), predictions)]
    return rows, reference


def test_same_distribution_does_not_raise_false_drift_for_discrete_features():
    rows, reference = reference_rows()
    report = build_report(rows, reference)
    assert report["status"] == "checked"
    assert all(feature["psi"] == pytest.approx(0, abs=1e-6) for feature in report["features"])
    assert report["prediction_shift"]["abs_difference"] == pytest.approx(0)


def test_shifted_population_alerts_and_cli_exits_one(tmp_path, capsys):
    rows, reference = reference_rows()
    for row in rows:
        row["inputs"]["Age"] = 100
    path = tmp_path / "shifted.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert main(["--log", str(path), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "alert"


def test_small_missing_old_version_and_malformed_logs_are_not_healthy_verdicts(tmp_path, capsys):
    path = tmp_path / "missing.jsonl"
    assert main(["--log", str(path), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "insufficient_data"
    rows, reference = reference_rows()
    for row in rows:
        row["model_version"] = "old-version"
    path.write_text("not-json\n" + "\n".join(json.dumps(row) for row in rows))
    assert main(["--log", str(path), "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["excluded_rows"] == 100 and report["malformed_rows"] == 1
