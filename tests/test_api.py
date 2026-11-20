"""Contract, serving, observability and migration regressions."""
import json
import os
from pathlib import Path

import pytest

os.environ["PREDICTION_LOG"] = "off"
from app import app, metrics, prediction_logger
from monitoring import PredictionLogger

VALID = json.loads((Path(__file__).resolve().parents[1] / "examples/request.json").read_text())


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health_and_schema_identify_the_loaded_model(client):
    health = client.get("/health")
    schema = client.get("/schema").get_json()
    assert health.status_code == 200 and health.get_json()["model_loaded"]
    assert health.get_json()["schema_version"] == schema["schema_version"] == "2"
    assert set(schema["required_fields"]) == set(VALID)


def test_readme_example_is_accepted(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    result = response.get_json()
    assert result["probability_yes"] + result["probability_no"] == pytest.approx(1)
    assert result["attrition_prediction"] in {"Yes", "No"}
    assert result["probabilities_calibrated"] is False
    assert result["risk_factors"] == []
    assert result["warnings"] == []
    assert response.headers["X-Request-ID"]


def test_form_renders_and_accepts_the_same_contract(client):
    assert client.get("/form").status_code == 200
    response = client.post("/form", data=VALID)
    assert response.status_code == 200
    assert b"Model score (uncalibrated)" in response.data
    assert b"Top Rule-Based Risk Factors" not in response.data


def test_form_invalid_input_returns_error_status_and_keeps_input(client):
    response = client.post("/form", data={**VALID, "Age": "150"})
    assert response.status_code == 400 and b'Error:' in response.data and b'value="150"' in response.data


@pytest.mark.parametrize("value", [None, True, False, "NaN", "inf", "-inf", [], {}, 150, 20.5, "abc"])
def test_invalid_numbers_are_rejected_without_500(client, value):
    response = client.post("/predict", json={**VALID, "Age": value})
    assert response.status_code == 400 and "error" in response.get_json()


@pytest.mark.parametrize("body", [None, [], "not an object", 17])
def test_nonobject_body_is_rejected(client, body):
    assert client.post("/predict", json=body).status_code == 400


def test_invalid_category_and_missing_and_extra_fields(client):
    assert client.post("/predict", json={**VALID, "Job Role": "Sales Executive"}).status_code == 400
    assert client.post("/predict", json={k: v for k, v in VALID.items() if k != "Age"}).status_code == 400
    assert client.post("/predict", json={**VALID, "employee_email": "do-not-log@example.org"}).status_code == 400


def test_age_experience_inconsistency_is_rejected(client):
    assert client.post("/predict", json={**VALID, "Age": 19, "Years of Experience": 15}).status_code == 400


def test_legacy_units_and_demographics_are_ignored_with_explicit_warnings(client):
    original = client.post("/predict", json=VALID).get_json()
    legacy = client.post("/predict", json={**VALID, "Income (Euro)": 350000, "Working Hours": 9,
                                           "Gender": "Female", "Marital Status": "Single"})
    assert legacy.status_code == 200
    result = legacy.get_json()
    assert len(result["warnings"]) == 4
    assert result["probability_yes"] == original["probability_yes"]


def test_range_warnings_are_returned_in_api_and_form(client):
    data = {**VALID, "Salary (USD)": 200000}
    assert "outside the observed training range" in " ".join(client.post("/predict", json=data).get_json()["warnings"])
    assert b"outside the observed training range" in client.post("/form", data=data).data


def test_metrics_count_success_errors_and_extrapolation(client):
    before = metrics.snapshot()
    client.post("/predict", json={**VALID, "Salary (USD)": 200000})
    client.post("/predict", json={**VALID, "Age": None})
    after = metrics.snapshot()
    assert sum(after["predictions"].values()) == sum(before["predictions"].values()) + 1
    assert after["validation_errors"] == before["validation_errors"] + 1
    assert after["out_of_training_range_warnings"] == before["out_of_training_range_warnings"] + 1
    response = client.get("/metrics")
    assert response.status_code == 200 and "version=0.0.4" in response.content_type
    assert b"attrition_request_latency_ms_bucket" in response.data
    assert client.get("/metrics?format=json").get_json()["requests"]


def test_prediction_logging_failure_does_not_destroy_successful_response(client, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk unavailable")
    monkeypatch.setattr(prediction_logger, "log", fail)
    before = metrics.snapshot()["prediction_log_errors"]
    assert client.post("/predict", json=VALID).status_code == 200
    assert metrics.snapshot()["prediction_log_errors"] == before + 1


def test_logs_only_validated_features_and_model_version(client, monkeypatch, tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    writer = PredictionLogger(str(log_path))
    monkeypatch.setattr(prediction_logger, "log", writer.log)
    response = client.post("/predict", json={**VALID, "Gender": "Male", "Marital Status": "Single"})
    record = json.loads(log_path.read_text())
    assert set(record["inputs"]) == set(VALID)
    assert record["model_version"] == response.get_json()["model_version"]
    assert record["request_id"] == response.headers["X-Request-ID"]


def test_request_id_is_bounded_and_usable(client):
    assert client.get("/health", headers={"X-Request-ID": "client-123"}).headers["X-Request-ID"] == "client-123"
    assert len(client.get("/health", headers={"X-Request-ID": "x" * 1000}).headers["X-Request-ID"]) < 65


def test_oversized_payload_has_json_error(client):
    response = client.post("/predict", data='x' * 17000, content_type="application/json")
    assert response.status_code == 413 and response.is_json
