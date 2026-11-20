from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from flask import Flask, Response, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from monitoring import PredictionLogger, RequestMetrics, new_request_id
from schema import CATEGORY_OPTIONS, CATEGORICAL_FIELDS, EXCLUDED_FIELDS, FEATURE_COLUMNS, NUMERIC_FIELDS, NUMERIC_RULES, SCHEMA_VERSION, validate_input

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models/employee_attrition_model.pkl")
model_info = json.loads((BASE_DIR / "models/model_info.json").read_text())
reference_stats = json.loads((BASE_DIR / "models/reference_stats.json").read_text())
if (model_info["schema_version"] != SCHEMA_VERSION
        or reference_stats["schema_version"] != SCHEMA_VERSION
        or reference_stats["model_version"] != model_info["model_version"]
        or model_info["sklearn_version"] != sklearn.__version__
        or list(model.feature_names_in_) != FEATURE_COLUMNS):
    raise RuntimeError("Model, schema and drift reference must be regenerated together.")

app = Flask(__name__, template_folder="template")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("attrition_api")
metrics = RequestMetrics()
prediction_logger = PredictionLogger()


@app.before_request
def start_timer():
    g.start = time.perf_counter()
    candidate = request.headers.get("X-Request-ID", "")
    g.request_id = candidate if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", candidate) else new_request_id()


@app.after_request
def record_request(response):
    latency_ms = (time.perf_counter() - g.start) * 1000
    path = request.url_rule.rule if request.url_rule else "unmatched"
    metrics.record_request(request.method, path, response.status_code, latency_ms)
    response.headers["X-Request-ID"] = g.request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    logger.info(json.dumps({"event": "request", "request_id": g.request_id, "method": request.method,
                            "path": path, "status": response.status_code, "latency_ms": round(latency_ms, 2)}))
    return response


@app.errorhandler(HTTPException)
def handle_http_error(error):
    if error.code in {400, 413, 415}:
        metrics.record_validation_error()
    return jsonify({"error": error.description}), error.code


@app.route("/")
def home():
    return jsonify({"message": "Employee Attrition API: portfolio demo", "web_form": "/form",
                    "health": "/health", "metrics": "/metrics", "schema": "/schema", "model_version": model_info["model_version"]})


@app.route("/schema")
def schema():
    return jsonify({"schema_version": SCHEMA_VERSION, "required_fields": FEATURE_COLUMNS,
                    "numeric": NUMERIC_RULES, "categorical": CATEGORY_OPTIONS, "excluded_legacy_fields": EXCLUDED_FIELDS})


def make_prediction(data):
    clean, warnings = validate_input(data)
    warnings.extend(out_of_training_range_warnings(clean))
    values = model.predict_proba(pd.DataFrame([clean], columns=FEATURE_COLUMNS))[0]
    probabilities = dict(zip(model.classes_, values))
    yes, no = float(probabilities[1]), float(probabilities[0])
    result = {
        "attrition_prediction": "Yes" if yes > no else "No",
        "confidence": max(yes, no), "probability_yes": yes, "probability_no": no,
        "probabilities_calibrated": False, "risk_factors": [],
        "warnings": warnings, "model_version": model_info["model_version"], "schema_version": SCHEMA_VERSION,
        "usage_note": "Portfolio demo. Scores are uncalibrated and unsuitable for real employment decisions.",
    }
    return clean, result


def out_of_training_range_warnings(data):
    warnings = []
    for name, reference in reference_stats["numeric"].items():
        low, high = reference["typical_min"], reference["typical_max"]
        if not low <= data[name] <= high:
            warnings.append(f"{name}={data[name]:g} is outside the observed training range [{low:g}, {high:g}].")
    return warnings


def record_prediction(source, inputs, result):
    metrics.record_prediction(result["attrition_prediction"])
    metrics.record_out_of_range(sum("outside the observed training range" in message for message in result["warnings"]))
    try:
        prediction_logger.log(g.request_id, source, inputs, result, (time.perf_counter() - g.start) * 1000)
    except OSError:
        metrics.record_log_error()
        logger.error(json.dumps({"event": "prediction_log_failed", "request_id": g.request_id}))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    try:
        clean, result = make_prediction(data)
    except ValueError as error:
        metrics.record_validation_error()
        return jsonify({"error": f"Invalid input: {error}"}), 400
    except Exception:
        logger.exception(json.dumps({"event": "prediction_failed", "request_id": g.request_id}))
        return jsonify({"error": "Unexpected server error."}), 500
    record_prediction("api", clean, result)
    return jsonify(result)


@app.route("/form", methods=["GET", "POST"])
def form():
    result, error, status = {}, None, 200
    submitted = request.form.to_dict() if request.method == "POST" else {}
    if request.method == "POST":
        try:
            clean, result = make_prediction(submitted)
            record_prediction("form", clean, result)
        except ValueError as failure:
            metrics.record_validation_error()
            error, status = str(failure), 400
        except Exception:
            logger.exception(json.dumps({"event": "form_prediction_failed", "request_id": g.request_id}))
            error, status = "Unexpected server error.", 500
    return render_template("index.html", numeric_rules=NUMERIC_RULES, numeric_fields=NUMERIC_FIELDS,
                           categorical_fields=CATEGORICAL_FIELDS, dropdown_options=CATEGORY_OPTIONS,
                           prediction=result.get("attrition_prediction"), confidence=result.get("confidence"),
                           probability_yes=result.get("probability_yes"), probability_no=result.get("probability_no"),
                           warnings=result.get("warnings", []), error=error, submitted_values=submitted), status


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "model_version": model_info["model_version"],
                    "schema_version": SCHEMA_VERSION, "n_features": len(FEATURE_COLUMNS),
                    "model_trained_at": model_info["trained_at"], "sklearn_version_at_training": model_info["sklearn_version"]})


@app.route("/metrics")
def metrics_endpoint():
    if request.args.get("format") == "json":
        return jsonify(metrics.snapshot())
    return Response(metrics.render_prometheus(), content_type="text/plain; version=0.0.4; charset=utf-8")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
