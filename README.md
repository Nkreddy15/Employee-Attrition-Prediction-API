---
title: Employee Attrition API
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Employee Attrition Prediction API

A Flask and scikit-learn **portfolio demo** with a web form, JSON predictions, health checks,
Prometheus metrics, structured logs, and version-aware data-drift checks.

The service uses nine input fields. The source has **100 rows but only 41 distinct
observed profiles after feature exclusions and deterministic checks**. It is too small, repetitive and
poorly documented to support real employment decisions. Model probabilities are uncalibrated scores.

## Evaluation

All imputers, encoders and scaling fit inside each training fold. A five-fold `StratifiedGroupKFold`
keeps identical observed feature profiles together. The final serving pipeline is then fit on all 100 rows.

| Model | Accuracy (mean ± fold SD) | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Gradient Boosting | 0.827 ± 0.087 | 0.843 | 0.775 | 0.768 | 0.934 |
| Logistic Regression | 0.897 ± 0.083 | 0.916 | 0.850 | 0.850 | 0.930 |
| Majority Baseline | 0.599 ± 0.018 | 0.000 | 0.000 | 0.000 | 0.500 |

Gradient boosting remains the serving classifier to preserve the original model family while correcting
the data and evaluation. Logistic regression performs better on these folds; complexity is not an advantage
here. These are internal comparisons, with no independent test set. Exact-profile grouping cannot remove
all similarity between incomplete or near-duplicate rows. Fold SD is **not** a confidence interval.

The former 100% holdout and 91% ordinary-CV claims are superseded. Full fold scores, data audit, dataset
hash, runtime versions and model identifier are saved in [`models/model_info.json`](models/model_info.json).
See the [model card](docs/model_card.md) and [engineering write-up](docs/project_writeup.md).

## Data cleanup and API migration

The original CSV is retained unchanged. Training applies deterministic validation before evaluation,
then learns numeric median and categorical mode imputation within each fold.

| Change | Reason |
| --- | --- |
| Exclude `Income (Euro)` | Units are unverified and values largely duplicate salary at 7x. No invented exchange rate. |
| Exclude `Working Hours` | Daily/weekly units are unverified; raw values include -545 and 10,000. No guessed conversion. |
| Exclude `Gender` and `Marital Status` | Remove these explicit demographic inputs; this does not establish fairness. |
| Invalid ratings, under-contract ages, impossible experience and `Talk` commute become missing | Fit replacements within training folds. Do not invent corrections. |
| Remove handwritten risk explanations | They did not explain model output. `risk_factors` is retained as an empty compatibility field. |
| One serialized preprocessing + model pipeline | Training and serving use the same transformations. |

`Age` is years, salary is USD, experience is years, and rating is 1–5. The source does not specify salary
period or distance units; the interface does not invent them. The adult-work contract (age ≥18, experience
≤age−18) is an explicit demo assumption, not a claim about all employment histories.

Old 13-field requests still work when the nine required fields are valid: the four excluded fields are
ignored, omitted from prediction logs, and identified in response warnings. Unknown extra fields are rejected.
Get the exact input contract from `GET /schema`.

## Run and test

```bash
git clone https://github.com/Nkreddy15/Employee-Attrition-Prediction-API.git
cd Employee-Attrition-Prediction-API
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                         # validate committed artifacts
python train_model.py             # regenerate model and reference together
pytest -q
# Production-style local server:
gunicorn --bind 0.0.0.0:7860 --workers 1 --threads 4 --timeout 30 app:app
```

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Service links and serving model identifier |
| `GET /form` / `POST /form` | Web form using the same validation as the API |
| `POST /predict` | JSON prediction; 400 for invalid input, 413 above 16 KiB |
| `GET /schema` | Required fields, ranges, categories and excluded legacy inputs |
| `GET /health` | Readiness after artifacts have loaded, model version and training runtime |
| `GET /metrics` | Prometheus exposition; `?format=json` gives a readable snapshot |

## Example request

The checked request is in [`examples/request.json`](examples/request.json):

```json
{
  "Age": 32,
  "Salary (USD)": 65000,
  "Years of Experience": 7,
  "Performance Rating": 4,
  "Distance from Home": 10,
  "Education Level": "Bachelor's",
  "Department": "IT",
  "Job Role": "Software Engineer",
  "Commute Method": "Car"
}
```

```bash
curl --fail http://localhost:7860/health
curl --fail -H 'Content-Type: application/json'   --data-binary @examples/request.json http://localhost:7860/predict
curl --fail http://localhost:7860/metrics
```

A response includes `attrition_prediction`, `probability_yes`, `probability_no`, an uncalibrated
`confidence` score, `probabilities_calibrated: false`, `warnings`, `model_version` and `schema_version`.
A valid prediction does not fail merely because writing its monitoring log fails; the failure has its own counter.

## Monitoring

Every response includes a bounded `X-Request-ID`. Structured request logs contain status, route,
latency and request ID. Successful predictions append sanitized model inputs and model version to
`logs/predictions.jsonl`. Set `PREDICTION_LOG=off` to disable prediction logging.

```bash
python drift_report.py --json
```

Drift compares actual training bin shares with the latest 1,000 matching-version requests. Fewer than
30 rows returns `insufficient_data`, not a clean health verdict. Exit codes: **0** checked without alert,
**1** drift alert, **2** insufficient matching data. It does not monitor accuracy without new ground-truth labels.
The [operations runbook](docs/monitoring.md) covers interpretation, privacy and retraining.

## Docker and Hugging Face

```bash
docker build -t employee-attrition-api .
docker run --rm -p 7860:7860 employee-attrition-api
```

The Docker image runs Gunicorn as UID 1000 with one worker and four threads; in-process metrics are
consistent within that worker and reset on restart. It exposes port 7860 and includes a health check.
GitHub CI tests the committed artifacts, retrains, tests again, builds Docker and smoke-tests the container.

The README metadata supports Docker Spaces. To sync this exact checkout to the existing Space:

```bash
pip install -r requirements-deploy.txt
hf auth login
python deploy_space.py --repo-id Nkreddy15/employee-attrition-api
```

The deployment helper checks signed-in write access to the existing Space, uploads only versioned
project files, and preserves unrelated Space files. It never creates a Space, changes its visibility or upgrades hardware.
Successful upload triggers the Space build. Confirm `/health` shows the model identifier from
`models/model_info.json`, then test `/form`, `/predict` and `/metrics` on the deployed host.

## Layout

```text
app.py                  Flask service
schema.py               Shared input contract
train_model.py          Fold-local preprocessing and grouped evaluation
monitoring.py           Counters, histogram and JSONL predictions
drift_report.py         Version-filtered PSI CLI
deploy_space.py         Explicit sync to an existing Docker Space
data/                   Original source CSV (unchanged)
models/                 Serving pipeline, evaluation and reference profile
template/               Web form
tests/                  Serving, evaluation and drift regressions
docs/                   Model card, operations and project write-up
```

Author: Nikhil Kumar Reddy Chalamalla.
