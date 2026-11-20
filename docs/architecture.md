# Architecture

`train_model.py` reads the unchanged CSV, applies deterministic checks, and evaluates a pipeline using
profile-grouped folds. Numeric median imputation/scaling and categorical mode imputation/one-hot encoding
fit inside each fold. A final pipeline fit on all rows is saved with evaluation metadata and its drift reference.

`app.py` validates nine inputs against `schema.py`, loads the pipeline once, and serves JSON predictions
or the HTML form. A single Gunicorn process uses four threads. Request metrics use a lock; prediction
logging writes validated features and model version to JSONL. `/health` identifies the loaded model;
`/metrics` exposes counters and latency buckets. The separate drift CLI compares matching-version logs
with empirical reference distributions. There is no automatic retraining or external alert service.
