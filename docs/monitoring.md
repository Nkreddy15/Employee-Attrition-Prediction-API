# Monitoring and operations

## Signals

| Signal | Where | Meaning |
| --- | --- | --- |
| Request ID, method, route, HTTP status, latency | JSON stdout; `X-Request-ID` response header | Trace a request without logging its raw body |
| Request/status counters and latency histogram | `/metrics`; JSON snapshot at `/metrics?format=json` | Traffic, failure rate, latency |
| Predictions by class; validation and extrapolation counters | `/metrics` | Input problems and prediction behavior |
| Prediction-log write failures | `attrition_prediction_log_errors_total` | Disk/path failure; successful predictions remain available |
| Validated nine-field inputs, score, result, model and schema version | `logs/predictions.jsonl` | Distribution drift analysis |
| Loaded model version and training runtime | `/health` | Match the release to its artifacts |

Prediction logging can be disabled using `PREDICTION_LOG=off`, or moved using `PREDICTION_LOG=/path/file.jsonl`.
Unknown request fields are rejected; excluded legacy fields never enter log inputs. No raw request body is
written to request logs. Predicted probabilities are uncalibrated.

## Drift checks

```bash
python drift_report.py --json
python drift_report.py --log /path/predictions.jsonl --min-rows 30 --window 1000 --json
```

The latest 1,000 usable records matching both the current model and schema are selected; minimum 30.
Malformed and incompatible records are counted. A truncated final JSONL line is skipped and reported.
Use complete lines when exporting a log being actively written.

Numeric PSI uses saved observed bin masses, not a uniform distribution assumption. Categorical PSI
includes unseen values. The Yes-rate comparison uses the *same serving model's* predictions on reference
data; training label prevalence is reported separately. Alerts do not demonstrate reduced accuracy:
new labeled outcomes are required to measure model performance.

| Exit | Status | Interpretation |
| --- | --- | --- |
| 0 | checked | Enough data; no ALERT (WARN values can still exist) |
| 1 | alert | Any PSI >0.25 or absolute predicted-Yes-rate change >0.20 |
| 2 | insufficient_data | No usable verdict; missing log, small sample, or only old versions |

PSI >0.10 is WARN, >0.25 is ALERT. Prediction-rate differences >0.10 warn and >0.20 alert.
These are configurable-in-code demo heuristics, not statistically calibrated alarm thresholds.

## Triage

| Trigger | First action |
| --- | --- |
| Health unavailable or model version wrong | Inspect startup logs and the build; artifacts must match the schema |
| 5xx increase | Trace request IDs and check inference errors |
| Validation-error share rises | Compare client requests against `/schema` |
| Prediction-log failures >0 | Check permissions and disk; inference may still be healthy while drift data is lost |
| PSI or prediction-rate alert | Inspect sample/population changes and version filtering before considering retraining |
| Insufficient data | Collect more valid traffic; do not report an all-clear |

The counters exist, but no external alert destination or automatic scheduler is configured.

## Retrain and release

1. Preserve the previous commit so it can be restored. Verify new labels, source units and provenance.
2. Install `requirements-dev.txt` under Python 3.12. Run `python train_model.py`.
3. Review `models/model_info.json` and `models/data_quality.json`; compare grouped folds and baselines.
4. Run `pytest -q`. Commit the pipeline plus both metadata/reference files together.
5. Build/redeploy. Check the live `/health` model identifier and smoke-test `/form`, `/predict` and `/metrics`.
6. Treat old prediction logs as a separate model version. `drift_report.py` excludes them automatically.

## Privacy and operational limits

Validated inputs include compensation and age; treat logs as sensitive. Use synthetic profiles for a
public demo, do not log names/IDs, and disable prediction logging for traffic that should not be retained.
The default local log is append-only; operators must rotate or delete it before disk capacity is exhausted.
Gender and marital status from legacy requests are excluded from logs.

Metrics reset on process restart and are per worker. The Dockerfile intentionally uses one worker and
four threads. Multiple workers or replicas require an external aggregation design. Logs and metrics on
free Spaces are ephemeral; durable retention needs a separately configured destination or volume.
`/metrics` is public operational telemetry; it contains no individual feature values. No authentication,
rate limiting, external alert delivery or throughput guarantee is claimed. This release was not load-tested.
