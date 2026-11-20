"""Lightweight monitoring for the Employee Attrition API.

Three pieces, all standard-library only:
  * RequestMetrics   - in-process counters and a latency histogram, exposed at /metrics
  * PredictionLogger - append-only JSONL log of every prediction, used by drift_report.py
  * new_request_id   - short id returned in the X-Request-ID header for tracing

Limits: metrics live in process memory (they reset on restart and are per worker),
and the JSONL file is local disk. For production, ship both to a real backend
(Prometheus, CloudWatch, a database). See docs/monitoring.md.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LATENCY_BUCKETS_MS = (10, 25, 50, 100, 250, 500, 1000)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class RequestMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests: Dict[str, int] = {}           # "METHOD path status" -> count
        self.predictions = {"Yes": 0, "No": 0}
        self.validation_errors = 0
        self.out_of_range_warnings = 0
        self.log_errors = 0
        self.latency_buckets = {b: 0 for b in LATENCY_BUCKETS_MS}
        self.latency_count = 0
        self.latency_sum_ms = 0.0

    def record_request(self, method: str, path: str, status: int, latency_ms: float) -> None:
        key = f"{method} {path} {status}"
        with self._lock:
            self.requests[key] = self.requests.get(key, 0) + 1
            self.latency_count += 1
            self.latency_sum_ms += latency_ms
            for bound in LATENCY_BUCKETS_MS:
                if latency_ms <= bound:
                    self.latency_buckets[bound] += 1

    def record_prediction(self, label: str) -> None:
        with self._lock:
            self.predictions[label] = self.predictions.get(label, 0) + 1

    def record_out_of_range(self, n_features: int) -> None:
        with self._lock:
            self.out_of_range_warnings += n_features

    def record_validation_error(self) -> None:
        with self._lock:
            self.validation_errors += 1

    def record_log_error(self) -> None:
        with self._lock:
            self.log_errors += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total_predictions = sum(self.predictions.values())
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "requests": dict(self.requests),
                "predictions": dict(self.predictions),
                "positive_prediction_rate": (
                    round(self.predictions.get("Yes", 0) / total_predictions, 4) if total_predictions else None
                ),
                "validation_errors": self.validation_errors,
                "prediction_log_errors": self.log_errors,
                "out_of_training_range_warnings": self.out_of_range_warnings,
                "latency_ms_avg": round(self.latency_sum_ms / self.latency_count, 2) if self.latency_count else None,
            }

    def render_prometheus(self) -> str:
        """Prometheus text exposition format (counters and a cumulative histogram)."""
        with self._lock:
            lines = [
                "# HELP attrition_requests_total HTTP requests by method, path and status.",
                "# TYPE attrition_requests_total counter",
            ]
            for key, count in sorted(self.requests.items()):
                method, path, status = key.split(" ")
                lines.append(f'attrition_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}')
            lines += [
                "# HELP attrition_predictions_total Predictions by predicted class.",
                "# TYPE attrition_predictions_total counter",
            ]
            for label, count in sorted(self.predictions.items()):
                lines.append(f'attrition_predictions_total{{label="{label}"}} {count}')
            lines += [
                "# HELP attrition_validation_errors_total Requests rejected by input validation.",
                "# TYPE attrition_validation_errors_total counter",
                f"attrition_validation_errors_total {self.validation_errors}",
                "# HELP attrition_prediction_log_errors_total Predictions whose log could not be written.",
                "# TYPE attrition_prediction_log_errors_total counter",
                f"attrition_prediction_log_errors_total {self.log_errors}",
                "# HELP attrition_out_of_range_features_total Inputs outside the typical training range.",
                "# TYPE attrition_out_of_range_features_total counter",
                f"attrition_out_of_range_features_total {self.out_of_range_warnings}",
                "# HELP attrition_request_latency_ms Request latency in milliseconds.",
                "# TYPE attrition_request_latency_ms histogram",
            ]
            for bound in LATENCY_BUCKETS_MS:
                lines.append(f'attrition_request_latency_ms_bucket{{le="{bound}"}} {self.latency_buckets[bound]}')
            lines += [
                f'attrition_request_latency_ms_bucket{{le="+Inf"}} {self.latency_count}',
                f"attrition_request_latency_ms_sum {round(self.latency_sum_ms, 3)}",
                f"attrition_request_latency_ms_count {self.latency_count}",
            ]
        return "\n".join(lines) + "\n"


class PredictionLogger:
    """Appends one JSON line per prediction. Disable with PREDICTION_LOG=off."""

    def __init__(self, path: Optional[str] = None) -> None:
        raw = path if path is not None else os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl")
        self.enabled = raw.lower() != "off"
        self.path = Path(raw) if self.enabled else None
        self._lock = threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, request_id: str, source: str, inputs: Dict[str, Any], result: Dict[str, Any], latency_ms: float) -> None:
        if not self.enabled:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request_id": request_id,
            "model_version": result.get("model_version"),
            "schema_version": result.get("schema_version"),
            "source": source,  # "api" or "form"
            "inputs": inputs,
            "prediction": result.get("attrition_prediction"),
            "probability_yes": result.get("probability_yes"),
            "warnings": result.get("warnings", []),
            "latency_ms": round(latency_ms, 2),
        }
        line = json.dumps(record, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
