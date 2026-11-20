"""Version-filtered PSI report. Exit 0=checked, 1=alert, 2=insufficient/unusable data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
EPS = 1e-4


def psi(expected, actual) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), EPS, None)
    actual = np.clip(np.asarray(actual, dtype=float), EPS, None)
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def status(value: float) -> str:
    return "ALERT" if value > 0.25 else "WARN" if value > 0.10 else "ok"


def numeric_psi(reference: dict, values: list[float]) -> float:
    edges = reference["bin_edges"]
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return psi(reference["bin_probabilities"], counts / len(values))


def categorical_psi(reference: dict, values: list[str]) -> float:
    categories = sorted(set(reference) | set(values))
    counts = np.array([values.count(category) for category in categories], dtype=float)
    return psi([reference.get(category, 0) for category in categories], counts / len(values))


def build_report(rows: list[dict], reference: dict) -> dict:
    report = {"status": "checked", "model_version": reference["model_version"], "n_logged": len(rows), "features": []}
    for name, ref in reference["numeric"].items():
        values = [float(row["inputs"][name]) for row in rows]
        score = numeric_psi(ref, values)
        report["features"].append({"feature": name, "type": "numeric", "psi": round(score, 6),
                                   "status": status(score), "train_mean": ref["mean"], "live_mean": float(np.mean(values))})
    for name, ref in reference["categorical"].items():
        score = categorical_psi(ref, [row["inputs"][name] for row in rows])
        report["features"].append({"feature": name, "type": "categorical", "psi": round(score, 6), "status": status(score)})
    rate = float(np.mean([row["prediction"] == "Yes" for row in rows]))
    baseline = reference["reference_prediction_rate"]
    difference = abs(rate - baseline)
    report["prediction_shift"] = {"reference_model_yes_rate": baseline, "training_label_yes_rate": reference["positive_rate"],
                                   "live_yes_rate": rate, "abs_difference": difference,
                                   "status": "ALERT" if difference > 0.20 else "WARN" if difference > 0.10 else "ok"}
    if any(feature["status"] == "ALERT" for feature in report["features"]) or report["prediction_shift"]["status"] == "ALERT":
        report["status"] = "alert"
    return report


def load_log(path: Path) -> tuple[list[dict], int]:
    rows, malformed = [], 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("not an object")
                rows.append(value)
            except (ValueError, TypeError):
                malformed += 1
    return rows, malformed


def compatible(row, reference) -> bool:
    if row.get("model_version") != reference["model_version"] or row.get("schema_version") != reference["schema_version"]:
        return False
    if not isinstance(row.get("inputs"), dict) or row.get("prediction") not in {"Yes", "No"}:
        return False
    for name in reference["numeric"]:
        try:
            if not np.isfinite(float(row["inputs"][name])):
                return False
        except (TypeError, ValueError, KeyError, OverflowError):
            return False
    return all(isinstance(row["inputs"].get(name), str) for name in reference["categorical"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=BASE_DIR / "logs/predictions.jsonl")
    parser.add_argument("--reference", type=Path, default=BASE_DIR / "models/reference_stats.json")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--window", type=int, default=1000, help="most recent matching-version rows")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.min_rows < 1 or args.window < args.min_rows:
        parser.error("window must be >= min-rows >= 1")
    reference = json.loads(args.reference.read_text())
    rows, malformed = load_log(args.log) if args.log.exists() else ([], 0)
    matching = [row for row in rows if compatible(row, reference)]
    selected = matching[-args.window:]
    metadata = {"excluded_rows": len(rows) - len(matching), "malformed_rows": malformed,
                "window": args.window, "minimum_rows": args.min_rows}
    if len(selected) < args.min_rows:
        report = {"status": "insufficient_data", "n_logged": len(selected), "model_version": reference["model_version"], **metadata}
        code = 2
    else:
        report = {**build_report(selected, reference), **metadata}
        code = 1 if report["status"] == "alert" else 0
    if args.json:
        print(json.dumps(report, indent=2))
    elif report["status"] == "insufficient_data":
        print(f"Insufficient matching-version data: {report['n_logged']} rows; need {args.min_rows}. No drift verdict.")
    else:
        print(f"Drift report: {report['n_logged']} predictions; model {reference['model_version']}")
        for feature in sorted(report["features"], key=lambda entry: -entry["psi"]):
            print(f"{feature['feature']:<24} PSI={feature['psi']:.3f}  {feature['status']}")
        print("Prediction-rate shift:", report["prediction_shift"])
    return code


if __name__ == "__main__":
    sys.exit(main())
