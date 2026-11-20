"""Shared serving and training contract. No guessed currency or time conversions."""
from __future__ import annotations

import math

SCHEMA_VERSION = "2"
NUMERIC_RULES = {
    "Age": {"min": 18, "max": 100, "step": 1, "default": 32, "unit": "years"},
    "Salary (USD)": {"min": 20000, "max": 250000, "step": 1000, "default": 65000, "unit": "USD"},
    "Years of Experience": {"min": 0, "max": 65, "step": 1, "default": 7, "unit": "years"},
    "Performance Rating": {"min": 1, "max": 5, "step": 1, "default": 4, "unit": "rating"},
    "Distance from Home": {"min": 0, "max": 100, "step": 1, "default": 10, "unit": "source unit unspecified"},
}
CATEGORY_OPTIONS = {
    "Education Level": ["Bachelor's", "Diploma", "Master's", "PhD"],
    "Department": ["Finance", "IT", "Marketing", "Operations", "Research", "Sales"],
    "Job Role": ["Financial Analyst", "Marketing Analyst", "Marketing Executive", "Marketing Manager",
                 "Operations Manager", "Research Manager", "Research Scientist", "Sales Representative",
                 "Software Engineer"],
    "Commute Method": ["Car", "Public Transit", "Walk"],
}
NUMERIC_FIELDS = list(NUMERIC_RULES)
CATEGORICAL_FIELDS = list(CATEGORY_OPTIONS)
FEATURE_COLUMNS = NUMERIC_FIELDS + CATEGORICAL_FIELDS
EXCLUDED_FIELDS = {
    "Income (Euro)": "Excluded: currency units are unverified and values largely duplicate Salary (USD) at 7x.",
    "Working Hours": "Excluded: the source does not verify daily versus weekly units and contains invalid outliers.",
    "Gender": "Excluded from model inputs.",
    "Marital Status": "Excluded from model inputs.",
}


def validate_input(data: dict) -> tuple[dict, list[str]]:
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")
    missing = [name for name in FEATURE_COLUMNS if name not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    unknown = sorted(set(data) - set(FEATURE_COLUMNS) - set(EXCLUDED_FIELDS))
    if unknown:
        raise ValueError(f"Unknown fields: {unknown}")
    clean = {}
    for name, rule in NUMERIC_RULES.items():
        raw = data[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise ValueError(f"{name} must be a finite number.")
        try:
            value = float(raw)
        except (ValueError, OverflowError):
            raise ValueError(f"{name} must be a finite number.") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if not rule["min"] <= value <= rule["max"]:
            raise ValueError(f"{name} must be between {rule['min']} and {rule['max']}.")
        if name in {"Age", "Performance Rating"} and not value.is_integer():
            raise ValueError(f"{name} must be a whole number.")
        clean[name] = value
    if clean["Years of Experience"] > clean["Age"] - 18:
        raise ValueError("Years of Experience cannot exceed Age minus 18 under this demo's adult-work contract.")
    for name, options in CATEGORY_OPTIONS.items():
        value = data[name]
        if not isinstance(value, str) or value not in options:
            raise ValueError(f"Unknown category for {name}. Accepted values: {options}")
        clean[name] = value
    warnings = [f"{name}: {reason}" for name, reason in EXCLUDED_FIELDS.items() if name in data]
    return clean, warnings
