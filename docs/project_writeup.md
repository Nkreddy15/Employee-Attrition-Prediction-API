# Turning a brittle attrition demo into an observable service

## Problem

The original project reported a perfect score on a 20-row test split while its API accepted income and
hours in ranges that disagreed with the training data. The upgrade ZIP added useful monitoring but
retained that mismatch, whole-dataset preprocessing, row-shuffled evaluation of repeated profiles,
and handwritten explanations. Flask also searched the wrong template directory.

## Changes

- Added a shared input contract and a single serialized preprocessing/model pipeline.
- Kept the source CSV unchanged; audited deterministic invalid-value handling and missingness.
- Excluded unverified income/hours fields instead of guessing unit conversions. Excluded gender and marital status.
- Replaced ordinary shuffled evaluation with fold-local preprocessing and grouped profiles; added majority
  and logistic-regression comparisons. Published complete fold results and limitations.
- Added `/health`, `/schema`, Prometheus metrics, bounded request IDs, structured request logs and
  model-versioned sanitized prediction logs. Log-write failures have a counter and preserve inference availability.
- Fixed the HTML form and migration path for legacy requests. Removed non-model demographic risk text.
- Corrected numeric PSI for tied values and excluded old-model logs. Insufficient traffic receives an explicit
  inconclusive status, not a clean verdict.
- Added contract/evaluation/drift regressions and CI that tests saved and regenerated artifacts, then builds
  and smoke-tests the non-root Gunicorn container.

## Findings

There are only 41 distinct retained observed profiles among 100 rows, including three conflicting-label
groups. Grouped gradient-boosting accuracy is 0.827 ± 0.087 fold SD; F1 is 0.768. Logistic regression
scores 0.897 ± 0.083 accuracy on the same internal folds. This supports reporting limitations and keeping
a simpler model as a serious candidate; it does not establish generalization or an improvement attributable
to monitoring. Similar incomplete profiles may still cross folds.

## Resume wording supported by the implementation

“Instrumented a Flask/scikit-learn attrition API with health checks, Prometheus metrics, request tracing,
and version-aware drift checks; added regression tests, CI, and a model card documenting dataset and evaluation limits.”

“Reworked preprocessing and evaluation to keep repeated profiles within folds, audited 100 source rows,
and removed unverified income/hours inputs; reported grouped validation results alongside simple baselines.”

A live-deployment claim should be added only after the deployed `/health` version and successful requests
have been verified. No production impact, accuracy gain, fairness certification or SHAP explanation is claimed.
