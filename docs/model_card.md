# Model card: employee attrition demo, schema v2

## Scope and evidence

Gradient boosting with a scikit-learn preprocessing pipeline; nine inputs; binary Yes/No output.
The source CSV has 100 rows (60 No, 40 Yes), unknown provenance and repeated template-like profiles.
After deterministic checks and feature exclusions, there are 41 distinct observed feature groups,
59 repeated-profile rows and three profile groups with conflicting labels.

This is an engineering and learning demonstration. It is not suitable for hiring, termination,
promotion, compensation, or other real employment decisions. The scores are uncalibrated and no
independent generalization or subgroup fairness claim is supported.

## Cleaning and feature contract

The raw CSV is unchanged and hashed in `models/data_quality.json`. Thirty-two rows have missing
source values. Invalid values are marked missing, not clamped to a boundary. Numeric medians and
categorical modes are learned exclusively on each fold's training rows during evaluation.

- `Income (Euro)` is excluded: a largely 7x-salary relationship does not establish a valid exchange rate.
- `Working Hours` is excluded: neither daily nor weekly interpretation is documented; extreme outliers exist.
- Gender and marital status are excluded. Other attributes may still proxy demographic information.
- Age outside 18–100, rating outside 1–5, experience above age−18, and unknown categories become missing
  during training. Serving rejects invalid required inputs. The age/experience rule is a demo assumption.
- `Talk` is an unknown commute category; it is not silently rewritten as `Walk`.
- No missing target labels are imputed; training stops on missing or unknown labels.
- Salary is in USD with unspecified period; distance is in an unspecified source unit.

## Evaluation

| Model | Accuracy (mean ± fold SD) | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Gradient Boosting | 0.827 ± 0.087 | 0.843 | 0.775 | 0.768 | 0.934 |
| Logistic Regression | 0.897 ± 0.083 | 0.916 | 0.850 | 0.850 | 0.930 |
| Majority Baseline | 0.599 ± 0.018 | 0.000 | 0.000 | 0.000 | 0.500 |

Five-fold stratified group cross-validation with seed 42 groups identical *observed, cleaned retained*
features before fitting any imputation or encoding. Each row is evaluated once and no group crosses
training and validation within a fold. All preprocessing fits on training folds. Final serving artifacts
are fit on all 100 rows after evaluation.

Gradient boosting is retained to avoid conflating the monitoring/data correction with a claim of improved
classifier selection. Logistic regression scores better here; future model selection needs additional
independent data. The former 100% 20-row holdout and ~91% row-shuffled CV are superseded.
Fold SD is not a confidence interval. Incomplete or near-duplicate profiles can remain similar across
folds, and the dataset cannot support strong performance conclusions.

## Output and monitoring

No individual feature attribution is provided. Handwritten demographic risk statements were removed;
`risk_factors: []` remains only for response compatibility. The UI labels scores as uncalibrated.
Input warnings flag numeric values outside observed training ranges. Four excluded legacy fields are
accepted and ignored with explicit warnings; they never enter the pipeline or prediction log.

PSI monitors input distribution shifts, not causal effects, fairness or prediction correctness. Its
reference includes empirical bin masses (important for discrete values), model and schema versions,
and the serving model's reference prediction rate.

## Limits and next work

Collect a documented larger dataset, verify all units and the labeling procedure, identify near-duplicate
families, obtain an independent test set, evaluate calibration and subgroup errors, and compare simpler
models before making deployment-quality claims. Removing explicit demographics alone does not prove fairness.
See `models/model_info.json` for the exact serving version, full fold metrics, dependencies and audit.
