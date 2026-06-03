from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "employee_attrition_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
ENCODERS_PATH = BASE_DIR / "models" / "label_encoders.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

app = Flask(__name__)

NUMERIC_RULES = {
    "Age": {"min": 18, "max": 65, "step": 1, "default": 32},
    "Salary (USD)": {"min": 20000, "max": 250000, "step": 1000, "default": 65000},
    "Income (Euro)": {"min": 18000, "max": 230000, "step": 1000, "default": 60000},
    "Years of Experience": {"min": 0, "max": 45, "step": 1, "default": 7},
    "Performance Rating": {"min": 1, "max": 5, "step": 1, "default": 4},
    "Working Hours": {"min": 20, "max": 80, "step": 1, "default": 42},
    "Distance from Home": {"min": 0, "max": 100, "step": 1, "default": 10}
}

NUMERIC_FIELDS = list(NUMERIC_RULES.keys())
CATEGORICAL_FIELDS = [col for col in feature_columns if col in label_encoders]


@app.route("/")
def home():
    return jsonify({
        "message": "Employee Attrition Prediction API is running.",
        "web_form": "/form",
        "api_usage": "Send POST request to /predict with employee details as JSON."
    })


@app.route("/form", methods=["GET", "POST"])
def form():
    prediction = None
    confidence = None
    probability_yes = None
    probability_no = None
    error = None
    risk_factors = []
    submitted_values = {}

    dropdown_options = {
        col: list(label_encoders[col].classes_)
        for col in CATEGORICAL_FIELDS
    }

    if request.method == "POST":
        try:
            input_data = parse_form_input(request.form)
            submitted_values = input_data.copy()
            result = make_prediction(input_data)

            prediction = result["attrition_prediction"]
            confidence = result["confidence"]
            probability_yes = result["probability_yes"]
            probability_no = result["probability_no"]
            risk_factors = result["risk_factors"]

        except Exception as e:
            error = str(e)
            submitted_values = request.form.to_dict()

    return render_template(
        "index.html",
        numeric_rules=NUMERIC_RULES,
        numeric_fields=NUMERIC_FIELDS,
        categorical_fields=CATEGORICAL_FIELDS,
        dropdown_options=dropdown_options,
        prediction=prediction,
        confidence=confidence,
        probability_yes=probability_yes,
        probability_no=probability_no,
        risk_factors=risk_factors,
        error=error,
        submitted_values=submitted_values
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        result = make_prediction(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def parse_form_input(form_data):
    input_data = {}

    for field, rules in NUMERIC_RULES.items():
        raw_value = form_data.get(field)

        if raw_value is None or raw_value == "":
            raise ValueError(f"{field} is required.")

        try:
            value = float(raw_value)
        except ValueError:
            raise ValueError(f"{field} must be a number.")

        if value < rules["min"] or value > rules["max"]:
            raise ValueError(f"{field} must be between {rules['min']} and {rules['max']}.")

        input_data[field] = value

    for field in CATEGORICAL_FIELDS:
        value = form_data.get(field)
        if not value:
            raise ValueError(f"{field} is required.")
        input_data[field] = value

    return input_data


def make_prediction(data):
    df = pd.DataFrame([data])

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required fields: {missing_cols}")

    df = df[feature_columns].copy()

    for field, rules in NUMERIC_RULES.items():
        value = float(df.loc[0, field])
        if value < rules["min"] or value > rules["max"]:
            raise ValueError(f"{field} must be between {rules['min']} and {rules['max']}.")
        df.loc[0, field] = value

    risk_factors = calculate_risk_factors(data)

    for col in CATEGORICAL_FIELDS:
        value = df.loc[0, col]
        accepted_values = list(label_encoders[col].classes_)

        if value not in accepted_values:
            raise ValueError(
                f"Unknown category '{value}' for column '{col}'. Accepted values: {accepted_values}"
            )

        df[col] = label_encoders[col].transform(df[col])

    df_scaled = scaler.transform(df)
    prediction = int(model.predict(df_scaled)[0])

    confidence = None
    probability_no = None
    probability_yes = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df_scaled)[0]
        probability_no = float(probabilities[0])
        probability_yes = float(probabilities[1])
        confidence = float(max(probabilities))

    return {
        "attrition_prediction": "Yes" if prediction == 1 else "No",
        "confidence": confidence,
        "probability_no": probability_no,
        "probability_yes": probability_yes,
        "risk_factors": risk_factors
    }


def calculate_risk_factors(data):
    factors = []

    age = float(data.get("Age", 0))
    salary = float(data.get("Salary (USD)", 0))
    income = float(data.get("Income (Euro)", 0))
    experience = float(data.get("Years of Experience", 0))
    performance = float(data.get("Performance Rating", 0))
    working_hours = float(data.get("Working Hours", 0))
    distance = float(data.get("Distance from Home", 0))
    marital_status = str(data.get("Marital Status", ""))

    if salary <= 40000:
        factors.append("Low salary may increase attrition risk.")
    if income <= 35000:
        factors.append("Low income level may contribute to dissatisfaction.")
    if working_hours >= 60:
        factors.append("High working hours may indicate burnout risk.")
    if distance >= 50:
        factors.append("Long commute distance may increase leaving risk.")
    if performance <= 2:
        factors.append("Low performance rating may indicate role mismatch or disengagement.")
    if experience <= 1:
        factors.append("Low experience may indicate early-career mobility.")
    if age <= 25:
        factors.append("Younger employees may be more likely to switch roles.")
    if marital_status.lower() == "single":
        factors.append("Single employees may have higher flexibility to relocate or change jobs.")

    if not factors:
        factors.append("No obvious rule-based risk factors detected from the entered profile.")

    return factors[:5]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
