# Employee Attrition Prediction API

A machine learning-powered REST API that predicts whether an employee is likely to leave an organization based on demographic, job, compensation, commute, performance, and work-related features.

This project is built with **Python, Flask, Scikit-Learn, Docker**, and is suitable for deployment on **Hugging Face Spaces** or any Docker-compatible hosting platform.

## Project Overview

Employee attrition is a major business problem because employee turnover increases hiring costs, disrupts productivity, and affects team stability. This project uses machine learning to analyze employee-related factors and return an attrition prediction through a REST API.

The system includes:

- Data preprocessing
- Missing value handling
- Categorical feature encoding
- Feature scaling
- Gradient Boosting classification
- Flask API endpoint for predictions
- Dockerized deployment setup

## Tech Stack

- Python
- Flask
- Pandas
- Scikit-Learn
- Joblib
- Docker

## Repository Structure

```text
Employee-Attrition-Prediction-API/
├── app.py
├── train_model.py
├── Dockerfile
├── requirements.txt
├── README.md
├── data/
│   └── employee_attrition_dataset.csv
├── models/
├── screenshots/
└── docs/
    └── architecture.md
```

## Dataset

The project uses an employee attrition dataset containing features such as:

- Age
- Gender
- Education Level
- Department
- Job Role
- Salary
- Income
- Years of Experience
- Performance Rating
- Working Hours
- Distance from Home
- Commute Method
- Marital Status
- Attrition Target

The target variable is:

```text
Attrition (Target)
```

## Machine Learning Workflow

1. Load employee dataset
2. Handle missing values
   - Categorical columns are filled using the mode
   - Numerical columns are filled using the median
3. Encode categorical variables using `LabelEncoder`
4. Separate features and target variable
5. Standardize features using `StandardScaler`
6. Split data into training and testing sets
7. Train a `GradientBoostingClassifier`
8. Save model artifacts using `Joblib`
9. Serve predictions through a Flask API

## API Endpoints

### Home

```http
GET /
```

Returns a basic API status message.

### Predict Employee Attrition

```http
POST /predict
```

Accepts employee details as JSON and returns an attrition prediction.

## Example Request

```json
{
  "Age": 32,
  "Gender": "Male",
  "Education Level": "Bachelor's",
  "Department": "IT",
  "Job Role": "Software Engineer",
  "Salary (USD)": 50000,
  "Income (Euro)": 350000,
  "Years of Experience": 3,
  "Performance Rating": 3,
  "Working Hours": 8,
  "Distance from Home": 10,
  "Commute Method": "Car",
  "Marital Status": "Single"
}
```

## Example Response

```json
{
  "attrition_prediction": "No",
  "confidence": 0.87
}
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Nkreddy15/Employee-Attrition-Prediction-API.git
cd Employee-Attrition-Prediction-API
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

### 4. Start the Flask API

```bash
python app.py
```

The API will run at:

```text
http://localhost:7860
```

## Test the API with curl

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 32,
    "Gender": "Male",
    "Education Level": "Bachelor's",
    "Department": "Sales",
    "Job Role": "Sales Executive",
    "Salary (USD)": 65000,
    "Income (Euro)": 60000,
    "Years of Experience": 7,
    "Performance Rating": 4,
    "Working Hours": 42,
    "Distance from Home": 10,
    "Commute Method": "Car",
    "Marital Status": "Single"
  }'
```

## Docker Usage

### Build the Docker image

```bash
docker build -t employee-attrition-api .
```

### Run the container

```bash
docker run -p 7860:7860 employee-attrition-api
```

## Deployment Notes

This project is compatible with Docker-based deployment platforms, including Hugging Face Spaces.

If deploying on Hugging Face Spaces, keep the Hugging Face metadata block in the Hugging Face `README.md`. The GitHub README does not require that metadata block.

## Future Improvements

- Add a simple frontend form for non-technical users
- Add model comparison with Logistic Regression, Random Forest, and XGBoost
- Add SHAP-based explainability
- Improve validation for incoming API requests
- Add unit tests
- Add screenshots of API testing and deployment

## Author

Nikhil Kumar Reddy Chalamalla
