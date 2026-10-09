# Iris Classification Model Deployment

This project uses machine learning to classify Iris flowers from four measurements: sepal length, sepal width, petal length and petal width. I trained a classification model and built a Flask API to make predictions from input data.

## Project Overview

The project covers the main steps from model training to API testing:

- Explored the Iris dataset.
- Prepared the data and trained a classification model.
- Saved the trained model for reuse.
- Built a Flask API to return predictions.
- Added tests for the API endpoints.
- Set up GitHub Actions to run the tests automatically.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Flask
- Joblib
- Pytest
- GitHub Actions

## Project Files

| File | Description |
|---|---|
| `iris_classification_model_deployment.ipynb` | Notebook containing the analysis and model workflow |
| `2. Iris Dataset.xlsx` | Dataset used for the project |
| `iris_model.joblib` | Saved trained classification model |
| `app.py` | Flask application and prediction endpoint |
| `requirements.txt` | Python dependencies |
| `tests/test_app.py` | Tests for the API endpoints |
| `.github/workflows/tests.yml` | GitHub Actions workflow for automated testing |

## Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/aaryanshukla777/model-deployment-flask-api.git
   cd model-deployment-flask-api
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   pip install pytest
   ```

3. Start the Flask application:

   ```bash
   python app.py
   ```

4. Run the tests in a separate terminal:

   ```bash
   pytest tests/
   ```

## API Usage

The API accepts the four Iris flower measurements and returns a predicted class.

Send a POST request to `/predict` with JSON in this format:

```json
{
  "Sepal Length": 5.1,
  "Sepal Width": 3.5,
  "Petal Length": 1.4,
  "Petal Width": 0.2
}
```

## Automated Testing

GitHub Actions runs the test suite when changes are pushed to the `main` branch or a pull request targets `main`.

The workflow completed successfully in the current repository.

## What I Learned

This project helped me practise training and saving a machine learning model, serving predictions through a Flask API, testing endpoints with Pytest, and automating tests with GitHub Actions.
