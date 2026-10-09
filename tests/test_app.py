
import sys
import os

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from app import app


def test_home_endpoint():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200

    data = response.get_json()

    assert data["message"] == "Iris classification API is running"


def test_predict_endpoint():
    client = app.test_client()

    data = {
        "Sepal Length": 5.1,
        "Sepal Width": 3.5,
        "Petal Length": 1.4,
        "Petal Width": 0.2
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 200

    result = response.get_json()

    assert "prediction" in result


def test_missing_feature():
    client = app.test_client()

    data = {
        "Sepal Length": 5.1,
        "Sepal Width": 3.5,
        "Petal Length": 1.4
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 400

    result = response.get_json()

    assert "error" in result


def test_invalid_feature_type():
    client = app.test_client()

    data = {
        "Sepal Length": "invalid",
        "Sepal Width": 3.5,
        "Petal Length": 1.4,
        "Petal Width": 0.2
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 400

    result = response.get_json()

    assert "error" in result
