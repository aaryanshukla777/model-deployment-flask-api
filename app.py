
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "iris_model.joblib"
)

model = joblib.load(model_path)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris classification API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body must contain JSON data."
            }), 400

        required_features = [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ]

        for feature in required_features:
            if feature not in data:
                return jsonify({
                    "error": f"Missing feature: {feature}"
                }), 400

        try:
            features = np.array([[
                float(data["Sepal Length"]),
                float(data["Sepal Width"]),
                float(data["Petal Length"]),
                float(data["Petal Width"])
            ]])
        except (ValueError, TypeError):
            return jsonify({
                "error": "All feature values must be numeric."
            }), 400

        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": str(prediction)
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Model prediction failed.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
