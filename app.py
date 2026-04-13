
# Flask backend for Student Performance Predictor


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__, static_folder=".")
CORS(app)

# Load saved model
saved = joblib.load("model.pkl")
rf             = saved["model"]
FEATURE_MEDIANS = saved["feature_medians"]
FEATURE_COLS   = saved["feature_cols"]

print("Model loaded from model.pkl")


# Routes 
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Build a feature row using medians values for the other less significant features
    row = FEATURE_MEDIANS.copy()

    # Override with user-supplied values (names match dataset columns)
    mapping = {
        "attendance":      "Attendance",
        "hours_studied":   "Hours_Studied",
        "previous_scores": "Previous_Scores",
        "sleep_hours":     "Sleep_Hours",
    }
    for key, col in mapping.items():
        if key in data and data[key] is not None:
            row[col] = float(data[key])

    # Predict
    X_input = pd.DataFrame([row])[FEATURE_COLS]
    prediction = rf.predict(X_input)[0]
    probability = rf.predict_proba(X_input)[0][1]  # probability of At-Risk

    return jsonify({
        "prediction": "At-Risk" if prediction == 1 else "Not At-Risk",
        "probability": round(float(probability) * 100, 1),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
