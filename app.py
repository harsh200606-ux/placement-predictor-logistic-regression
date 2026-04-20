from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "placement_model.pkl")

FEATURE_ORDER = [
    "CGPA",
    "AptitudeScore",
    "CommunicationScore",
    "Internship",
    "Projects",
    "Backlogs",
    "Certifications",
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        row = {
            "CGPA": float(data["cgpa"]),
            "AptitudeScore": int(data["aptitude"]),
            "CommunicationScore": int(data["communication"]),
            "Internship": int(data["internship"]),
            "Projects": int(data["projects"]),
            "Backlogs": int(data["backlogs"]),
            "Certifications": int(data["certifications"]),
        }

        features = pd.DataFrame([row], columns=FEATURE_ORDER)
        probability = float(model.predict_proba(features)[0][1])
        prediction = int(model.predict(features)[0])

        return jsonify({
            "prediction": "Placed" if prediction == 1 else "Not Placed",
            "probability": round(probability * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
