"""
ANA680, Assignment 5, Problem 1
Author: Jesus Adrian Alvarado Gadea

Flask app that takes the 11 wine measurements from index.html, runs them
through the saved Random Forest model, and shows the predicted quality score.
"""

import math
import os

import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model once, when the app starts
model = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))

# Form field names, labels, and units, in the order the model was trained on
FEATURES = [
    ("fixed_acidity", "Fixed acidity", "g/dm³"),
    ("volatile_acidity", "Volatile acidity", "g/dm³"),
    ("citric_acid", "Citric acid", "g/dm³"),
    ("residual_sugar", "Residual sugar", "g/dm³"),
    ("chlorides", "Chlorides", "g/dm³"),
    ("free_sulfur_dioxide", "Free sulfur dioxide", "mg/dm³"),
    ("total_sulfur_dioxide", "Total sulfur dioxide", "mg/dm³"),
    ("density", "Density", "g/cm³"),
    ("ph", "pH", ""),
    ("sulphates", "Sulphates", "g/dm³"),
    ("alcohol", "Alcohol", "% vol"),
]

# The form starts with the first red wine in the UCI data (rated 5)
EXAMPLE = dict(zip([name for name, _, _ in FEATURES],
                   ["7.4", "0.7", "0", "1.9", "0.076", "11", "34", "0.9978", "3.51", "0.56", "9.4"]))


def parse(values):
    """Returns the inputs as numbers, or None if any of them is missing or out of range."""
    try:
        row = [float(values[name]) for name, _, _ in FEATURES]
    except ValueError:
        return None
    for (name, _, _), value in zip(FEATURES, row):
        if not math.isfinite(value) or value < 0 or (name == "ph" and value > 14):
            return None
    return row


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES, values=EXAMPLE)


@app.route("/predict", methods=["POST"])
def predict():
    values = {name: request.form.get(name, "").strip() for name, _, _ in FEATURES}
    row = parse(values)
    if row is None:
        return render_template("index.html", features=FEATURES, values=values,
                               error="Enter a number of 0 or more in every field (pH from 0 to 14).")

    estimate = model.predict(np.array([row]))[0]
    score = int(np.clip(np.rint(estimate), 0, 10))
    return render_template("index.html", features=FEATURES, values=values,
                           score=score, estimate=f"{estimate:.2f}")


if __name__ == "__main__":
    app.run(debug=True)
