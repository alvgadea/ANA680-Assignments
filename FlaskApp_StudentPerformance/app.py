"""
ANA680, Midterm Part 2
Author: Jesus Adrian Alvarado Gadea

Flask app that takes a student's math, reading and writing scores from
index.html, runs them through the saved model, and shows the predicted
race/ethnicity group.
"""

import os
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model once, when the app starts
with open(os.path.join(os.path.dirname(__file__), "model.pkl"), "rb") as f:
    model = pickle.load(f)

# Form field names and labels, in the order the model was trained on
FEATURES = [("math", "Math score"), ("reading", "Reading score"), ("writing", "Writing score")]


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES, values={})


@app.route("/predict", methods=["POST"])
def predict():
    values = {name: request.form.get(name, "") for name, _ in FEATURES}
    try:
        row = [int(values[name]) for name, _ in FEATURES]
    except ValueError:
        row = []
    if len(row) != len(FEATURES) or not all(0 <= v <= 100 for v in row):
        return render_template("index.html", features=FEATURES, values=values,
                               error="Enter a whole number from 0 to 100 for each score.")

    sample = np.array([row], dtype=float)
    group = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0].max()
    return render_template("index.html", features=FEATURES, values=values,
                           result=group, probability=f"{probability:.0%}")


if __name__ == "__main__":
    app.run(debug=True)
