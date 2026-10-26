"""
ANA680, Assignment 4
Author: Jesus Adrian Alvarado Gadea

Flask app that takes the nine cell measurements from index.html, runs them
through the saved XGBoost model, and shows whether the tumor is predicted to
be benign or malignant.
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
FEATURES = [
    ("clump_thickness", "Clump Thickness"),
    ("cell_size", "Uniformity of Cell Size"),
    ("cell_shape", "Uniformity of Cell Shape"),
    ("marginal_adhesion", "Marginal Adhesion"),
    ("epithelial_size", "Single Epithelial Cell Size"),
    ("bare_nuclei", "Bare Nuclei"),
    ("bland_chromatin", "Bland Chromatin"),
    ("normal_nucleoli", "Normal Nucleoli"),
    ("mitoses", "Mitoses"),
]


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
    if len(row) != len(FEATURES) or not all(1 <= v <= 10 for v in row):
        return render_template("index.html", features=FEATURES, values=values,
                               error="Enter a whole number from 1 to 10 in every field.")

    sample = np.array([row], dtype=float)
    result = "Malignant" if model.predict(sample)[0] == 1 else "Benign"
    probability = model.predict_proba(sample)[0][1]
    return render_template("index.html", features=FEATURES, values=values,
                           result=result, probability=f"{probability:.0%}")


if __name__ == "__main__":
    app.run(debug=True)
