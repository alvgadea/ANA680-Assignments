"""
ANA680, Assignment 4
Author: Jesus Adrian Alvarado Gadea

Retrains the best model from the Building Classification Models assignment
(XGBoost, 97.08% test accuracy) and saves it as model.pkl for the Flask app.
"""

import pickle

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Same preparation as before: drop the 16 records with a missing value ("?"),
# skip the patient ID column, and recode the classes 2 and 4 as 0 and 1
dataset = pd.read_csv("breast-cancer-wisconsin.csv", na_values="?").dropna()
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].astype(int).map({2: 0, 4: 1}).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved model.pkl")
