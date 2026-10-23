"""
ANA680 Building Classification Models
(e) Naive Bayes
Author: Jesus Adrian Alvarado Gadea

Classifies breast tumors in the UCI Breast Cancer Wisconsin (Original) data as
benign or malignant with a Gaussian Naive Bayes model.
It prints the accuracy and confusion matrix for a test set that holds 25% of
the patients.

Keep the dataset CSV in the same folder and run
    python e_naive_bayes.py
"""

import os

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# UCI Breast Cancer Wisconsin (Original) data file. The assignment page calls
# it Breast_Cancer_Data.csv, so that file name works too.
DATA_FILE = "breast-cancer-wisconsin.csv"
if not os.path.exists(DATA_FILE):
    DATA_FILE = "Breast_Cancer_Data.csv"

# Read every cell as a number. A header row or a "?" (the UCI code for a
# missing value) turns blank, and those rows are dropped.
dataset = pd.read_csv(DATA_FILE, header=None)
dataset = dataset.apply(pd.to_numeric, errors="coerce").dropna()

# The first column is the sample code number, a patient ID, so it is not a feature.
# Class 2 (benign) becomes 0 and class 4 (malignant) becomes 1.
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].astype(int).map({2: 0, 4: 1}).values

# 75% training data and 25% test data, with the same benign/malignant mix in both
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# Gaussian Naive Bayes estimates each feature's mean and variance per class,
# so its predictions do not change with feature scaling.

# Train the model and predict the test set
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Results
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Naive Bayes")
print(f"Data: {len(dataset)} patients, {X.shape[1]} features, "
      f"{len(y_test)} patients in the test set")
print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
print("Confusion matrix (rows = actual, columns = predicted)")
print(pd.DataFrame(cm, index=["Actual benign", "Actual malignant"],
                   columns=["Predicted benign", "Predicted malignant"]))
