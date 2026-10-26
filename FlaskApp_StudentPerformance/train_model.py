"""
ANA680, Midterm Part 2
Author: Jesus Adrian Alvarado Gadea

Trains a model that predicts a student's race/ethnicity group from the math,
reading and writing scores (Kaggle Students Performance in Exams data) and
saves it as model.pkl for the Flask app. GitHub Actions runs this script
before every deployment.
"""

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("StudentsPerformance.csv")
X = dataset[["math score", "reading score", "writing score"]].values
y = dataset["race/ethnicity"].values

# 80% training data and 20% test data, with the same group mix in both
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Compare a few algorithms with 5-fold cross-validation on the training data
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN (k = 31)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31)),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = {name: cross_val_score(model, X_train, y_train, cv=cv).mean()
          for name, model in candidates.items()}
for name, score in scores.items():
    print(f"{name:20s} cross-validation accuracy {score:.4f}")

# Keep the best algorithm, train it on all the training data, and check the test set
best_name = max(scores, key=scores.get)
model = candidates[best_name].fit(X_train, y_train)
baseline = (y_test == pd.Series(y_train).mode()[0]).mean()
print(f"Selected {best_name}")
print(f"Test accuracy {accuracy_score(y_test, model.predict(X_test)):.4f} "
      f"(always guessing the most common group scores {baseline:.4f})")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved model.pkl")
