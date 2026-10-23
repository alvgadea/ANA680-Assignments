"""
ANA680, Design a Three-Layered ANN Classifier
Author: Jesus Adrian Alvarado Gadea

Predicts whether a bank customer will churn (Exited = 1) with a three-layer
artificial neural network. The script identifies the features that are
significant for churn, encodes the categorical data, scales the features,
trains the network on 80% of the customers, and prints the confusion matrix
and accuracy for the remaining 20% (the test set).

Keep Churn_Modelling.csv in the same folder and run
    python churn_ann_classifier.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TensorFlow log messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # more consistent results across computers

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

SEED = 42
keras.utils.set_random_seed(SEED)  # repeatable weights and training order

# Step 1. Load the data and hold out the test set
dataset = pd.read_csv("Churn_Modelling.csv")
print(f"Loaded {len(dataset):,} customers with {dataset.shape[1]} columns")

X = dataset.drop(columns="Exited")
y = dataset["Exited"]

# 20% of the customers form the test set. Splitting before any analysis means
# the significance tests and the scaler only learn from the training data.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

# Step 2. Identify the significant features
# RowNumber, CustomerId and Surname only identify a customer and carry no
# information about churn, so they are not tested or used. Each remaining
# feature is tested against Exited. Chi-square suits the categorical and count
# features, and Mann-Whitney U suits the continuous features because it does
# not assume normally distributed data.
categorical = ["Geography", "Gender", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"]
continuous = ["CreditScore", "Age", "Balance", "EstimatedSalary"]

rows = []
for col in categorical:
    p = chi2_contingency(pd.crosstab(X_train_raw[col], y_train))[1]
    rows.append((col, "Chi-square", p))
for col in continuous:
    p = mannwhitneyu(X_train_raw.loc[y_train == 1, col], X_train_raw.loc[y_train == 0, col]).pvalue
    rows.append((col, "Mann-Whitney U", p))

tests = pd.DataFrame(rows, columns=["Feature", "Test", "p-value"]).sort_values("p-value")
tests["Significant"] = np.where(tests["p-value"] < 0.05, "Yes", "No")
significant = tests.loc[tests["Significant"] == "Yes", "Feature"].tolist()
not_significant = tests.loc[tests["Significant"] == "No", "Feature"].tolist()

tests["p-value"] = tests["p-value"].map(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}")
print("\nFeature significance on the training data (alpha = 0.05)")
print(tests.to_string(index=False))
print("\nSignificant features used by the ANN:", ", ".join(significant))
print("Not significant and left out:", ", ".join(not_significant))

# Step 3. Encode the categorical data
# Geography (France, Germany, Spain) and Gender become 0/1 dummy columns,
# with France and Female as the baseline categories
X_encoded = pd.get_dummies(X[significant], columns=["Geography", "Gender"], drop_first=True, dtype=int)
X_train = X_encoded.loc[X_train_raw.index]
X_test = X_encoded.loc[X_test_raw.index]
print("Encoded input columns:", ", ".join(X_encoded.columns))

# Step 4. Feature scaling
# The scaler learns each column's mean and standard deviation from the
# training set and applies the same transformation to the test set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5. Build the three-layer ANN
ann = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),             # one input per encoded column
    keras.layers.Dense(units=6, activation="relu"),     # layer 1 (hidden)
    keras.layers.Dense(units=6, activation="relu"),     # layer 2 (hidden)
    keras.layers.Dense(units=1, activation="sigmoid"),  # layer 3 (output), probability of churn
])

# Step 6. Compile and train
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("\nTraining the ANN for 100 epochs...")
history = ann.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")

# Step 7. Predict the test set and evaluate
# A predicted probability above 0.5 means the customer is expected to churn
y_pred = (ann.predict(X_test, verbose=0) > 0.5).astype(int).ravel()

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion matrix for the test set ({len(y_test):,} customers)")
print(pd.DataFrame(cm, index=["Actual stayed (0)", "Actual exited (1)"],
                   columns=["Predicted stayed (0)", "Predicted exited (1)"]))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy:.2%})")
