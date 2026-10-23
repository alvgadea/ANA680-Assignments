# Building Classification Models

ANA680, Building Classification Models  
Jesus Adrian Alvarado Gadea

Eight classifiers predict whether a breast tumor is benign (class 2) or malignant (class 4) from the [UCI Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) dataset, using the 683 patients with complete records. Each model is coded in its own Python file, and the test set holds 25% of the patients.

## Repository contents

| File | Contents |
|------|----------|
| `a_logistic_regression.py` | (a) Logistic Regression |
| `b_knn.py` | (b) KNN with k = 5 |
| `c_linear_svm.py` | (c) Linear SVM, kernel = linear |
| `d_kernel_svm.py` | (d) Kernel SVM, kernel = rbf |
| `e_naive_bayes.py` | (e) Naive Bayes |
| `f_decision_tree.py` | (f) Decision Tree |
| `g_random_forest.py` | (g) Random Forest with 10 estimators |
| `h_xgboost.py` | (h) XGBoost |
| `Classification_Models_Results.docx` | Word table of the accuracy and confusion matrix for every model |
| `breast-cancer-wisconsin.csv` | UCI Breast Cancer Wisconsin (Original) data with a header row added. It holds all 699 original records, and the scripts drop the 16 with a missing value (marked ?). The scripts also accept the name `Breast_Cancer_Data.csv`. |
| `requirements.txt` | Python packages needed to run the scripts |

## How to run

```
pip install -r requirements.txt
python a_logistic_regression.py
```

Run any of the eight files the same way from the folder that contains the CSV. Each one prints the accuracy and confusion matrix for the test set.

## Approach

1. Dropped the sample code number, which is a patient ID, and kept the nine cytology features.
2. Recoded the class as 0 for benign (2) and 1 for malignant (4), which XGBoost requires.
3. Split the patients into 75% training data (512) and 25% test data (171), stratified by class with random_state 42, so all eight models use the same test set.
4. Standardized the features for logistic regression, KNN, and both SVMs, fitting the scaler on the training data only. The tree-based models and Naive Bayes do not need scaling.

## Results

| Model | Accuracy | Benign predicted benign | Benign predicted malignant | Malignant predicted benign | Malignant predicted malignant |
|-------|---------:|---:|---:|---:|---:|
| (a) Logistic Regression | 95.91% | 106 | 5 | 2 | 58 |
| (b) KNN (k = 5) | 95.32% | 106 | 5 | 3 | 57 |
| (c) Linear SVM | 95.91% | 106 | 5 | 2 | 58 |
| (d) Kernel SVM (rbf) | 96.49% | 106 | 5 | 1 | 59 |
| (e) Naive Bayes | 95.91% | 106 | 5 | 2 | 58 |
| (f) Decision Tree | 95.91% | 105 | 6 | 1 | 59 |
| (g) Random Forest (10 estimators) | 95.91% | 106 | 5 | 2 | 58 |
| (h) XGBoost | 97.08% | 106 | 5 | 0 | 60 |

XGBoost scored highest and was the only model that caught every malignant tumor in the test set. The Malignant predicted benign column counts missed malignant tumors (false negatives), the costliest error in cancer screening.
