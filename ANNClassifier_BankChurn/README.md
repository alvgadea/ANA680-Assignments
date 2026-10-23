# Three-Layered ANN Classifier

ANA680, Jesus Adrian Alvarado Gadea

`churn_ann_classifier.py` identifies the significant features in `Churn_Modelling.csv`, trains a three-layer ANN on 80% of the customers, and prints the confusion matrix and accuracy for the other 20%. Test accuracy was 86.45%.

To run from this folder: `python churn_ann_classifier.py` (packages are listed in `requirements.txt`).

## Notes

- Significance was tested on the training data with chi-square (categorical features) and Mann-Whitney U (continuous features) at p < 0.05. EstimatedSalary, HasCrCard, and Tenure were not significant, so the network uses the other seven features. In a separate check, using all ten gave about the same accuracy.
- RowNumber, CustomerId, and Surname are identifiers and were not used.
