# Bank Customer Churn Prediction with a Three-Layer ANN

ANA680, Design a Three-Layered ANN Classifier  
Jesus Adrian Alvarado Gadea

This project predicts whether a bank customer will leave the bank (Exited = 1) from the Churn_Modelling dataset of 10,000 customers, using a three-layer artificial neural network built with TensorFlow and Keras.

## Repository contents

| File | Description |
|------|-------------|
| `churn_ann_classifier.py` | Identifies the significant features, trains the ANN, and prints the confusion matrix and accuracy |
| `Churn_Modelling.csv` | Bank customer dataset (10,000 rows, 14 columns) |
| `requirements.txt` | Python packages needed to run the script |

## How to run

```
pip install -r requirements.txt
python churn_ann_classifier.py
```

Run the script from the folder that contains `Churn_Modelling.csv`. TensorFlow currently supports Python 3.10 through 3.13. On Windows, if pip stops with a long path error while installing TensorFlow, use Python from python.org (its installer offers "Disable path length limit") instead of the Microsoft Store version.

## Approach

1. Held out 20% of the customers as a stratified test set before any analysis.
2. Excluded the identifier columns RowNumber, CustomerId, and Surname, which carry no information about churn.
3. Tested each remaining feature against Exited on the training data, using chi-square for categorical and count features and Mann-Whitney U for continuous features. Features with p < 0.05 became the model inputs. In a 5-fold cross-validation on the training data, the network scored about the same with these seven features as with all ten (86.0% and 85.8%), so the simpler input set was kept.
4. One-hot encoded Geography and Gender and standardized every input with a StandardScaler fitted on the training set.
5. Built the ANN with two hidden layers of 6 ReLU units and a sigmoid output layer, compiled it with the adam optimizer and binary cross-entropy loss, and trained it for 100 epochs with a batch size of 32. A larger network with 16 and 8 hidden units did not score higher in that cross-validation.
6. Classified a test customer as a churner when the predicted probability was above 0.5, then printed the confusion matrix and accuracy.

The cross-validation comparisons in steps 3 and 5 were run separately while designing the model and are not part of `churn_ann_classifier.py`.

## Results

| Feature group | Features |
|---------------|----------|
| Significant (p < 0.05) | NumOfProducts, Age, Geography, IsActiveMember, Balance, Gender, CreditScore |
| Not significant | EstimatedSalary, HasCrCard, Tenure |

CreditScore is the weakest of the significant features (p = 0.009).

Confusion matrix for the test set of 2,000 customers

| | Predicted stayed (0) | Predicted exited (1) |
|---|---|---|
| Actual stayed (0) | 1,531 | 62 |
| Actual exited (1) | 209 | 198 |

Test accuracy is 86.45%. Predicting that every customer stays would score 79.65%, so the network adds about 7 percentage points. Exact numbers can shift slightly on a different computer or TensorFlow version.
