# Building Classification Models

ANA680, Jesus Adrian Alvarado Gadea

Eight classification models (a to h), each in its own Python file, trained on the UCI Breast Cancer Wisconsin (Original) data with 25% of the patients held out for testing. The accuracy and confusion matrix for each model are in `Classification_Models_Results.docx`.

To run a model from this folder: `python a_logistic_regression.py` (packages are listed in `requirements.txt`).

## Notes

- `breast-cancer-wisconsin.csv` is the original UCI data file with a header row added, because the CSV linked in D2L contained the dataset description. The scripts drop the 16 records with a missing value, leaving 683 patients.
- The sample code number is a patient ID, so the models use the other 9 features.
- Classes 2 (benign) and 4 (malignant) are recoded as 0 and 1 because XGBoost requires 0/1 labels.
