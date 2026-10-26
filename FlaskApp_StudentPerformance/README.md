# Student Race/Ethnicity Prediction App (Midterm Part 2)

ANA680, Jesus Adrian Alvarado Gadea

A Flask app that predicts a student's race/ethnicity group (A to E) from the math, reading, and writing scores in the Kaggle Students Performance in Exams dataset. GitHub Actions trains the model, tests the app, and deploys this folder to Heroku whenever the folder or its workflow changes.

## Files

- `train_model.py` compares three algorithms with cross-validation, trains the best one, and saves it as `model.pkl`
- `app.py` and `templates/index.html` take the scores and show the predicted group
- `test_app.py` checks the app before each deployment
- `Procfile`, `requirements.txt`, and `.python-version` tell Heroku how to run the app
- `StudentsPerformance.csv` is the Kaggle dataset
- The CI/CD workflow is `.github/workflows/student_performance_app.yaml` at the repository root

## Run locally

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

## Notes

- Logistic regression had the best cross-validation accuracy (33.6%) and scores 37.0% on the test set, compared with 32.0% for always guessing the most common group. The three scores carry little information about race/ethnicity, so accuracy stays low with any algorithm.
- GitHub only runs workflows stored in `.github/workflows` at the repository root, so the workflow is there and deploys just this folder to Heroku.
