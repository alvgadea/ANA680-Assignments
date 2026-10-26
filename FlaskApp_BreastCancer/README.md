# Breast Cancer Prediction App (Assignment 4)

ANA680, Jesus Adrian Alvarado Gadea

A Flask app that predicts whether a breast tumor is benign or malignant from nine cell measurements. It uses XGBoost, the most accurate model from the Building Classification Models assignment (97.08% test accuracy). GitHub Actions retrains the model, tests the app, and deploys this folder to Heroku whenever the folder or its workflow changes.

## Files

- `train_model.py` retrains the XGBoost model and saves it as `model.pkl`
- `app.py` and `templates/index.html` take the user input and show the prediction
- `test_app.py` checks the app before each deployment
- `Procfile`, `requirements.txt`, and `.python-version` tell Heroku how to run the app
- The CI/CD workflow is `.github/workflows/breast_cancer_app.yaml` at the repository root

## Run locally

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

## Notes

- GitHub only runs workflows stored in `.github/workflows` at the repository root, so the workflow is there and deploys just this folder to Heroku.
- `requirements.txt` uses `xgboost-cpu`, the CPU-only build of XGBoost, because the regular package also downloads GPU libraries on Linux, which makes the Heroku build much larger.
