# Wine Quality Prediction App (Assignment 5, Problem 1)

ANA680, Jesus Adrian Alvarado Gadea

A Flask app that predicts a wine's quality score (0 to 10) from the 11 lab measurements in the UCI Wine Quality dataset. The app runs in a Docker container on Heroku. Whenever this folder or its workflow changes, GitHub Actions runs the notebook to train the model, builds and tests the container, then deploys this folder to Heroku, which builds the same Dockerfile.

## Files

- `wine_quality_model.ipynb` compares three algorithms, trains the best one (Random Forest), and saves it as `model.joblib`, which the container uses
- `app.py` and `templates/index.html` take the measurements and show the predicted score
- `Dockerfile` builds the container image, and `heroku.yml` tells Heroku to build the app from it
- `requirements.txt` lists the Python packages installed in the container
- `test_app.py` checks the running container before each deployment
- `winequality-red.csv` and `winequality-white.csv` are the UCI data files
- The CI/CD workflow is `.github/workflows/wine_quality_app.yaml` at the repository root

## Run the container locally

Run the notebook first to create `model.joblib`, then:

```
docker build -t wine-quality-app .
docker run --rm -p 5000:5000 wine-quality-app
```

Then open http://localhost:5000.

## Notes

- The model uses both the red and white wine files. 1,179 rows are exact duplicates, so they were removed before splitting the data, which keeps the same wine from appearing in both the training and test sets.
- Random Forest had the lowest cross-validation error. On the test set it has an RMSE of 0.68 (0.88 when always guessing the average score), gives the exact score for 55.5% of wines after rounding, and is within one point for 97.6%.
- GitHub only runs workflows stored in `.github/workflows` at the repository root, so the workflow is there and deploys just this folder to Heroku.
