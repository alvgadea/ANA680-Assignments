# Wine Quality with Linear Regression on SageMaker (Assignment 5, Problem 2)

ANA680, Jesus Adrian Alvarado Gadea

Problem 1 repeated with linear regression on Amazon SageMaker, in two ways, following the module's Iris example:

- **(a) Without container technology:** scikit-learn's `LinearRegression` is trained and tested inside the notebook.
- **(b) With container technology:** SageMaker's built-in Linear Learner algorithm runs as a Docker container on AWS. It is trained on data in S3, deployed to a SageMaker endpoint, used to predict the test wines, and then deleted.

## Files

- `wine_quality_sagemaker.ipynb` has both parts and a comparison
- `winequality-red.csv` and `winequality-white.csv` are the UCI data files (the notebook downloads them if they are missing)

## Notes

- The assignment asks for SageMaker Studio Lab, but Studio Lab no longer accepts new customers, and AWS names SageMaker Studio as its replacement ([AWS notice](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-availability-change.html)). The notebook was run in a SageMaker Studio JupyterLab space, which uses the Studio domain's execution role for part (b).
- The data preparation and train/test split match Problem 1, so both problems are scored on the same test wines.
- On the 1,064 test wines, part (a) has an RMSE of 0.734 and part (b) has 0.733 (R² of 0.30 and 0.31). Both are a little worse than Problem 1's Random Forest (RMSE 0.68).
- The Iris example uses SageMaker SDK v1 calls, so the notebook installs SDK v2 and uses the v2 names (`image_uris.retrieve`, `TrainingInput`, `CSVSerializer`, `JSONDeserializer`).
- The AWS account's training quota is 0 for every CPU instance type (new accounts start that way), so training uses one `ml.g5.xlarge` GPU, which Linear Learner supports. The endpoint uses `ml.m5.xlarge`, an instance type in the SageMaker free tier, and is deleted at the end so it stops billing.
