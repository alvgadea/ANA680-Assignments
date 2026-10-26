"""Quick checks that GitHub Actions runs before deploying to Heroku."""

from app import app

client = app.test_client()

# The form page loads
assert client.get("/").status_code == 200

# Two patients from the dataset, one benign (sample 1000025) and one malignant (sample 1017122)
benign = dict(clump_thickness=5, cell_size=1, cell_shape=1, marginal_adhesion=1, epithelial_size=2,
              bare_nuclei=1, bland_chromatin=3, normal_nucleoli=1, mitoses=1)
malignant = dict(clump_thickness=8, cell_size=10, cell_shape=10, marginal_adhesion=8, epithelial_size=7,
                 bare_nuclei=10, bland_chromatin=9, normal_nucleoli=7, mitoses=1)
assert b"<strong>Benign</strong>" in client.post("/predict", data=benign).data
assert b"<strong>Malignant</strong>" in client.post("/predict", data=malignant).data

# Out-of-range input shows a message instead of crashing
assert b"whole number" in client.post("/predict", data={**benign, "mitoses": 11}).data

print("All checks passed")
