"""
Checks the running container before it is deployed. GitHub Actions starts the
container and runs this script against it. Locally, run the container and then
python test_app.py http://localhost:5000
"""

import sys
import time
import urllib.parse
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"

# The first red wine in the UCI data (rated 5)
WINE = {"fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0, "residual_sugar": 1.9,
        "chlorides": 0.076, "free_sulfur_dioxide": 11, "total_sulfur_dioxide": 34,
        "density": 0.9978, "ph": 3.51, "sulphates": 0.56, "alcohol": 9.4}


def get(path="/", data=None):
    body = urllib.parse.urlencode(data).encode() if data else None
    with urllib.request.urlopen(BASE_URL + path, data=body, timeout=10) as response:
        return response.read().decode()


# Wait up to 30 seconds for gunicorn to start inside the container
for _ in range(30):
    try:
        home = get()
        break
    except OSError:
        time.sleep(1)
else:
    sys.exit("The app did not respond")

# The form page loads
assert "Wine Quality Prediction" in home

# The example wine gets the score the notebook model gives it
assert "Predicted quality: <strong>5</strong>" in get("/predict", WINE)

# Invalid input shows a message instead of crashing
assert "Enter a number" in get("/predict", {**WINE, "alcohol": -1})
assert "Enter a number" in get("/predict", {**WINE, "ph": "abc"})

print("All checks passed")
