"""Quick checks that GitHub Actions runs before deploying to Heroku."""

from app import app

client = app.test_client()

# The form page loads
assert client.get("/").status_code == 200

# A valid set of scores returns one of the five groups
page = client.post("/predict", data={"math": 72, "reading": 72, "writing": 74}).data
assert any(f"<strong>group {g}</strong>".encode() in page for g in "ABCDE")

# Out-of-range input shows a message instead of crashing
assert b"whole number" in client.post("/predict", data={"math": 120, "reading": 72, "writing": 74}).data

print("All checks passed")
