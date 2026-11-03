# Container image for the wine quality Flask app
FROM python:3.13-slim

WORKDIR /app

# Install the Python packages first, so Docker can reuse this layer when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and the trained model
COPY app.py model.joblib ./
COPY templates/ templates/

# Heroku sets PORT when it starts the container. 5000 is used when running it locally.
# Heroku also runs containers as a non-root user, so gunicorn's optional control socket is turned off.
ENV PORT=5000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --no-control-socket app:app"]
