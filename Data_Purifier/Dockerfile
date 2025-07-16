FROM python:3.10-slim-buster AS builder

# Dockerfile
# This Dockerfile uses a multi-stage build to create a lean and efficient Docker image
# for the Data Purification System. Multi-stage builds help reduce the final image size
# by separating build-time dependencies from runtime dependencies.

# --- Stage 1: Builder ---
# This stage is responsible for installing build-time dependencies and Python packages.
# It will not be part of the final image, ensuring a smaller footprint.

# Use a slim Python base image for a smaller starting point.
FROM python:3.10-slim-buster AS builder

# Set the working directory inside the container.
WORKDIR /app

# Install system-level build dependencies required for some Python packages.
# `build-essential` and `gcc` are common for compiling C extensions.
# `rm -rf /var/lib/apt/lists/*` cleans up apt cache to keep the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt file first to leverage Docker's layer caching.
# If requirements.txt doesn't change, this layer and subsequent ones can be cached.
COPY requirements.txt .

# Install Python dependencies from requirements.txt.
# `--no-cache-dir` prevents pip from storing downloaded packages, further reducing layer size.
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy and NLTK models in the builder stage.
# This ensures they are downloaded only once during the build process.
# These models are necessary for the MetaAnalyzerAgent and TransformerAgent.
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# --- Stage 2: Production ---
# This stage is the final, lean image that will be deployed.
# It copies only the necessary runtime artifacts from the builder stage.

# Use the same slim Python base image for consistency and small size.
FROM python:3.10-slim-buster

# Set the working directory for the application.
WORKDIR /app

# Copy installed Python packages from the builder stage's site-packages directory.
# This avoids reinstalling dependencies and keeps the final image clean.
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy any executables or scripts installed by pip from the builder stage.
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy the rest of the application code from the local directory into the container.
# This should be done after copying dependencies to maximize layer caching efficiency.
COPY . .

# Set environment variables for production.
# PYTHONUNBUFFERED=1 ensures that Python's stdout and stderr are not buffered,
# which is important for real-time logging in containerized environments.
ENV PYTHONUNBUFFERED=1

# Define the command to run the application when the container starts.
# This specifies that the `main.py` script should be executed using the Python interpreter.
CMD ["python", "main.py"]