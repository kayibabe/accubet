FROM python:3.12-slim

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install Python package (no ML extras on the worker — keeps image lean)
COPY pyproject.toml README.md ./
COPY accubet/ accubet/
COPY config/ config/
RUN pip install --no-cache-dir -e "."

# Data directory for the SQLite database
RUN mkdir -p /data

# Minimal HTTP health server so Fly's healthcheck passes
# The real work is done via `fly ssh console` or Fly scheduled machines
COPY docker/healthcheck.py /app/docker/healthcheck.py

EXPOSE 8080

CMD ["python", "docker/healthcheck.py"]
