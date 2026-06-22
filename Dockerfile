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

EXPOSE 8080

CMD ["uvicorn", "accubet.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
