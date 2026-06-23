FROM python:3.12-slim

WORKDIR /app

# Install build deps + cron daemon
RUN apt-get update && apt-get install -y --no-install-recommends gcc cron && rm -rf /var/lib/apt/lists/*

# Install Python package (no ML extras on the worker — keeps image lean)
COPY pyproject.toml README.md ./
COPY accubet/ accubet/
COPY config/ config/
RUN pip install --no-cache-dir -e "."

# Data directory for the SQLite database
RUN mkdir -p /data

# Cron schedule — loaded at container startup via entrypoint
COPY scripts/crontab /etc/cron.d/accubet
RUN chmod 0644 /etc/cron.d/accubet

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

CMD ["/entrypoint.sh"]
