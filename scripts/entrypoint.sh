#!/bin/sh
set -e

# Dump process environment into a file that cron can source.
# Fly secrets (APIFOOTBALL_KEY, etc.) are in the process env, not /etc/environment.
printenv | grep -v '^_=' | sed "s/'/'\\\\''/g; s/=/='/" | sed "s/$/'/" \
  | awk '{print "export " $0}' > /etc/cron_env

# Start the cron daemon in the background
cron

echo "AccuBet started - cron daemon running, launching API server..."

# Hand off to uvicorn (PID 1)
exec uvicorn accubet.api.app:app --host 0.0.0.0 --port 8080