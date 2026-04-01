#!/bin/bash
# Startup script for Railway deployment.
# Injects the dynamic $PORT into the config before launching.

set -e

LISTEN_PORT="${PORT:-8080}"

# Replace the hardcoded port in config with Railway's assigned port
sed "s/\"listen_port\": 8080/\"listen_port\": ${LISTEN_PORT}/" \
    config_railway.json > /tmp/config_active.json

echo "Starting Freqtrade on port ${LISTEN_PORT} (dry_run mode)"

exec freqtrade trade \
    --config /tmp/config_active.json \
    --strategy ClaudeStrategy \
    --logfile /freqtrade/user_data/logs/freqtrade.log
