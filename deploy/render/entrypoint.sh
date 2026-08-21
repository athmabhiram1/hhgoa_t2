#!/bin/sh
set -e

# Render injects $PORT (default 10000). nginx listens there and proxies /v1/
# to uvicorn on 127.0.0.1:8000.
PORT="${PORT:-10000}"
envsubst '$PORT' < /etc/nginx/templates/vakrag.conf.template > /etc/nginx/conf.d/default.conf

# Start the API in the background, then serve the frontend in the foreground.
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1 &
sleep 2

exec nginx -g 'daemon off;'