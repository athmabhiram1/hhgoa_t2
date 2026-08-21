# VakRAG — single-container Render deployment.
# Serves the built Vite frontend (nginx) + FastAPI backend (uvicorn) in one
# image. Embeddings, generation, and STT are API-backed (Gemini + Sarvam), so
# NO local models ship in the image — it stays ~200MB and fits the Render free
# tier (512MB RAM / 0.1 CPU). LightRAG and faster-whisper are optional imports
# in the backend and degrade gracefully when absent, so they are intentionally
# omitted here.
#
# Build context: REPO ROOT (render.yaml sets dockerContext: "."). The backend
# imports are relative (backend.main:app), so the app is copied under /app.
#
# Contract notes:
#   - Render injects $PORT (default 10000); nginx listens on $PORT and proxies
#     /v1/ to uvicorn on 127.0.0.1:8000.
#   - No .env in the image; all settings come from Render service env vars
#     (pydantic-settings reads them with case-insensitive matching).

# --- Stage 1: backend python deps -----------------------------------------
FROM python:3.12-slim AS backend-deps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /deps
COPY deploy/render/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: frontend build -----------------------------------------------
FROM node:24-alpine AS frontend-build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- Stage 3: final ---------------------------------------------------------
FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx gettext-base \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default

WORKDIR /app

# Backend deps + source
COPY --from=backend-deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY backend ./backend

# Frontend static assets
RUN mkdir -p /var/www/html
COPY --from=frontend-build /app/dist /var/www/html

# nginx vhost template (envsubst substitutes $PORT at boot) + entrypoint
COPY deploy/render/nginx.conf.template /etc/nginx/templates/vakrag.conf.template
COPY deploy/render/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Render expects the container to listen on 0.0.0.0:$PORT (default 10000).
EXPOSE 10000
ENTRYPOINT ["./entrypoint.sh"]