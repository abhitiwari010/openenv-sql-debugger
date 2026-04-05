FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy environment code
COPY . /app/

# Environment configurations setup
ENV PYTHONPATH="/app"
# Hugging Face Spaces may set PORT; must match README `app_port` (7860).
ENV PORT=7860

EXPOSE 7860

# Health check uses same default PORT as CMD (Spaces usually leave PORT=7860)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=5 \
    CMD sh -c 'curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1'

# OpenEnv app: FastAPI ASGI at server.app:app (see openenv.yaml)
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
