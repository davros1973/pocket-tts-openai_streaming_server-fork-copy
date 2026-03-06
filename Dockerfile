# Dockerfile for PocketTTS OpenAI-Compatible Server — xarta variant
# CPU-only inference (pocket-tts runs efficiently on CPU without GPU)
# Runs as root for simplicity; templates, voices, and logs are volume-mounted.

FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install "setuptools<70" wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt


FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application source (templates/ and voices/ are volume-mounted at runtime)
COPY app/ ./app/
COPY server.py ./

# Pre-create directories that will be volume-mounted or used at runtime
RUN mkdir -p /app/logs /app/voices /app/templates /app/static \
             /app/voices_cache /app/voices_user /app/data /root/.cache/huggingface

ENV POCKET_TTS_HOST=0.0.0.0 \
    POCKET_TTS_PORT=8000 \
    POCKET_TTS_VOICES_DIR=/app/voices \
    POCKET_TTS_LOG_DIR=/app/logs \
    POCKET_TTS_LOG_LEVEL=INFO \
    POCKET_TTS_STREAM_DEFAULT=true \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "server.py"]
