# ─────────────────────────────────────────────────────────────────────
# Dockerfile for HuggingFace Spaces (Docker SDK)
#
# v2 changes:
#   • Installs Playwright + Chromium for JS-rendered page scraping
#   • Pre-downloads BOTH models (BERT QA + DistilBERT sentiment)
#   • Larger image (~2.6 GB) — trades size for full functionality
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    PLAYWRIGHT_BROWSERS_PATH=/home/user/.cache/ms-playwright \
    PORT=7860

# System deps including Playwright Chromium runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        # Chromium runtime dependencies
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
        libatspi2.0-0 \
        libx11-xcb1 \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# CPU-only PyTorch first (big cacheable layer)
RUN pip install --user --no-cache-dir \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user:user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser (skip deps — we installed them above)
RUN python -m playwright install chromium

# Pre-download BOTH models so first request is fast
ARG HF_MODEL_NAME=deepset/bert-base-cased-squad2
ARG SENTIMENT_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
ENV HF_MODEL_NAME=${HF_MODEL_NAME}
ENV SENTIMENT_MODEL_NAME=${SENTIMENT_MODEL_NAME}
RUN python -c "import os; \
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification; \
qa = os.environ['HF_MODEL_NAME']; \
sent = os.environ['SENTIMENT_MODEL_NAME']; \
AutoTokenizer.from_pretrained(qa); \
AutoModelForQuestionAnswering.from_pretrained(qa); \
AutoTokenizer.from_pretrained(sent); \
AutoModelForSequenceClassification.from_pretrained(sent); \
print('Models pre-downloaded:', qa, '+', sent)"

# App code
COPY --chown=user:user src/       ./src/
COPY --chown=user:user templates/ ./templates/
COPY --chown=user:user static/    ./static/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

# Single worker — both models eat memory, and Playwright spawns child processes
CMD ["gunicorn", "src.app:app", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
