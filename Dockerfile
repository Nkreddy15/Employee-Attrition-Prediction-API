FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PREDICTION_LOG=/app/logs/predictions.jsonl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces runs containers as uid 1000, so use the same user everywhere.
RUN useradd -m -u 1000 user && mkdir -p /app/logs && chown -R user:user /app
COPY --chown=user:user . .
USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health', timeout=4).status == 200 else 1)"

# One worker with threads keeps the in-process /metrics counters consistent.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "30", "app:app"]
