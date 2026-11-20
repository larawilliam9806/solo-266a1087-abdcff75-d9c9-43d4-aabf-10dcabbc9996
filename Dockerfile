FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EXAM_DB=/app/data/app.db \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY examapp ./examapp
COPY wsgi.py .

# Data lives on a volume mounted at /app/data; create it up front and let the
# unprivileged runtime user create the SQLite files there.
RUN mkdir -p /app/data \
    && useradd --system --uid 10001 --no-create-home appuser \
    && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

# Compute-and-save runs inside the single gunicorn worker's background thread;
# 8 request threads keep pages and exports responsive during a recompute.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "120", "wsgi:app"]
