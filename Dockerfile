FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY fetch_weights.sh /usr/local/bin/fetch_weights.sh
RUN chmod +x /usr/local/bin/fetch_weights.sh

ENV MODEL_DIR=/models/mbti-distilbert-model \
    LABEL_ENCODER_PATH=/app/label_encoder.pkl \
    TORCH_NUM_THREADS=1

EXPOSE 8080
CMD ["/bin/bash","-lc","fetch_weights.sh && gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --workers 1 --threads 2 --timeout 120"]
