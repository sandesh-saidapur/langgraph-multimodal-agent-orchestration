# Build:  docker build -t multimodal-agent .
# Run:    docker run -p 8000:8000 multimodal-agent
# Then open http://127.0.0.1:8000/docs

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Checkpoints and uploads live in /data so they can be mounted as a volume.
ENV CHECKPOINT_DB=/data/checkpoints.db \
    UPLOAD_DIR=/data/uploads
RUN mkdir -p /data

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
