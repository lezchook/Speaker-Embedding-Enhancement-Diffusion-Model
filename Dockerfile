FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget \
 && rm -rf /var/lib/apt/lists/*
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /workspace/checkpoints /workspace/best_models /workspace/data
VOLUME ["/workspace/checkpoints", "/workspace/best_models", "/workspace/data"]


CMD ["python", "main.py", "--config", "/app/config.yaml", "--download", "True"]