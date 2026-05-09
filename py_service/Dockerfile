# Python FastAPI service
FROM python:3.10-slim
WORKDIR /app
# Build context is py_service/, so copy directly
COPY requirements.txt ./requirements.txt
## Install curl for healthchecks, then PyTorch CPU wheels from the official index first (pin versions)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --progress-bar off --timeout 120 --retries 5 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.2.2 torchvision==0.17.2 \
 && python -c "import torch, torchvision; print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__)" \
 && pip install --no-cache-dir -r requirements.txt
COPY . ./
# Models are expected in /app/saved_models (mounted by compose)
VOLUME ["/app/saved_models"]
ENV PYTHONUNBUFFERED=1
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
