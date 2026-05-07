# Python FastAPI service
FROM python:3.10-slim
WORKDIR /app
# Build context is py_service/, so copy directly
COPY requirements.txt ./requirements.txt
## Install curl for healthchecks, then PyTorch CPU wheels from the official index first (pin versions)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
	 torch==2.6.0+cpu \
 && python -c "import torch; print('Torch:', torch.__version__)" \
 && pip install --no-cache-dir -r requirements.txt
COPY . ./
# Models are expected in /app/saved_models (mounted by compose)
VOLUME ["/app/saved_models"]
ENV PYTHONUNBUFFERED=1
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
