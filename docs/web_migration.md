# MoodSyncAI — Streamlit → React + Node migration

This document outlines the new architecture, API contract, and how to run the stack locally.

## Architecture

- Frontend: React (Vite) + Tailwind + Axios + Zustand (optional)
- Backend: Node.js (Express)
- ML Inference: Python FastAPI service reusing existing model logic

```
frontend/      # React app (Vite)
backend/       # Express REST API (port 8080)
py_service/    # FastAPI ML microservice (port 8001)
```

Data flow:
1) User uploads face image + enters text + sets `visual_weight` in React
2) Frontend POSTs multipart form to `POST /api/analyze`
3) Express forwards to Python FastAPI `/analyze`
4) FastAPI runs image + text classifiers, fuses results, returns JSON
5) Frontend renders confidence bars, fusion status, and summary

## API Contract

- Endpoint: `POST /api/analyze`
  - Content-Type: `multipart/form-data`
  - Fields:
    - `image`: file (jpg/png/webp)
    - `text`: string (required, non-empty)
    - `visual_weight`: float in [0,1] (default 0.55)
  - Response 200 JSON:
    ```json
    {
      "visual": {
        "top_label": "happy",
        "top_score": 87.5,
        "all_scores": { "happy": 87.5, "neutral": 6.2, "surprise": 6.3 }
      },
      "text": {
        "top_label": "joy",
        "top_score": 81.2,
        "all_scores": { "joy": 81.2, "neutral": 10.1, "sadness": 8.7 }
      },
      "fusion": {
        "mismatch": false,
        "visual_polarity": "positive",
        "text_polarity": "positive",
        "dominant_emotion": "happy",
        "fused_confidence": 85.1,
        "visual_confidence": 87.5,
        "text_confidence": 81.2
      },
      "summary": "Both facial expression ..."
    }
    ```
  - Errors:
    - 400 `{ error | detail: string }` invalid image/text
    - 500 `{ error: string }` internal error

## Setup

Prereqs: Node 18+, Python 3.10+, optional: conda; models in `saved_models/` are auto-detected.

1) Python service
```
cd py_service
pip install -r requirements.txt
python app.py  # runs on :8001
```

2) Backend
```
cd backend
npm install
copy .env.example .env   # set PY_SERVICE_URL if different
npm run dev               # runs on :8080
```

3) Frontend
```
cd frontend
npm install
npm run dev               # opens Vite dev server :5173
```

Open http://localhost:5173 and test. Ensure Python (:8001) and Node (:8080) are running.

## Notes on Python logic
- Visual model: uses local `saved_models/cnn_emotion.h5` if present else ViT HF pipeline `trpakov/vit-face-expression`
- Text model: uses local `saved_models/text_sentiment` if present else HF `j-hartmann/emotion-english-distilroberta-base`
- Fusion: weighted average (default 0.55 visual)
- Summary: deterministic template message based on mismatch/alignment

## Deployment
- Recommended: containerise both services and deploy behind a reverse proxy (Nginx) or on a PaaS.
- Static frontend can be built and served via CDN or from the Node server in production.

### Docker (optional example)
- `backend/Dockerfile` (Node 18-alpine)
- `py_service/Dockerfile` (python:3.10-slim + `requirements.txt`)
- `docker-compose.yml` wiring ports 5173 (dev) / 8080 / 8001

> For GPU acceleration, run the Python service on a GPU-enabled host or configure a CUDA-enabled base image and install `torch` with CUDA.
