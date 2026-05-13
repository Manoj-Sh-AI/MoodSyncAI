# MoodSyncAI: Multimodal Emotion Intelligence Platform

## Quick Start / Running the Project

This system is designed to run in three separate terminals during local development. Start the services in the order below.

<table>
  <tr>
    <td><img src="docs/Screenshot 2026-05-09 151644.png" alt="Screenshot 1" width="400"/></td>
    <td><img src="docs/Screenshot 2026-05-09 151736.png" alt="Screenshot 2" width="400"/></td>
    <td><img src="docs/Screenshot 2026-05-09 151843.png" alt="Screenshot 3" width="400"/></td>
  </tr>
</table>

### Prerequisites

- Python 3.10 recommended
- Node.js 18.x recommended
- npm 9+ recommended
- Local model artifacts under `saved_models/`

### Install Dependencies

Python AI service:

```bash
cd py_service
pip install -r requirements.txt
```

Backend API:

```bash
cd backend
npm install
```

Frontend client:

```bash
cd frontend
npm install
```

### Environment Variables

Create environment files as needed.

Backend `.env` example:

```env
PORT=8080
CORS_ORIGIN=*
PY_SERVICE_URL=http://localhost:8001
PY_SERVICE_TIMEOUT_MS=180000
PY_SERVICE_VIDEO_TIMEOUT_MS=600000
IMAGE_MAX_UPLOAD_MB=12
AUDIO_MAX_UPLOAD_MB=30
VIDEO_MAX_UPLOAD_MB=200
```

Frontend `.env` example:

```env
VITE_API_URL=http://localhost:8080
```

The Python FastAPI service does not currently require environment variables for core runtime. It uses hardcoded defaults for model paths, local cache discovery, and port configuration when launched with the command below.

### Terminal 1 — Python AI Services

```bash
cd py_service
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

This starts the FastAPI/Python inference service. It handles AI model inference, processes text/image/audio/video requests, and runs fusion logic and summary generation.

### Terminal 2 — Backend API Server

```bash
cd backend
node --watch src/index.js
```

This starts the Node.js backend server. It manages API routing, communicates with frontend and Python services, and handles request orchestration.

### Terminal 3 — Frontend Client

```bash
cd frontend
npm run dev
```

This starts the React/Vite frontend. It handles uploads, renders visual analytics, and displays fusion results and summaries.

### Startup Validation

- Frontend: `http://localhost:5173`
- Backend health-style root: `http://localhost:8080/`
- Python service health: `http://localhost:8001/health`

## Project Overview

MoodSyncAI is a multimodal emotion analysis system that combines user-provided text with one additional media modality, currently image, audio, or video, to estimate emotional state, detect cross-modal agreement or disagreement, and generate a short natural-language interpretation.

The platform solves a common limitation in single-modality sentiment systems: spoken or written language alone often misses non-verbal cues, while visual or acoustic signals alone lack semantic context. MoodSyncAI addresses this by running independent modality-specific predictors, translating their outputs into a shared emotional/polarity space, and then computing a fused result used for presentation and summary generation.

Supported runtime modalities in the live web system:

- Text
- Image
- Audio
- Video

Runtime workflow:

1. A user selects one media modality and enters text.
2. The frontend sends a multipart request to the Node backend.
3. The backend forwards the request to FastAPI.
4. FastAPI runs modality inference and text inference.
5. A fusion function computes dominant emotion, match type, mismatch status, and fused confidence.
6. A summary generator produces a two-sentence interpretation.
7. The frontend renders per-modality confidence bars, fusion state, and the generated summary.

## Complete Project Structure

```text
Emotion_Detection/
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── README.md
├── app/
│   ├── main.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── generative_summary.py
│   │   ├── image_uploader.py
│   │   ├── results_panel.py
│   │   └── text_input.py
│   └── utils/
│       ├── __init__.py
│       ├── chart_builder.py
│       └── visualizer.py
├── backend/
│   ├── Dockerfile
│   ├── nodemon.json
│   ├── package.json
│   ├── package-lock.json
│   └── src/
│       ├── index.js
│       └── routes/
│           └── analyze.js
├── data/
│   ├── preprocessing/
│   │   ├── image_preprocessing.py
│   │   └── text_preprocessing.py
│   ├── processed/
│   └── raw/
├── docs/
│   └── web_migration.md
├── frontend/
│   ├── Dockerfile
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── src/
│       ├── api.js
│       ├── App.jsx
│       ├── main.jsx
│       ├── store.js
│       ├── styles.css
│       └── components/
│           ├── AudioUploader.jsx
│           ├── DebugPanel.jsx
│           ├── ImageUploader.jsx
│           ├── Navbar.jsx
│           ├── ResultsPanel.jsx
│           ├── SummaryBox.jsx
│           ├── TextInput.jsx
│           ├── VideoUploader.jsx
│           └── WeightSlider.jsx
├── models/
│   ├── audio/
│   │   ├── audio_features.py
│   │   └── audio_model.py
│   ├── fusion/
│   │   ├── fusion_layer.py
│   │   ├── mismatch_detector.py
│   │   └── video_text_fusion.py
│   ├── generative/
│   │   └── summary_generator.py
│   ├── text/
│   │   ├── attention_weights.py
│   │   └── sentiment_model.py
│   ├── video/
│   │   ├── video_features.py
│   │   ├── video_model.py
│   │   └── video_predictor.py
│   └── visual/
│       ├── cnn_emotion.py
│       ├── face_detector.py
│       └── gradcam.py
├── notebooks/
│   ├── 1_Face-emotion-detection.ipynb
│   ├── 2_text_model_experiments.ipynb
│   ├── 3_fusion_image_text.ipynb
│   ├── 4_tests_and_config.ipynb
│   ├── 5_audio_emotion_recognition.ipynb
│   ├── 6_fusion_audio_text.ipynb
│   ├── 7_video_emotion_recognition_ravdess.ipynb
│   └── 8_video_text_fusion.ipynb
├── py_service/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── saved_models/
│   ├── cnn_emotion.h5
│   ├── audio_sentiment/
│   │   ├── audio_model_meta.json
│   │   └── best_audio_model.pt
│   ├── text_sentiment/
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── text_model_meta.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   └── video_emotion/
│       ├── best_video_model.pt
│       └── video_model_meta.json
├── tests/
│   ├── test_fusion.py
│   ├── test_generator.py
│   ├── test_text_model.py
│   └── test_visual_model.py
└── training/
    ├── train_cnn.py
    ├── train_fusion.py
    ├── train_text.py
    └── configs/
        ├── cnn_config.yaml
        ├── fusion_config.yaml
        └── text_config.yaml
```

### What Each Major Area Does

`frontend/`

- Production-facing web client.
- Contains the active React UI, local state store, upload components, result renderer, and API client.
- Sends requests only to the Node backend, never directly to FastAPI.

`backend/`

- Thin orchestration and proxy layer.
- Accepts browser requests, validates required fields, applies upload size limits, and forwards binary payloads to the Python inference service.
- Central place for HTTP security middleware, CORS, rate limiting, and service URL configuration.

`py_service/`

- Active multimodal inference service.
- Loads text, image, audio, and video models lazily, performs fusion, and generates human-readable summaries.
- This is the live serving path used by the frontend and backend stack.

`models/`

- Research and reusable inference/training support modules.
- Contains more explicit model wrappers and fusion utilities than the web stack currently uses.
- Important for understanding training provenance and architectural intent, but most of these modules are not imported by `py_service/app.py` at runtime.

`training/`

- Offline training scripts and YAML configs.
- Documents intended architectures, model weights, and fusion defaults used during experimentation.

`tests/`

- Test suite for research modules and abstractions.
- Several tests target older interfaces that differ from the current FastAPI serving layer, so they should be treated as partial coverage rather than guaranteed validation of the live web stack.

`app/`

- Legacy Streamlit application and helper modules.
- Useful for understanding the original UI and migration history.
- Not part of the active React + Node + FastAPI runtime path.

`docs/web_migration.md`

- Documents the transition from Streamlit to the current web architecture.
- Helpful for understanding which concepts were preserved and which implementation details changed.

`saved_models/`

- Local model artifacts used by the Python service when present.
- These override hub-based fallbacks for text and can serve custom-trained audio/video models.

## System Architecture

### High-Level Service Topology

The deployed local development architecture is a three-tier system:

1. React/Vite frontend for user interaction and rendering.
2. Express backend for request validation, middleware, and proxying.
3. FastAPI inference service for model execution, fusion, and summary generation.

### Frontend Flow

The frontend is a single-page React application using Zustand for shared state. Users choose one of three modes: `Visual`, `Audio`, or `Video`. The selected mode determines which uploader is shown and which API function is called. Text is always required.

Frontend responsibilities:

- Collect file uploads and text input.
- Maintain local UI state such as selected mode, current file, fusion weight, loading state, and last result.
- Submit multipart form requests through axios.
- Render per-modality score bars, fused confidence, mismatch/alignment badge, generated summary, and raw JSON debug payload.

### Backend Flow

The Node.js backend is intentionally thin. Its primary role is to decouple the browser from the Python inference service and provide operational safeguards.

Backend responsibilities:

- CORS handling through `cors`.
- Security headers through `helmet`.
- Request logging through `morgan`.
- Basic API throttling through `express-rate-limit`.
- Multipart parsing through `multer` with modality-specific limits.
- Forwarding multipart payloads to FastAPI using `axios` and `form-data`.

The backend exposes separate routes for image, audio, and video because the expected file field names and Python endpoints differ.

### Python Inference Service Flow

The FastAPI service in `py_service/app.py` is the actual inference runtime.

It performs these tasks:

- Warms up visual and text models during application lifespan startup.
- Lazily loads additional summary, audio, and video assets on demand.
- Decodes incoming image, audio, and video payloads.
- Runs modality-specific prediction.
- Runs text classification.
- Applies a fusion heuristic over modality outputs.
- Generates a summary using a text-to-text model with fallback behavior.
- Returns a normalized JSON response to the backend.

### Model Inference Pipeline

Active inference paths:

- Image path: image file + text -> image classifier + text classifier -> fusion -> summary
- Audio path: audio file + text -> audio classifier + text classifier -> fusion -> summary
- Video path: video file + text -> video classifier + text classifier -> fusion -> summary

### Response Pipeline

The response contract is intentionally frontend-friendly. Each endpoint returns:

- Per-modality result objects with top label, top score, and full distribution.
- A `fusion` object summarizing agreement and confidence.
- A short `summary` string suitable for direct UI display.

## Visual Architecture Diagram

```text
User
  ↓
Frontend (React + Vite + Zustand)
  ↓
Axios multipart requests
  ↓
Node.js Backend API (Express)
  ↓
Route-level upload validation + proxy orchestration
  ↓
Python FastAPI Service
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Text Inference
  - Local Hugging Face model in saved_models/text_sentiment
  - Fallback: j-hartmann/emotion-english-distilroberta-base

Image Inference
  - Local Keras model: saved_models/cnn_emotion.h5 when available
  - Fallback: trpakov/vit-face-expression

Audio Inference
  - Custom PyTorch AudioEmotionNet checkpoint
  - Feature extraction: MFCC + delta + delta-delta + mel + chroma + ZCR + RMS

Video Inference
  - Custom PyTorch R3D-18 + temporal attention style model checkpoint
  - Uniform frame sampling + transform + clip inference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓
Fusion Layer
  ↓
Emotion Aggregation + Match/Mismatch Classification
  ↓
Summary Generator
  - google/flan-t5-base
  - Fallback deterministic summary string
  ↓
JSON Response
  ↓
Frontend Visualization Dashboard
  ↓
Bars + Status Badges + Summary + Debug Output
```

## AI/ML Pipeline Explanation

### Text Model

Active runtime implementation:

- The FastAPI service loads a Hugging Face `text-classification` pipeline.
- It first attempts to load a local model from `saved_models/text_sentiment/` using `AutoModelForSequenceClassification` and `AutoTokenizer`.
- If local loading fails, it falls back to `j-hartmann/emotion-english-distilroberta-base`.

Preprocessing and logic:

- The current FastAPI path does not apply a custom preprocessing function beyond using the raw text string.
- The model returns a multi-class emotion distribution.
- The service maps those labels into coarse polarity buckets: `positive`, `neutral`, `negative`.
- Scores for all labels mapped into the same polarity are summed.
- The result is normalized to a 3-class chart-friendly distribution.

Probability outputs:

- `top_label`: one of `positive`, `neutral`, `negative`
- `top_score`: top polarity score as percentage
- `all_scores`: normalized polarity distribution in percentages

Important nuance:

- The research module in `models/text/sentiment_model.py` documents a different text backbone, `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- That module is not the one used in the live web-serving path.

### Image Model

Active runtime implementation:

- The service first tries a local TensorFlow/Keras model from `saved_models/cnn_emotion.h5`.
- If unavailable or loading fails, it falls back to the Hugging Face image classification pipeline `trpakov/vit-face-expression`.

Preprocessing:

- Converts input to RGB.
- Center-crops to a square.
- Resizes to `224 x 224`.
- Scales pixel values to `[0, 1]` for the local Keras model.

Architecture:

- Local path: custom/local saved CNN or equivalent Keras model artifact.
- Fallback path: Vision Transformer facial expression model from Hugging Face.

Important limitation:

- Although the repository contains explicit face detection utilities under `models/visual/face_detector.py`, the active FastAPI image path does not perform separate face detection. It classifies the full uploaded image after center-crop preprocessing.

### Audio Model

Active runtime implementation:

- Uses a custom PyTorch model defined inside `py_service/app.py` as `AudioEmotionNet`.
- Loads weights and metadata from `saved_models/audio_sentiment/`.

Feature extraction pipeline:

- Resample to target sample rate, default `16000 Hz`
- Convert stereo to mono by averaging channels
- Pad or trim audio to fixed duration, default `3.0 s`
- Extract:
  - 40 MFCC coefficients
  - first-order delta MFCC
  - second-order delta MFCC
  - 64 mel-spectrogram bands
  - chroma STFT
  - zero-crossing rate
  - RMS energy
- Concatenate into a 198-channel feature tensor

Inference flow:

- Decode audio bytes using `soundfile`, with a `librosa` fallback through a temporary file when needed.
- Convert extracted features to a tensor of shape `(1, C, T)`.
- Run the CNN + BiLSTM network.
- Apply softmax to obtain per-class probabilities.
- Decode labels using a saved sklearn label encoder when present.

### Video Model

Active runtime implementation:

- Uses a PyTorch `VideoEmotionModel` defined inside `py_service/app.py`.
- Backbone: `r3d_18(weights=None)` with final pooling/classification removed.
- Temporal aggregation: attention-style pooling over temporal slices.
- Classifier head: LayerNorm -> Linear -> ReLU -> Dropout -> Linear.

Frame extraction and temporal analysis:

- Writes bytes to a temporary file on disk.
- Uses OpenCV to decode frames.
- Samples frames uniformly across the clip.
- Applies torchvision transforms: resize, center crop, tensor conversion, and normalization with ImageNet-style video statistics.
- Stacks frames into a clip tensor for inference.

Aggregation strategy:

- Temporal features from the 3D backbone are pooled using learned attention weights.
- The pooled representation is classified into the emotion classes stored in the checkpoint metadata or label encoder.

Important limitation:

- The current web runtime analyzes visual content from video but does not separately extract and fuse audio from the uploaded video file.
- Video analysis is therefore video-plus-text, not full video-audio-text multimodal fusion.

## Fusion Layer Explanation

### Active Runtime Fusion Strategy

The live system uses the `fuse()` function inside `py_service/app.py`. This is the authoritative runtime fusion behavior for the web application.

What it combines:

- Image mode: visual result + text result
- Audio mode: audio result + text result
- Video mode: video result + text result

How outputs are combined:

1. Each modality's top label is mapped to a coarse polarity via `_polarity()`.
2. Match quality is classified using polarity alignment and confidence thresholds.
3. Fused confidence is calculated as a weighted average of modality confidences.
4. Dominant emotion is selected by taking the label with the highest weighted top-score contribution.

### Match and Mismatch Logic

The runtime fusion classifier uses four categories:

- `Hard Match`: same polarity and same top emotion label
- `Soft Match`: same polarity but different top emotion labels
- `Soft Mismatch`: different polarities and at least one confidence is below threshold
- `Hard Mismatch`: different polarities and both dominant modalities exceed confidence threshold

The boolean `mismatch` flag is true whenever the match type contains `Mismatch`.

### Weighted Fusion Formula

For image and video mode, the fused confidence is effectively:

$$
fused = 100 \times (w_v \cdot c_v + w_t \cdot c_t + w_a \cdot c_a)
$$

where:

- $w_v$ is `visual_weight`
- $w_a$ is `audio_weight`
- $w_t = 1 - w_v - w_a`
- $c_v$, $c_t$, $c_a$ are modality confidences in the range $[0, 1]$

Runtime defaults:

- Image and video routes default `visual_weight = 0.55`
- Audio route hardcodes `visual_weight = 0.0` and `audio_weight = 0.5`, leaving text weight at `0.5`

### Modality Prioritization

- Image/video mode gives the user explicit control over visual-versus-text weighting through the frontend slider.
- Audio mode currently uses a fixed 50/50 split between audio and text.
- No learned fusion network is used in the active serving path.

### Confidence Handling

- Per-modality `top_score` values are converted from percentages into unit-scale confidences.
- Match severity depends on whether the dominant modality confidences exceed `confidence_threshold`, default `0.4`.
- Only the top class from each modality participates in dominant-emotion selection; full distributions are not fused in the live system.

### Synchronization Logic

- There is no temporal synchronization across modalities beyond pairing a single text input with a single uploaded image/audio/video asset.
- Audio and video are each analyzed independently as a whole sample.
- The system does not align utterances to frames or timestamps.

### Config Files and Relationship to Runtime

The repository contains `training/configs/fusion_config.yaml`, which documents a richer weighted-average and optional learned-MLP fusion design. That configuration is useful as training/research documentation, but it is not loaded by `py_service/app.py` during live inference.

This distinction is important:

- Implemented in live runtime: heuristic weighted fusion on top labels and confidences
- Present in research/config assets: probability-level weighted averaging and optional learned fusion designs

## Summary Generator

### Open-Source Model Used

The active summary generator uses `google/flan-t5-base` through Hugging Face `pipeline("text2text-generation")`.

### Where It Is Initialized

- Lazy loader: `_load_summary_pipe()` in `py_service/app.py`
- Generator function: `generate_summary()` in `py_service/app.py`

### Inference Flow

1. The service collects top labels and scores from modality inference and fusion.
2. It constructs a concise analysis context string.
3. It embeds that context in a prompt asking for exactly two short complete sentences.
4. It calls the FLAN-T5 generation pipeline with controlled sampling.
5. It post-processes the generated text and filters low-quality or field-dump style outputs.
6. If generation fails or the output is rejected, it falls back to a deterministic summary string.

### Prompt Construction

The prompt includes:

- Sanitized user text
- Visual/audio proxy label and confidence
- Text label and confidence
- Dominant fused emotion
- Fused confidence
- Match type
- Mismatch presence

The model is instructed to:

- produce exactly two short complete sentences
- avoid mechanically repeating analysis fields
- mention uncertainty if signals are mixed

### Response Generation Behavior

Generation parameters:

- `max_new_tokens=80`
- `do_sample=True`
- `temperature=0.8`
- `top_p=0.9`
- `truncation=True`

Fallback behavior is an important production safeguard. The API always attempts to preserve response shape even when generation fails.

## API Documentation

### Backend API

#### `GET /`

Purpose:

- Minimal backend liveness check.

Response:

```json
{ "status": "ok" }
```

#### `POST /api/analyze/image`

Purpose:

- Accept image + text input from the frontend and proxy it to the Python `/analyze` endpoint.

Request:

- Content type: `multipart/form-data`
- Fields:
  - `image`: required file
  - `text`: required string
  - `visual_weight`: optional float, default `0.55`

Success response:

```json
{
  "visual": {
    "top_label": "happy",
    "top_score": 82.1,
    "all_scores": {
      "happy": 82.1,
      "neutral": 9.3,
      "sad": 3.2
    }
  },
  "text": {
    "top_label": "positive",
    "top_score": 76.4,
    "all_scores": {
      "positive": 76.4,
      "neutral": 18.2,
      "negative": 5.4
    }
  },
  "fusion": {
    "mismatch": false,
    "match_type": "Soft Match",
    "visual_polarity": "positive",
    "text_polarity": "positive",
    "audio_polarity": "neutral",
    "dominant_emotion": "happy",
    "fused_confidence": 79.5,
    "visual_confidence": 82.1,
    "text_confidence": 76.4,
    "audio_confidence": 0.0
  },
  "summary": "..."
}
```

Error cases:

- `400` missing text
- `400` missing image file
- proxied Python errors such as invalid file content
- `413` upload too large
- `500` proxy or upstream service failure

#### `POST /api/analyze/audio`

Purpose:

- Accept audio + text input and proxy it to Python `/analyze_audio`.

Request:

- Content type: `multipart/form-data`
- Fields:
  - `audio`: required file
  - `text`: required string

Success response:

```json
{
  "audio": {
    "top_label": "sad",
    "top_score": 71.2,
    "all_scores": {
      "sad": 71.2,
      "neutral": 11.0,
      "happy": 6.1
    }
  },
  "text": { "top_label": "negative", "top_score": 80.0, "all_scores": {} },
  "fusion": {
    "mismatch": false,
    "match_type": "Soft Match",
    "dominant_emotion": "negative",
    "fused_confidence": 75.6
  },
  "summary": "..."
}
```

Note:

- The response uses the `audio` key instead of `visual` for the media modality.

#### `POST /api/analyze/video`

Purpose:

- Accept video + text input and proxy it to Python `/analyze_video`.

Request:

- Content type: `multipart/form-data`
- Fields:
  - `video`: required file
  - `text`: required string
  - `visual_weight`: optional float, default `0.55`

Success response:

- Same shape as image mode, except `visual` contains the video-derived emotion result.

### Python FastAPI API

#### `GET /health`

Purpose:

- Health probe for the inference service.

Response:

```json
{ "status": "ok" }
```

#### `POST /analyze`

Purpose:

- Run image + text inference, fusion, and summary generation.

Request:

- `image`: required upload file
- `text`: required form field
- `visual_weight`: optional float, default `0.55`

Response:

- `visual`, `text`, `fusion`, `summary`

#### `POST /analyze_audio`

Purpose:

- Run audio + text inference, fusion, and summary generation.

Request:

- `audio`: required upload file
- `text`: required form field

Response:

- `audio`, `text`, `fusion`, `summary`

#### `POST /analyze_video`

Purpose:

- Run video + text inference, fusion, and summary generation.

Request:

- `video`: required upload file
- `text`: required form field
- `visual_weight`: optional float, default `0.55`

Response:

- `visual`, `text`, `fusion`, `summary`

## Frontend Documentation

### React Architecture

The frontend is a compact single-page application composed of focused presentation and input components.

Core pieces:

- `App.jsx`: top-level composition and analysis trigger logic
- `store.js`: Zustand store for mode, files, text, weight, loading, and result state
- `api.js`: axios wrapper around backend endpoints
- `Navbar.jsx`: mode selection
- `ImageUploader.jsx`, `AudioUploader.jsx`, `VideoUploader.jsx`: modality-specific inputs
- `TextInput.jsx`: transcript/message input with examples
- `WeightSlider.jsx`: visual-to-text weight control for image/video modes
- `ResultsPanel.jsx`: confidence bar charts and fusion summary
- `SummaryBox.jsx`: natural-language explanation panel
- `DebugPanel.jsx`: collapsible raw JSON viewer

### Visualization System

The visualization layer is intentionally lightweight and implemented in plain React rather than with a dedicated charting library.

Rendered views include:

- Horizontal confidence bars per label
- Color-coded emotion labels
- Alignment/mismatch badges
- Fused confidence and dominant-emotion chips
- Summary card with different accent color for mismatch versus alignment

There is no advanced plotting library or timeline visualization in the current frontend runtime.

### Upload Flow

1. User selects a mode.
2. User uploads one media file appropriate to that mode.
3. User enters text.
4. Optional: user adjusts visual weight in visual/video mode.
5. `onAnalyze()` in `App.jsx` dispatches the correct API call.
6. Store is updated with loading state and final result.

### API Integration

- `analyzeImage()` -> `POST /api/analyze/image`
- `analyzeAudio()` -> `POST /api/analyze/audio`
- `analyzeVideo()` -> `POST /api/analyze/video`

All calls use multipart form payloads and expose clearer network error messages when the backend is unreachable.

### State Management

Zustand store fields:

- `mode`
- `imageFile`
- `audioFile`
- `videoFile`
- `text`
- `weight`
- `result`
- `loading`

This state model is simple and local to the client. There is no server-state library, offline cache, or background retry mechanism.

### Prediction Rendering

The frontend assumes the backend response contract is already normalized for UI rendering. It does not recalculate fusion or summary logic client-side.

## Environment Variables

### Active Variables Detected in Code

Backend:

- `PORT`: Express server port, default `8080`
- `CORS_ORIGIN`: `*` or comma-separated whitelist
- `PY_SERVICE_URL`: URL for FastAPI service, default `http://localhost:8001`
- `PY_SERVICE_TIMEOUT_MS`: timeout for image/audio inference proxy calls, default `180000`
- `PY_SERVICE_VIDEO_TIMEOUT_MS`: timeout for video inference proxy calls, default `600000`
- `IMAGE_MAX_UPLOAD_MB`: image size limit, default `12`
- `AUDIO_MAX_UPLOAD_MB`: audio size limit, default `30`
- `VIDEO_MAX_UPLOAD_MB`: video size limit, default `200`

Frontend:

- `VITE_API_URL`: backend base URL, default `http://localhost:8080`

Docker Compose-specific values:

- `PY_SERVICE_URL=http://py_service:8001`
- `CORS_ORIGIN=*`
- `VITE_API_URL=http://localhost:8080` in the production frontend build args
- `VITE_API_URL=http://backend:8080` in the optional `frontend_dev` service

### Not Detected as Active Runtime Variables

The FastAPI service currently does not use environment variables for:

- model paths
- model IDs
- threshold tuning
- device selection
- cache location

Those values are coded directly in `py_service/app.py`.

## Dependencies

### Frontend Dependencies

- `react`, `react-dom`: UI runtime
- `zustand`: lightweight state management
- `axios`: HTTP client
- `vite`: development server and build tool
- `tailwindcss`, `postcss`, `autoprefixer`: styling pipeline

Why they are used:

- The frontend is intentionally minimal and fast to iterate on.
- Zustand avoids boilerplate for cross-component shared state.
- Axios simplifies multipart upload handling and timeout configuration.

### Backend Dependencies

- `express`: HTTP server
- `multer`: multipart upload parsing in memory
- `axios`: upstream proxy HTTP client
- `form-data`: rebuild multipart bodies for FastAPI forwarding
- `cors`: browser access control
- `helmet`: security headers
- `morgan`: request logging
- `express-rate-limit`: basic abuse protection
- `dotenv`: environment variable loading
- `nodemon`: development reload support

### ML / Python Dependencies

- `fastapi`, `uvicorn[standard]`: inference API service
- `python-multipart`: form upload parsing in FastAPI
- `transformers`: text, image fallback, and summary pipelines
- `torch`, `torchvision`: audio/video model execution and video backbone utilities
- `tensorflow`: local Keras image model loading
- `numpy`: numerical processing
- `pillow`: image decoding
- `opencv-python-headless`: video decoding and frame extraction
- `librosa`: audio feature extraction and fallback decoding
- `soundfile`: direct audio decoding from bytes
- `scikit-learn`: label encoder compatibility for saved artifacts

## End-to-End Data Flow Walkthrough

1. User uploads media in the frontend and enters the associated text.
2. Frontend packages the data as multipart form data and sends it to the correct backend route based on selected mode.
3. Backend validates required fields and checks file-size constraints through the appropriate `multer` middleware.
4. Backend forwards the request body to FastAPI using `axios` and `form-data`.
5. FastAPI decodes the media payload.
6. FastAPI runs text inference on the supplied text.
7. FastAPI runs media inference:
   - image classifier for image mode
   - audio feature extractor + AudioEmotionNet for audio mode
   - frame decoder + video classifier for video mode
8. FastAPI maps dominant labels to polarity categories.
9. Fusion logic combines the top modality confidences using configured or hardcoded weights.
10. Fusion logic determines whether the signals are aligned, softly mismatched, or strongly mismatched.
11. Summary generation constructs a prompt and runs `google/flan-t5-base`.
12. If generation fails or output quality filters reject it, a deterministic fallback summary is returned.
13. FastAPI responds with modality outputs, fusion metadata, and summary text.
14. Backend relays the JSON to the frontend.
15. Frontend renders emotion bars, fused confidence, status badge, summary text, and raw debug payload.

## Error Handling + Debugging

### Common Failure Points

Model loading issues:

- Missing files under `saved_models/`
- Incompatible checkpoint structure for audio/video models
- Local TensorFlow image model failing to load
- Hugging Face model downloads blocked by connectivity or cache problems

API connection issues:

- Frontend cannot reach backend when `VITE_API_URL` is incorrect
- Backend cannot reach FastAPI when `PY_SERVICE_URL` is incorrect
- Service startup order causes transient failures if backend starts before Python is ready

CORS issues:

- Browser requests may fail if `CORS_ORIGIN` is too restrictive for the backend
- FastAPI itself allows `*`, but browser traffic goes through Express, so backend CORS is the main control point

Missing dependencies:

- `uvicorn` missing in Python environment
- `torch` / `tensorflow` installation mismatch
- Node dependencies not installed in backend or frontend

Port conflicts:

- Frontend expects `5173`
- Backend expects `8080`
- FastAPI expects `8001`

Inference crashes:

- Invalid image/audio/video content
- Unsupported codecs for uploaded media
- Video decoding failure through OpenCV
- Unsupported or corrupt audio format

### Practical Debugging Tips

Check service health first:

- Open `http://localhost:8001/health`
- Open `http://localhost:8080/`

Validate startup logs:

- FastAPI should report model warmup status at startup
- Backend should log `API server listening on http://0.0.0.0:8080`
- Frontend should print the Vite local URL

Use the frontend debug panel:

- The UI exposes the raw response object, which is the fastest way to verify schema mismatches or unexpected labels

Watch for environment mismatches:

- If the browser reports a network error, verify `VITE_API_URL`
- If backend proxy calls time out, verify Python service startup and consider increasing `PY_SERVICE_TIMEOUT_MS` or `PY_SERVICE_VIDEO_TIMEOUT_MS`

Windows-specific note:

- The video inference path already closes temporary files before OpenCV reads them, which avoids a common Windows temp-file lock issue

### Current Architectural Gaps Worth Knowing

- The repository contains richer model abstractions and fusion modules under `models/`, but the live service does not import most of them.
- Several tests target older interfaces and may not validate the current FastAPI runtime end to end.
- The image path does not run explicit face detection despite the presence of face detector utilities.
- The video path does not extract or fuse audio from uploaded videos.
- FastAPI defines `/health` twice; behavior is harmless but redundant.

## Future Improvements

Production-grade next steps for this system would include:

1. Replace synchronous request handling with async task queues for long-running video inference.
2. Add WebSocket or Server-Sent Events streaming so the frontend can display progress for large uploads and multi-stage inference.
3. Introduce GPU-aware model serving with configurable device placement.
4. Move model IDs, thresholds, and weights into environment variables or structured config files loaded at runtime.
5. Promote the richer `models/` abstractions into the live FastAPI service so research and production paths do not drift.
6. Add batching and model instance reuse strategies for higher throughput.
7. Add structured logging, request IDs, and metrics collection for observability.
8. Add end-to-end integration tests that exercise the actual frontend -> backend -> FastAPI contract.
9. Expand video processing to include audio extraction and true multimodal video-audio-text fusion.
10. Add model caching, artifact versioning, and startup validation for every required checkpoint.
11. Harden containerization for production and add a reverse proxy, TLS termination, and artifact mounts.
12. Add Kubernetes-ready deployment manifests and autoscaling policies for inference workloads.

## Notes on Active vs Legacy Code

To avoid ambiguity, the current project contains three layers of code maturity:

- Active runtime: `frontend/`, `backend/`, `py_service/`
- Research and reusable modules: `models/`, `training/`, `notebooks/`
- Legacy UI path: `app/` Streamlit application

This README documents the active runtime first, then uses the research modules to explain model provenance and intended architecture where appropriate. Where implementation and research assets differ, the live runtime behavior has been treated as authoritative.
