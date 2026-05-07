from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.video import r3d_18

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
SAVED_MODELS = BASE_DIR / "saved_models"
LOCAL_TEXT_MODEL_DIR = SAVED_MODELS / "text_sentiment"
LOCAL_VISUAL_MODEL_PATHS = [
    SAVED_MODELS / "cnn_emotion.h5",
    SAVED_MODELS / "cnn_emotion.pt",
]

# Additional artefact paths
AUDIO_DIR = SAVED_MODELS / "audio_sentiment"
VIDEO_DIR = SAVED_MODELS / "video_emotion"

# Video normalisation (ImageNet stats used during training)
IMAGENET_MEAN = (0.43216, 0.394666, 0.37645)
IMAGENET_STD = (0.22803, 0.22145, 0.216989)

from contextlib import asynccontextmanager


# ── Lifespan event handler ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Warmup models to avoid first-call timeouts
    print("[lifespan] warming up models...")
    try:
        _load_visual_pipe()
        print("[lifespan] visual model loaded.")
    except Exception as e:
        print("[lifespan] visual model load failed:", e)
    try:
        _load_text_pipe()
        print("[lifespan] text model loaded.")
    except Exception as e:
        print("[lifespan] text model load failed:", e)
    print("[lifespan] warmup complete.")
    yield
    # Shutdown: any cleanup can go here
    print("[lifespan] shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="MoodSyncAI Python Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def _health():
    """Simple health check endpoint."""
    return {"status": "ok"}


# ── Lazy pipelines ────────────────────────────────────────────────────────────
_visual_pipe = None
_text_pipe = None
_audio_model: Optional[nn.Module] = None
_audio_le = None
_audio_meta: Optional[dict] = None
_video_model: Optional[nn.Module] = None
_video_le = None
_video_meta: Optional[dict] = None


def _load_visual_pipe():
    global _visual_pipe
    if _visual_pipe is None:
        from transformers import pipeline

        _visual_pipe = pipeline(
            "image-classification",
            model="trpakov/vit-face-expression",
            device=-1,
        )
    return _visual_pipe


def _load_text_pipe():
    global _text_pipe
    if _text_pipe is None:
        from transformers import pipeline

        if LOCAL_TEXT_MODEL_DIR.exists():
            try:
                _text_pipe = pipeline(
                    "text-classification",
                    model=str(LOCAL_TEXT_MODEL_DIR),
                    tokenizer=str(LOCAL_TEXT_MODEL_DIR),
                    top_k=None,
                    device=-1,
                )
                return _text_pipe
            except Exception:
                pass
        _text_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1,
        )
    return _text_pipe


# ── Audio helpers ───────────────────────────────────────────────────────────


class AudioEmotionNet(nn.Module):
    def __init__(self, in_channels: int = 198, num_classes: int = 7):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
        )
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])


def _audio_extract_features(
    audio: np.ndarray,
    sr: int,
    *,
    target_sr: int = 16000,
    n_mfcc: int = 40,
    n_mels: int = 64,
    hop_length: int = 512,
    n_fft: int = 2048,
    max_duration_s: float = 3.0,
) -> np.ndarray:
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    max_samples = int(target_sr * max_duration_s)
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]

    mfcc = librosa.feature.mfcc(
        y=audio, sr=target_sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio, sr=target_sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
        ),
        ref=np.max,
    )
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=target_sr, hop_length=hop_length, n_fft=n_fft
    )
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)

    feat = np.concatenate([mfcc, delta, delta2, mel, chroma, zcr, rms], axis=0)
    return feat.astype(np.float32)


def _load_audio_assets():
    global _audio_model, _audio_le, _audio_meta
    if _audio_model is not None and _audio_le is not None and _audio_meta is not None:
        return _audio_model, _audio_le, _audio_meta
    meta_path = AUDIO_DIR / "audio_model_meta.json"
    le_path = AUDIO_DIR / "label_encoder.pkl"
    ckpt_path = AUDIO_DIR / "best_audio_model.pt"
    if not (meta_path.exists() and le_path.exists() and ckpt_path.exists()):
        raise RuntimeError(
            "Audio model artefacts missing in saved_models/audio_sentiment"
        )
    with open(meta_path) as f:
        _audio_meta = json.load(f)
    import pickle

    with open(le_path, "rb") as f:
        _audio_le = pickle.load(f)
    in_ch = int(_audio_meta.get("feature_dim", 198))
    n_classes = int(
        _audio_meta.get("num_classes", len(getattr(_audio_le, "classes_", [])) or 7)
    )
    model = AudioEmotionNet(in_channels=in_ch, num_classes=n_classes)
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in sd.items()}, strict=False
        )
    model.eval()
    _audio_model = model
    return _audio_model, _audio_le, _audio_meta


# ── Helpers ───────────────────────────────────────────────────────────────────
VISUAL_POSITIVE = {"happy", "surprise"}
VISUAL_NEGATIVE = {"sad", "angry", "fear", "disgust"}
TEXT_POSITIVE = {"joy", "surprise", "positive"}
TEXT_NEGATIVE = {"anger", "disgust", "fear", "sadness", "negative"}


def _polarity(label: str, modality: str) -> str:
    label = label.lower()
    pos = VISUAL_POSITIVE if modality == "visual" else TEXT_POSITIVE
    neg = VISUAL_NEGATIVE if modality == "visual" else TEXT_NEGATIVE
    if label in pos:
        return "positive"
    if label in neg:
        return "negative"
    return "neutral"


# ── Inference ─────────────────────────────────────────────────────────────────


def run_visual_inference(image: Image.Image) -> Dict[str, Any]:
    """Try local Keras model; fallback to ViT HF pipeline."""
    # Try local Keras model
    local_path = next((p for p in LOCAL_VISUAL_MODEL_PATHS if p.exists()), None)
    if local_path is not None and local_path.suffix == ".h5":
        try:
            import tensorflow as tf  # type: ignore

            def _preprocess(img_pil: Image.Image, size: int = 224) -> np.ndarray:
                img = img_pil.convert("RGB")
                w, h = img.size
                s = min(w, h)
                left = (w - s) // 2
                top = (h - s) // 2
                img = img.crop((left, top, left + s, top + s)).resize((size, size))
                arr = np.asarray(img).astype("float32") / 255.0
                return arr

            img_arr = _preprocess(image)
            model = tf.keras.models.load_model(str(local_path), compile=False)
            preds = model.predict(np.expand_dims(img_arr, 0), verbose=0)
            scores = preds[0].astype(float)
            classes = [
                "angry",
                "disgust",
                "fear",
                "happy",
                "neutral",
                "sad",
                "surprise",
            ]
            top_idx = int(np.argmax(scores))
            top_label = classes[top_idx]
            probs = {
                classes[i]: round(float(scores[i]) * 100, 1)
                for i in range(len(classes))
            }
            return {
                "top_label": top_label,
                "top_score": probs[top_label],
                "all_scores": probs,
            }
        except Exception:
            pass

    # Fallback to HF pipeline
    pipe = _load_visual_pipe()
    results = pipe(image)
    top = max(results, key=lambda x: x["score"])
    return {
        "top_label": top["label"].lower(),
        "top_score": round(top["score"] * 100, 1),
        "all_scores": {r["label"].lower(): round(r["score"] * 100, 1) for r in results},
    }


def run_text_inference(text: str) -> Dict[str, Any]:
    pipe = _load_text_pipe()
    results = pipe(text)[0]
    # Aggregate scores into positive, neutral, negative
    scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for r in results:
        polarity = _polarity(r["label"], "text")
        scores[polarity] += r["score"]
    # Find the top label and score
    top_label = max(scores, key=scores.get)
    top_score = round(scores[top_label] * 100, 1)
    # Normalize scores to add up to 100% for the chart
    total_score = sum(scores.values())
    all_scores = {k: round((v / total_score) * 100, 1) for k, v in scores.items()}

    return {
        "top_label": top_label,
        "top_score": top_score,
        "all_scores": all_scores,
    }


def fuse(
    visual_result: dict,
    text_result: dict,
    audio_result: dict = None,
    visual_weight: float = 0.55,
    audio_weight: float = 0.2,
) -> Dict[str, Any]:
    v_pol = _polarity(visual_result["top_label"], "visual")
    t_pol = _polarity(text_result["top_label"], "text")
    a_pol = _polarity(audio_result["top_label"], "audio") if audio_result else "neutral"

    mismatch = (v_pol != t_pol or v_pol != a_pol or t_pol != a_pol) and (
        "neutral" not in (v_pol, t_pol, a_pol)
    )

    t_weight = 1.0 - visual_weight - audio_weight
    v_conf = float(visual_result["top_score"]) / 100.0
    t_conf = float(text_result["top_score"]) / 100.0
    a_conf = float(audio_result["top_score"]) / 100.0 if audio_result else 0.0

    fused_conf = round(
        (visual_weight * v_conf + t_weight * t_conf + audio_weight * a_conf) * 100, 1
    )

    dominant = visual_result["top_label"]
    if t_conf > v_conf and t_conf > a_conf:
        dominant = text_result["top_label"]
    elif a_conf > v_conf and a_conf > t_conf:
        dominant = audio_result["top_label"]

    return {
        "mismatch": mismatch,
        "visual_polarity": v_pol,
        "text_polarity": t_pol,
        "audio_polarity": a_pol,
        "dominant_emotion": dominant,
        "fused_confidence": fused_conf,
        # optional fields to support donut chart
        "visual_confidence": round(v_conf * 100, 1),
        "text_confidence": round(t_conf * 100, 1),
        "audio_confidence": round(a_conf * 100, 1),
    }


def run_audio_inference(audio_bytes: bytes) -> Dict[str, Any]:
    """Run audio model inference using saved PyTorch checkpoint.

    Returns dict with keys: top_label, top_score (percent), all_scores.
    """
    model, le, meta = _load_audio_assets()
    # Decode audio bytes; try SoundFile first, then librosa via temp file
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            try:
                data, sr = librosa.load(tmp.name, sr=None, mono=False)
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail="Unsupported or invalid audio file"
                ) from exc
    feat = _audio_extract_features(
        np.asarray(data),
        sr,
        target_sr=int(meta.get("sample_rate", 16000)),
        n_mfcc=40,
        n_mels=64,
        hop_length=512,
        n_fft=2048,
        max_duration_s=float(meta.get("max_duration_s", 3.0)),
    )
    inp = torch.tensor(feat).unsqueeze(0)  # (1, C, T)
    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1).squeeze().cpu().numpy()
    classes: List[str] = list(getattr(le, "classes_", meta.get("emotion_classes", [])))
    if not classes:
        classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    top_idx = int(np.argmax(probs))
    top_label = classes[top_idx]
    all_scores = {
        classes[i]: round(float(probs[i] * 100.0), 1) for i in range(len(classes))
    }
    return {
        "top_label": top_label,
        "top_score": all_scores[top_label],
        "all_scores": all_scores,
    }


# ── Video inference helpers ─────────────────────────────────────────────────


class TemporalAttentionPool(nn.Module):
    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(3, 4))
        x = x.permute(0, 2, 1)
        w = torch.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1)


class VideoEmotionModel(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        backbone_r3d = r3d_18(weights=None)
        self.backbone = nn.Sequential(*list(backbone_r3d.children())[:-2])
        self.temp_attn = TemporalAttentionPool(512)
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        pooled = self.temp_attn(feats)
        return self.head(pooled)


def _load_video_assets():
    global _video_model, _video_le, _video_meta
    if _video_model is not None and _video_le is not None and _video_meta is not None:
        return _video_model, _video_le, _video_meta
    meta_path = VIDEO_DIR / "video_model_meta.json"
    le_path = VIDEO_DIR / "label_encoder.pkl"
    ckpt_path = VIDEO_DIR / "best_video_model.pt"
    if not (meta_path.exists() and le_path.exists() and ckpt_path.exists()):
        raise RuntimeError(
            "Video model artefacts missing in saved_models/video_emotion"
        )
    with open(meta_path) as f:
        _video_meta = json.load(f)
    import pickle

    with open(le_path, "rb") as f:
        _video_le = pickle.load(f)
    n_classes = int(
        _video_meta.get("num_classes", len(getattr(_video_le, "classes_", [])) or 7)
    )
    model = VideoEmotionModel(num_classes=n_classes)
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    remapped = {}
    for k, v in sd.items():
        nk = k.replace("module.", "")
        nk = nk.replace("features.", "backbone.")
        nk = nk.replace("pool.", "temp_attn.")
        nk = nk.replace("classifier.", "head.")
        remapped[nk] = v
    model.load_state_dict(remapped, strict=False)
    model.eval()
    _video_model = model
    return _video_model, _video_le, _video_meta


def _build_video_transform(frame_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((frame_size, frame_size)),
            T.CenterCrop(frame_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _video_decode_frames_cv2(path: str, num_frames: int) -> List[np.ndarray]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(path)
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return [frames[i] for i in idx]


def run_video_inference(video_bytes: bytes) -> Dict[str, Any]:
    model, le, meta = _load_video_assets()
    num_frames = int(meta.get("num_frames", 16))
    size = int(meta.get("img_size", 112))
    transform = _build_video_transform(size)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        frames = _video_decode_frames_cv2(tmp.name, num_frames)
    if not frames:
        raise HTTPException(status_code=400, detail="Invalid or unreadable video file")
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    sampled = [frames[i] for i in indices]
    while len(sampled) < num_frames:
        sampled.append(sampled[-1])
    tensors = [transform(f) for f in sampled]
    clip = torch.stack(tensors, dim=1).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(clip), dim=1).squeeze().cpu().numpy()
    classes: List[str] = list(getattr(le, "classes_", meta.get("emotion_classes", [])))
    if not classes:
        classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    top_idx = int(np.argmax(probs))
    top_label = classes[top_idx]
    all_scores = {
        classes[i]: round(float(probs[i] * 100.0), 1) for i in range(len(classes))
    }
    return {
        "top_label": top_label,
        "top_score": all_scores[top_label],
        "all_scores": all_scores,
    }


def generate_summary(
    visual_result: dict, text_result: dict, fusion: dict, text_input: str
) -> str:
    v_label = str(visual_result["top_label"]).capitalize()
    v_score = visual_result["top_score"]
    t_label = str(text_result["top_label"]).capitalize()
    t_score = text_result["top_score"]
    mismatch = bool(fusion["mismatch"])  # noqa: F841 (kept for clarity)

    if fusion["mismatch"]:
        return (
            f"Despite expressing a {t_label.lower()} emotional tone verbally "
            f"(confidence: {t_score}%), the speaker's facial cues indicate "
            f"{v_label.lower()} (confidence: {v_score}%). "
            f"This incongruence between verbal and non-verbal signals is worth noting — "
            f"the individual may be masking or suppressing their true emotional state."
        )
    else:
        return (
            f"Both facial expression ({v_label}, {v_score}% confidence) and verbal tone "
            f"({t_label}, {t_score}% confidence) are aligned, suggesting a consistent "
            f"emotional state of {fusion['dominant_emotion'].lower()}. "
            f"No significant incongruence was detected between modalities."
        )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    text: str = Form(...),
    visual_weight: float = Form(0.55),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        content = await image.read()
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    visual_result = run_visual_inference(pil)
    text_result = run_text_inference(text)
    fusion_result = fuse(
        visual_result=visual_result,
        text_result=text_result,
        visual_weight=visual_weight,
    )
    summary = generate_summary(visual_result, text_result, fusion_result, text)

    return {
        "visual": visual_result,
        "text": text_result,
        "fusion": fusion_result,
        "summary": summary,
    }


@app.post("/analyze_audio")
async def analyze_audio(audio: UploadFile = File(...), text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        content = await audio.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio file")

    audio_result = run_audio_inference(content)
    text_result = run_text_inference(text)

    fusion_result = fuse(
        visual_result={
            "top_label": audio_result["top_label"],
            "top_score": audio_result["top_score"],
            "all_scores": audio_result["all_scores"],
        },
        text_result=text_result,
        audio_result=audio_result,
        visual_weight=0.0,
        audio_weight=0.5,
    )
    summary = generate_summary(
        {
            "top_label": audio_result["top_label"],
            "top_score": audio_result["top_score"],
            "all_scores": audio_result["all_scores"],
        },
        text_result,
        fusion_result,
        text,
    )

    return {
        "audio": audio_result,
        "text": text_result,
        "fusion": fusion_result,
        "summary": summary,
    }


@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(...),
    text: str = Form(...),
    visual_weight: float = Form(0.55),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        content = await video.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid video file")

    visual_result = run_video_inference(content)
    text_result = run_text_inference(text)
    fusion_result = fuse(
        visual_result=visual_result,
        text_result=text_result,
        visual_weight=visual_weight,
    )
    summary = generate_summary(visual_result, text_result, fusion_result, text)
    return {
        "visual": visual_result,
        "text": text_result,
        "fusion": fusion_result,
        "summary": summary,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
